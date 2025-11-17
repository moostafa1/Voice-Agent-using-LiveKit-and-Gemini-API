import os
import json
from langchain_ollama import OllamaEmbeddings
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma

import shutil
import logging

# import os, sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.config import CONFIG
from rag.embedding_wrapper import prepare_text_for_embedding
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)

# --- CONFIGURATION (Reads from environment variable for Docker) ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")

def load_jsonl(path: str) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(data)} items from {path}")
    logger.info(f"Data keys: {list(data[0].keys()) if data else 'No data'}")
    return data


def sanitize_metadata(docs):
    """
    Ensures all metadata values are primitives by serializing lists/dicts into JSON strings.
    """
    for doc in docs:
        clean_meta = {}
        for k, v in doc.metadata.items():
            if isinstance(v, (list, dict)):
                clean_meta[k] = json.dumps(v, ensure_ascii=False)  # serialize
            elif v is None or isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)  # fallback
        doc.metadata = clean_meta
    return docs

def load_vectorstore(persist_path=CONFIG["kb_vectorstore_path"]):
    """
    Load a previously saved Chroma vectorstore from disk.
    """
    import chromadb
    chromadb.config.Settings(anonymized_telemetry=False)  # ✅ Disable telemetry globally

    embeddings = OllamaEmbeddings(
        model=CONFIG["embed_model"],
        base_url=OLLAMA_BASE_URL
    )
    vectorstore = Chroma(persist_directory=persist_path, embedding_function=embeddings)
    return vectorstore

def load_data_for_creating_vectorstore(file_path=CONFIG["company_data"]):
    return load_jsonl(file_path)

def load_or_create_vectorstore(persist_directory=CONFIG["kb_vectorstore_path"], persist_path=CONFIG["kb_vectorstore_path"]):
    """
    Loads an existing vectorstore or creates a new one by embedding documents
    in batches, logging progress every 250 documents.
    """
    import chromadb
    chromadb.config.Settings(anonymized_telemetry=False)  # ✅ Disable telemetry globally

    # --- Check for existing, valid vectorstore ---
    if os.path.exists(persist_directory):
        logger.info(f"Detected existing vectorstore at: {persist_directory}")
        try:
            vectorstore = load_vectorstore(persist_path)
            logger.info(f"✅ Successfully loaded existing vectorstore from: {persist_path}")
            logger.info(f"Vectorstore contains {vectorstore._collection.count()} vectors.")
            return vectorstore # Return the loaded store
        except Exception as e:
            logger.error(f"❌ Failed to load existing vectorstore '{persist_path}': {e}")
            shutil.rmtree(persist_directory)
            logger.warning(f"[⚠️] Removed corrupted vectorstore directory: {persist_directory}")

    # --- Create new vectorstore ---
    logger.info(f"🚀 Creating new vectorstore at: {persist_directory}")
    model_name = CONFIG["embed_model"]
    
    # 1. Prepare documents for embedding
    docs_dicts = load_data_for_creating_vectorstore()
    if not docs_dicts:
        logger.error("No documents found. Aborting vectorstore creation.")
        return None
        
    docs = []
    for doc_dict in docs_dicts:
        if "page_content" in doc_dict:
            page_content = prepare_text_for_embedding(doc_dict["page_content"], "doc", model_name)
            metadata = doc_dict.get("metadata", {})  # keep metadata if exists
            docs.append(Document(page_content=page_content, metadata=metadata))
    
    if not docs:
        logger.error("Document list is empty after processing. Aborting.")
        return None

    logger.info(f"Prepared {len(docs)} total documents for embedding.")

    # ---
    # ** ADD THE FIX HERE **
    # ---
    logger.info("Sanitizing metadata for ChromaDB compatibility...")
    docs = sanitize_metadata(docs)
    logger.info("✅ Metadata sanitized.")
    # ---

    # 2. Initialize embedding function
    embeddings = OllamaEmbeddings(
        model=model_name,
        base_url=OLLAMA_BASE_URL
    )

    # 3. Initialize an empty Chroma vectorstore
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # 4. Add documents in batches with progress logging
    batch_size = 250
    total_docs = len(docs)
    
    for i in range(0, total_docs, batch_size):
        batch_docs = docs[i : i + batch_size]
        
        # This call will now succeed because the metadata is sanitized
        vectorstore.add_documents(batch_docs)
        
        # Log progress
        processed_count = min(i + batch_size, total_docs)
        logger.info(f"📊 Embedded and added {processed_count} / {total_docs} documents...")
    
    logger.info(f"✅ Created new vectorstore at: {persist_directory}")
    logger.info(f"Vectorstore contains {vectorstore._collection.count()} vectors.")
    logger.info("Vectorstore uses the cosine distance metric for similarity search.")
    return vectorstore

if __name__ == "__main__":
    vectorstore = load_or_create_vectorstore()
    if vectorstore:
        logger.info("Vectorstore is ready.")
    else:
        logger.error("Failed to load or create vectorstore.")