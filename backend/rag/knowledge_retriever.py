import os
import json
import logging
import ollama
# from langchain.prompts import PromptTemplate
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from typing import List

# --- Assumed Imports from your project ---
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.config import CONFIG
from rag.decode_placeholders import format_youtube_link, restore_placeholders
from dotenv import load_dotenv
# You can uncomment this if/when you add a cross-encoder
from rag.cross_encoder_reranker import CrossEncoderReranker
# ---

logger = logging.getLogger(__name__)
load_dotenv()

# --- CONFIGURATION (Reads from environment variable for Docker) ---
ollama_client = ollama.Client(host=os.environ.get("OLLAMA_BASE_URL"))
embeddings = OllamaEmbeddings(model=CONFIG["embed_model"], base_url=os.getenv("OLLAMA_BASE_URL"))  # , num_ctx=2048

# --- INITIALIZATION ---
reranker = CrossEncoderReranker()


def retrieve_relevant_documents(
    question: str,
    vectorstore,
    k_value,
    filter,
    relevance_threshold = CONFIG["rag_chain"]["retriever_common"].get("relevance_threshold", 0.7),
    type_of_docs = "article"
) -> List[Document]:
    
    """
    Retrieves and filters candidate documents from the single vector store.
    """
    try:
        # Filter out products
        results = vectorstore.similarity_search_with_score(
            question, 
            k=k_value,
            filter=filter
        )
    except Exception as e:
        logger.error(f"Error retrieving from vectorstore: {e}")
        return {"answer": "I encountered an error while searching my knowledge base.", "references": "", "youtube_tutorials": ""}

    candidate_docs = []
    for doc, score in results:
        if score < relevance_threshold: 
            doc.metadata["similarity_score"] = score
            candidate_docs.append(doc)
    
    if not candidate_docs:
        if type_of_docs == "article":
            logger.warning("No documents found after initial retrieval.")
            return {"answer": "I couldn't find any relevant articles for that topic.", "references": "", "youtube_tutorials": ""}
        elif type_of_docs == "product":
            logger.warning("No products found after initial retrieval.")
            return {"answer": "I couldn't find any specific products matching that description.", "references": "", "youtube_tutorials": ""}
    return candidate_docs


# --- 1. THIS IS YOUR NEW, LIGHTWEIGHT "RETRIEVAL ENGINE" ---
def retrieve_and_rerank_knowledge(
    question: str,
    vectorstore,
    key_value = CONFIG["rag_chain"]["retriever_knowledge"].get("k_initial", 10),
    filter = {"doc_type": "faq"},
    relevance_threshold = CONFIG["rag_chain"]["retriever_common"].get("relevance_threshold", 0.7),
    use_cross_encoder=CONFIG["rag_chain"]["retriever_common"]["enable_cross_encoder"],
    use_redundant_filter=CONFIG["rag_chain"]["retriever_common"]["enable_redundancy_filter"]
):
    """
    Core retrieval function: Fetches, reranks, and formats context and links
    WITHOUT generating an LLM answer.
    """
    logger.info(f"Core Knowledge Retrieval for: '{question}'")
    
    # 1. RETRIEVE
    ## Cosine distance: 0.0 = perfect match, 2.0 = perfect mismatch.
    candidate_docs = retrieve_relevant_documents(question, vectorstore, key_value, filter, relevance_threshold)
    if isinstance(candidate_docs, dict):
        return candidate_docs # Return error dict
    if not candidate_docs:
        return {"debug": "No documents found after initial retrieval."}

    # 2. RERANK
    if use_cross_encoder:
        logger.info("Applying Cross-Encoder reranking...")
        top_n_for_cross_encoder = CONFIG["rag_chain"]["retriever_common"]["top_n_before_cross_encoder"]
        reranked_docs = reranker.rerank(question, candidate_docs[:top_n_for_cross_encoder])
    else:
        reranked_docs = sorted(candidate_docs, key=lambda x: x.metadata["similarity_score"])

    if not reranked_docs:
        return {"debug": "No documents found after reranking."}

    # 3. FILTER & MERGE
    max_docs_to_llm = CONFIG["rag_chain"]["retriever_knowledge"].get("max_docs_to_llm", 3)
    top_docs_before_filter = reranked_docs[:max_docs_to_llm]

    if use_redundant_filter:
        redundancy_threshold = CONFIG["rag_chain"]["retriever_common"]["redundancy_threshold"]
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings, similarity_threshold=redundancy_threshold)
        top_docs = redundant_filter.transform_documents(top_docs_before_filter)
    else:
        top_docs = top_docs_before_filter

    if not top_docs:
        return {"debug": "All relevant docs were filtered as redundant."}

    # 4. GATHER CONTEXT
    best_doc = top_docs[0]

    # If it came as (Document, score) tuple, extract the first element
    if isinstance(best_doc, tuple):
        best_doc = best_doc[0]

    context_parts = []
    all_references = set()
    all_youtube_links = {}

    # Now 'best_doc' is a Document
    if isinstance(best_doc, Document):
        guide_images_map = {}
        if "guide_images" in best_doc.metadata:
            try:
                guide_images_map = json.loads(best_doc.metadata["guide_images"])
            except json.JSONDecodeError:
                pass

        content = restore_placeholders(best_doc.page_content, guide_images_map)
        context_parts.append(content)

        title = best_doc.metadata.get('title', 'Article')
        source_url = best_doc.metadata.get('source', '#')
        if source_url and source_url != "#":
            all_references.add(f"- [{title}]({source_url})")
        
        youtube_link = format_youtube_link(best_doc)
        if youtube_link:
            all_youtube_links[title] = youtube_link
    else:
        logger.warning("Top document is not a Document object.")

    formatted_context = "\n\n---\n\n".join(context_parts)
    

    # 7. RETURN THE CONTEXT AND LINKS (NO LLM CALL)
    references_text = "\n".join(sorted(list(all_references)))
    youtube_references = "\n".join(f"- 🎥 {title}: {link}" for title, link in all_youtube_links.items())

    return {
        "formatted_context": formatted_context,
        # "final_prompt": final_prompt,
        "references": references_text,
        "youtube_tutorials": youtube_references,
    }


if __name__ == "__main__":
    import logging
    from rag.loader import load_or_create_vectorstore

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    
    logger = logging.getLogger(__name__)

    # --- 1. Load or create the vectorstore ---
    vectorstore = load_or_create_vectorstore()
    if not vectorstore:
        logger.error("Vectorstore could not be loaded or created.")
        exit(1)
    
    logger.info(f"Vectorstore loaded: {type(vectorstore)}")

    # --- 2. Test a sample question ---
    sample_question = "What is AI Server?"
    result = retrieve_and_rerank_knowledge(sample_question, vectorstore)

    if isinstance(result, dict):
        logger.info("Retrieved result keys: " + ", ".join(result.keys()))
        for k, v in result.items():
            print(f"\n{k}:\n{v}")
    else:
        logger.warning("Unexpected result type: %s", type(result))
