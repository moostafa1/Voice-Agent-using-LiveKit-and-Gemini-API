from typing import Literal

INSTRUCTION_TUNED_MODELS = {
    # Add other instruction-tuned models here if needed
    "bge-small-en-v1.5": {
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "Represent this document for retrieval: "
    },
    "bge-large-en-v1.5": {
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "Represent this document for retrieval: "
    },
    "e5-base-v2": {
        "query_prefix": "query: ",
        "doc_prefix": "passage: "
    },
    "e5-large-v2": {
        "query_prefix": "query: ",
        "doc_prefix": "passage: "
    },
    "mxbai-embed-large": {
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "Represent this document for retrieval: "
    }
}

def prepare_text_for_embedding(
    text: str,
    mode: Literal["query", "doc"],
    model_name: str
) -> str:
    """
    Prepares text for embedding based on whether the embedding model is instruction-tuned.
    
    Args:
        text (str): The raw text to embed.
        mode (str): "query" for user queries, "doc" for KB/document chunks.
        model_name (str): The name of the embedding model (e.g., 'qllama/bge-small-en-v1.5').
    
    Returns:
        str: The processed text with prefixes if required.
    """
    model_key = model_name.split("/")[-1]  # Strip repo owner if exists

    if model_key in INSTRUCTION_TUNED_MODELS:
        prefixes = INSTRUCTION_TUNED_MODELS[model_key]
        prefix = prefixes["query_prefix"] if mode == "query" else prefixes["doc_prefix"]
        return f"{prefix}{text.strip()}"
    
    # Default: return raw text
    return text.strip()


if __name__ == "__main__":
    # Test query
    query_text = "What is Ollama?"
    prepared_query = prepare_text_for_embedding(query_text, mode="query", model_name="mxbai-embed-large")
    print("Query prepared:\n", prepared_query)

    # Test document chunk
    doc_text = "Ollama is a local LLM server for running models efficiently."
    prepared_doc = prepare_text_for_embedding(doc_text, mode="doc", model_name="mxbai-embed-large")
    print("Doc prepared:\n", prepared_doc)
