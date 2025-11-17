import os

current_path = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "embed_model": "nomic-embed-text",
    "company_data": os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rag_ready_faqs_data.jsonl"),
    "kb_vectorstore_path": os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb_chroma_db"),

    "rag_chain": {
    # --- Retriever settings ---   
    "retriever_common": {
        # Lower score = better match (cosine distance)
        "relevance_threshold": 0.75,
        # Optional: Enable redundant filter
        "enable_redundancy_filter": True,
        "redundancy_threshold": 0.95, 
        # Optional: Enable cross-encoder reranking
        "enable_cross_encoder": True,
        # How many docs to pass to the cross-encoder
        "top_n_before_cross_encoder": 10 
        },

    "retriever_knowledge": {
        # Initial net to cast for articles/FAQs
        "k_initial": 10,
        # Final number of docs to send to LLM
        "max_docs_to_llm": 3 
        },
    }
}