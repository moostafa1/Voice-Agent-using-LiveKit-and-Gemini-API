from sentence_transformers.cross_encoder import CrossEncoder
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # This model is small, fast, and very effective for reranking
        self.model = CrossEncoder(model_name)
        logger.info(f"Cross-encoder model '{model_name}' loaded.")

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        # Create pairs of [query, doc_content] for scoring
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get scores from the model. The scores are direct relevance probabilities.
        scores = self.model.predict(pairs)

        # Combine docs with their new scores
        scored_docs = list(zip(scores, documents))

        # Sort by score in descending order (higher score is better)
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Return the sorted documents
        logger.info(f"Reranked {len(documents)} documents. Best score: {scored_docs[0][0]:.4f}")
        return [doc for score, doc in scored_docs]