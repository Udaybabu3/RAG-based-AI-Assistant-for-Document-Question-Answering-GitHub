import os
from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass
from typing import List, Dict

from utils.loader import load_pdf_documents
from utils.splitter import split_documents
from utils.embeddings import create_vector_store
from utils.keyword_extractor import KeywordExtractor
from utils.resource_provider import ExternalResourceProvider
from utils.fallback import fallback_response


@dataclass
class RAGConfig:
    chunk_size: int = 500
    chunk_overlap: int = 100
    # Cosine similarity threshold (0.0 - 1.0).
    # Tune based on document density:
    #   0.20 - 0.25 -> sparse docs (resumes, short reports)
    #   0.30 - 0.40 -> dense docs (textbooks, research papers)
    similarity_threshold: float = 0.25
    top_k: int = 3
    openai_api_key: str = None


class RAGPipeline:

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.config.openai_api_key = (
            self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
        )

        self.vector_store = None
        self.keyword_extractor = KeywordExtractor()
        self.resource_provider = ExternalResourceProvider()

    def load_documents(self, paths: List[str]):
        self.documents = load_pdf_documents(paths)

    def process_documents(self):
        chunks = split_documents(
            self.documents,
            self.config.chunk_size,
            self.config.chunk_overlap
        )

        self.vector_store = create_vector_store(chunks)

    def _l2_to_cosine(self, l2_squared: float) -> float:
        """
        Convert FAISS squared-L2 distance to cosine similarity.

        all-MiniLM-L6-v2 produces unit-normalized embeddings, so:
            ||a - b||² = 2 - 2·cos(a, b)
            cos(a, b)  = 1 - (||a - b||² / 2)
        """
        return 1.0 - (l2_squared / 2.0)

    def query(self, query: str) -> Dict:
        results = self.vector_store.similarity_search_with_score(
            query, k=self.config.top_k
        )

        # ── Score conversion & out-of-scope detection ──────────────
        scored = [
            (doc, float(score), self._l2_to_cosine(float(score)))
            for doc, score in results
        ]

        max_similarity = max(cos for _, _, cos in scored) if scored else 0.0

        print(f"DEBUG  L2² scores : {[round(l2, 4) for _, l2, _ in scored]}")
        print(f"DEBUG  Cosine sims: {[round(cos, 4) for _, _, cos in scored]}")
        print(f"DEBUG  Max cosine : {max_similarity:.4f}  |  Threshold: {self.config.similarity_threshold}")

        # ── Fallback: query is outside document scope ──────────────
        if max_similarity < self.config.similarity_threshold:
            fb = fallback_response(query)
            # Preserve legacy keys for backward compatibility
            fb["keywords"] = self.keyword_extractor.extract_keywords(query)
            fb["suggestions"] = self.resource_provider.suggest_resources(fb["keywords"])
            fb["max_similarity"] = max_similarity
            return fb

        # ── Relevant: return matched document chunks ───────────────
        docs = [
            {
                "content": doc.page_content,
                "l2_score": l2,
                "cosine_similarity": cos,
            }
            for doc, l2, cos in scored
        ]

        return {
            "type": "relevant",
            "results": docs,
            "max_similarity": max_similarity,
        }