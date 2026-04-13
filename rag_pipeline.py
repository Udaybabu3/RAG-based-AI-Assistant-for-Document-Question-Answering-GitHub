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
@dataclass
class RAGConfig:
    chunk_size: int = 500
    chunk_overlap: int = 100
    similarity_threshold: float = 2.0
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
        

    def query(self, query: str) -> Dict:
        results = self.vector_store.similarity_search_with_score(
            query, k=self.config.top_k
        )

        # fallback
        if results and results[0][1] > self.config.similarity_threshold:
            keywords = self.keyword_extractor.extract_keywords(query)
            suggestions = self.resource_provider.suggest_resources(keywords)
            print("DEBUG SCORES:", [score for _, score in results])
            return {
                "type": "fallback",
                "keywords": keywords,
                "suggestions": suggestions
            }

        # relevant
        docs = [
            {
                "content": doc.page_content,
                "score": float(score)
            }
            for doc, score in results
        ]

        return {
            "type": "relevant",
            "results": docs
        }