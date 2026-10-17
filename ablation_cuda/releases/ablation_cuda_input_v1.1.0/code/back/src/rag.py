from typing import List, Tuple, Any, Optional
from src.models import Chunk
from src.repository.base import GraphRepository, VectorRepository
from src.vector_search import EmbeddingEngine
from src.services.rag_service import RAGService

class RAGPipeline:
    def __init__(
        self,
        graph_repo: GraphRepository,
        vector_repo: VectorRepository,
        embedding_engine: EmbeddingEngine,
        llm_engine: Any
    ):
        self.service = RAGService(
            graph_repo=graph_repo,
            vector_repo=vector_repo,
            embedding_engine=embedding_engine,
            llm_engine=llm_engine
        )

    def _get_reranker(self):
        return self.service._get_reranker()

    def build_context(self, similar_chunks: List[Tuple[Chunk, float]], limit: Optional[int] = None) -> Tuple[str, str]:
        return self.service.build_context(similar_chunks, limit=limit)

    def ask(self, query: str, limit: int = 5, history_str: str = "", paper_id: Optional[str] = None, filters: Optional[dict] = None, hyde_responses: Optional[int] = None) -> str:
        return self.service.ask(query, limit, history_str, paper_id=paper_id, filters=filters, hyde_responses=hyde_responses)
