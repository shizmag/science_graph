"""
ServiceContainer — dependency injection container for Science Graph.
"""

import os
from typing import Optional
from src import console as con
from src.config import config
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from src.vector_search import EmbeddingEngine
from src.llm_engine import LLMEngine, BaseLLMEngine
from src.services.rag_service import RAGService
from src.services.note_service import NoteService


class ServiceContainer:
    """
    Lazy-loads and caches instances of database repositories,
    NLP engines, and core domain services.
    """

    def __init__(self):
        self._graph_repo: Optional[SQLiteGraphRepository] = None
        self._vector_repo: Optional[SQLiteVectorRepository] = None
        self._embedding_engine: Optional[EmbeddingEngine] = None

        self._llm_engine_local: Optional[BaseLLMEngine] = None
        self._llm_engine_cloud: Optional[BaseLLMEngine] = None
        self._llm_engine_rag_local: Optional[BaseLLMEngine] = None
        self._llm_engine_rag_cloud: Optional[BaseLLMEngine] = None

        self._rag_service_local: Optional[RAGService] = None
        self._rag_service_cloud: Optional[RAGService] = None
        self._note_service: Optional[NoteService] = None

    def get_graph_repo(self) -> SQLiteGraphRepository:
        if self._graph_repo is None:
            self._graph_repo = SQLiteGraphRepository(config.db_path)
        return self._graph_repo

    def get_vector_repo(self) -> SQLiteVectorRepository:
        if self._vector_repo is None:
            self._vector_repo = SQLiteVectorRepository(config.db_path)
        return self._vector_repo

    def get_embedding_engine(self) -> EmbeddingEngine:
        if self._embedding_engine is None:
            self._embedding_engine = EmbeddingEngine()
        return self._embedding_engine

    def get_llm_engine(self, use_cloud: bool = False, purpose: str = "index") -> BaseLLMEngine:
        con.debug(f"LLM_GET_ENGINE pid={os.getpid()} container_id={id(self)} use_cloud={use_cloud} purpose={purpose} config_file={getattr(config, 'config_file', None)} llm_local_model_path={config.llm_local_model_path} llm_local_rag_model_path={config.llm_local_rag_model_path}")

        if purpose == "rag":
            if use_cloud:
                if self._llm_engine_rag_cloud is None:
                    self._llm_engine_rag_cloud = LLMEngine(use_cloud=True, purpose="rag")
                    con.info(f"Created LLMEngine rag cloud: id={id(self._llm_engine_rag_cloud)}, use_cloud=True, purpose=rag")
                return self._llm_engine_rag_cloud
            else:
                if self._llm_engine_rag_local is None:
                    self._llm_engine_rag_local = LLMEngine(use_cloud=False, purpose="rag")
                    con.info(f"Created LLMEngine rag local: id={id(self._llm_engine_rag_local)}, use_cloud=False, purpose=rag")
                return self._llm_engine_rag_local
        else:
            if use_cloud:
                if self._llm_engine_cloud is None:
                    self._llm_engine_cloud = LLMEngine(use_cloud=True)
                con.model_msg(f"Created LLMEngine cloud: id={id(self._llm_engine_cloud)}, use_cloud=True, purpose=index")
                return self._llm_engine_cloud
            else:
                if self._llm_engine_local is None:
                    self._llm_engine_local = LLMEngine(use_cloud=False)
                con.model_msg(f"Created LLMEngine local: id={id(self._llm_engine_local)}, use_cloud=False, purpose=index")
                return self._llm_engine_local

    def get_rag_service(self, use_cloud: bool = False, warmup: bool = True) -> RAGService:
        if use_cloud:
            if self._rag_service_cloud is None:
                llm = self.get_llm_engine(use_cloud=True, purpose="rag")
                self._rag_service_cloud = RAGService(
                    self.get_graph_repo(),
                    self.get_vector_repo(),
                    self.get_embedding_engine(),
                    llm,
                    warmup=warmup
                )
            return self._rag_service_cloud
        else:
            if self._rag_service_local is None:
                llm = self.get_llm_engine(use_cloud=False, purpose="rag")
                self._rag_service_local = RAGService(
                    self.get_graph_repo(),
                    self.get_vector_repo(),
                    self.get_embedding_engine(),
                    llm,
                    warmup=warmup
                )
            return self._rag_service_local

    def get_note_service(self) -> NoteService:
        if self._note_service is None:
            self._note_service = NoteService(
                self.get_graph_repo(),
                self.get_vector_repo(),
                self.get_embedding_engine(),
                self.get_llm_engine(use_cloud=False)
            )
        return self._note_service


container = ServiceContainer()
