"""
Indexing Orchestrator — coordinates batch document ingestion.
"""

import os
import re
from typing import Optional, List, Dict, Any

from src.services.container import container
from src.indexer import Indexer
from src import console as con


def run_batch_index(
    target: str,
    use_llm: bool,
    trace: bool,
    cloud: bool,
    chunk_pool_size: Optional[int] = None,
    pdf_parser_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Orchestrates dependency loading and target splitting for batch ingestion.
    Returns session trace dictionaries.
    """
    if cloud:
        os.environ["SCIENCE_GRAPH_USE_CLOUD"] = "1"
    if trace:
        con.SHOW_TIME = True

    graph_repo = container.get_graph_repo()
    vector_repo = container.get_vector_repo()
    embedding_engine = container.get_embedding_engine()

    llm_engine = None
    if use_llm:
        try:
            llm_engine = container.get_llm_engine(use_cloud=cloud)
        except Exception as e:
            con.warning(f"Could not load LLM engine: {e}")

    if use_llm and not llm_engine:
        con.warning("Proceeding with regex fallback extraction because LLM engine failed to load.")

    indexer = Indexer(graph_repo, vector_repo, embedding_engine, llm_engine)

    raw_targets = re.split(r'[,;]', target)
    targets = [t.strip() for t in raw_targets if t.strip()]

    if not targets:
        raise ValueError("No targets provided to index.")

    kwargs = {}
    if pdf_parser_type is not None:
        kwargs["pdf_parser_type"] = pdf_parser_type

    return indexer.index_batch(
        targets=targets,
        use_llm=use_llm,
        trace=trace,
        chunk_pool_size=chunk_pool_size,
        **kwargs
    )
