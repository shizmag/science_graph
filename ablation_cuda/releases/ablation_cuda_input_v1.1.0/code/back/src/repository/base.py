from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
from src.models import Paper, Author, Concept, Chunk
from dataclasses import dataclass

@dataclass
class ResolvedPaperNode:
    original_node_id: str
    canonical_paper_id: str | None
    node_type: str | None
    exists_in_papers_table: bool
    chunks_count: int
    is_placeholder: bool
    source_relation_type: str | None = None

class GraphRepository(ABC):
    @abstractmethod
    def get_papers_mentioning_concepts(self, concept_ids: List[str]) -> List[Tuple[str, str]]:
        """Retrieves papers that mention one or more of the given concepts as (paper_id, title) tuples."""
        pass

    @abstractmethod
    def get_concepts_for_papers(self, paper_ids: List[str]) -> List[Tuple[str, str, str]]:
        """Retrieves concepts associated with papers as (paper_id, concept_id, concept_name) tuples."""
        pass

    @abstractmethod
    def get_concept_document_frequencies(self, concept_ids: List[str]) -> Dict[str, int]:
        """Retrieves document frequency of concepts (i.e. number of distinct papers mentioning each concept)."""
        pass

    @abstractmethod
    def get_total_paper_count(self) -> int:
        """Retrieves total count of paper nodes in the database."""
        pass

    @abstractmethod
    def get_citation_neighbors(self, paper_ids: List[str]) -> List[Tuple[str, str, str, str]]:
        """Retrieves citation neighbors for given papers as (seed_id, candidate_id, direction, candidate_title) tuples."""
        pass

    @abstractmethod
    def search_chunks_within_papers(self, query_embedding: List[float], paper_ids: List[str], limit_per_paper: int = 1) -> List[Tuple[Chunk, float]]:
        """Performs scoped vector search within selected paper IDs and returns the best chunks for each paper."""
        pass

    @abstractmethod
    def get_neighbor_papers(self, seed_paper_ids: List[str], order: int = 2, allowed_edge_types: List[str] = None) -> List[str]:
        """Retrieves neighboring papers up to a given depth (order) from the seed papers, excluding the seed papers themselves."""
        pass

    @abstractmethod
    def resolve_graph_nodes_to_local_papers(self, node_ids: List[str]) -> List[ResolvedPaperNode]:
        """Resolves node IDs to canonical local paper IDs with metadata."""
        pass

    @abstractmethod
    def get_chunks_count_by_paper_ids(self, paper_ids: List[str]) -> Dict[str, int]:
        """Retrieves mapping of paper_id -> chunks count."""
        pass

    @abstractmethod
    def filter_papers_with_chunks(self, paper_ids: List[str]) -> List[str]:
        """Filters paper IDs to keep only those that have parsed chunks in the database."""
        pass

    @abstractmethod
    def count_total_local_papers(self) -> int:
        """Retrieves total count of non-placeholder paper nodes in the database."""
        pass

    @abstractmethod
    def chunks_count(self, paper_id: str) -> int:
        """Retrieves count of chunks for a paper."""
        pass

    @abstractmethod
    def has_chunks(self, paper_id: str) -> bool:
        """Checks if a paper has chunks."""
        pass

    @abstractmethod
    def is_local_indexed_paper(self, paper_id: str) -> bool:
        """Checks if a paper is locally indexed."""
        pass

    @abstractmethod
    def is_bridge_only_paper(self, paper_id: str) -> bool:
        """Checks if a paper is bridge-only."""
        pass

    @abstractmethod
    def is_retrieval_candidate_paper(self, paper_id: str) -> bool:
        """Checks if a paper is a retrieval candidate."""
        pass

    @abstractmethod
    def get_derived_bridge_edges(self, seed_paper_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieves derived bridge edges from the database view."""
        pass

    @abstractmethod
    def get_concept_idf(self, concept_ids: List[str]) -> Dict[str, float]:
        """Retrieves mapping of concept_id -> IDF score."""
        pass

    @abstractmethod
    def save_paper(self, paper: Paper) -> None:
        """Saves a paper node to the database."""
        pass

    @abstractmethod
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Retrieves a paper by its ID."""
        pass

    @abstractmethod
    def get_papers_batch(self, paper_ids: List[str]) -> Dict[str, Paper]:
        """Retrieves a dictionary of paper_id -> Paper for the requested paper IDs."""
        pass

    @abstractmethod
    def find_paper_by_title(self, title: str) -> Optional[Paper]:
        """Finds a paper or note by its exact or case-insensitive title."""
        pass

    @abstractmethod
    def find_paper_by_doi(self, doi: str) -> Optional[Paper]:
        """Finds a paper by its DOI."""
        pass

    @abstractmethod
    def find_paper_by_content_hash(self, content_hash: str) -> Optional[Paper]:
        """Finds a paper by its content hash."""
        pass

    @abstractmethod
    def save_author(self, author: Author) -> None:
        """Saves an author node to the database."""
        pass

    @abstractmethod
    def get_author(self, author_id: str) -> Optional[Author]:
        """Retrieves an author by ID."""
        pass

    @abstractmethod
    def save_concept(self, concept: Concept) -> None:
        """Saves a concept node to the database."""
        pass

    @abstractmethod
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Retrieves a concept by ID."""
        pass

    @abstractmethod
    def add_edge(self, source_id: str, target_id: str, edge_type: str, properties: Dict[str, Any] = None) -> None:
        """Adds a directed edge between two nodes."""
        pass

    @abstractmethod
    def get_neighbors(self, node_id: str, max_depth: int = 1, allowed_edge_types: List[str] = None) -> List[tuple[str, str, str, str, str, str]]:
        """
        Returns connections from node_id.
        Each connection is a tuple: (src_id, src_label, edge_type, target_id, target_label, edge_properties_json)
        """
        pass

    def get_neighbors_batch(self, node_ids: List[str]) -> List[tuple[str, str, str, str, str, str]]:
        """
        Returns direct (depth 1) connections from multiple node_ids in a single batch query.
        Each connection is a tuple: (src_id, src_label, edge_type, target_id, target_label, edge_properties_json)
        """
        results = []
        for node_id in node_ids:
            results.extend(self.get_neighbors(node_id, max_depth=1))
        return results

    @abstractmethod
    def get_stats(self) -> Dict[str, int]:
        """Returns statistical numbers of nodes and edges."""
        pass

    @abstractmethod
    def cleanup_orphaned_concepts(self) -> int:
        """Deletes Concept nodes with degree 0 and returns the number of deleted nodes."""
        pass

    @abstractmethod
    def get_all_nodes(self) -> List[tuple[str, str, str]]:
        """Retrieves all node rows as (id, label, properties)."""
        pass

    @abstractmethod
    def get_all_edges(self) -> List[tuple[str, str, str, str]]:
        """Retrieves all edge rows as (source_id, target_id, type, properties)."""
        pass

    @abstractmethod
    def get_node_by_id(self, node_id: str) -> Optional[tuple[str, str]]:
        """Retrieves label and properties for a node ID."""
        pass

    @abstractmethod
    def get_papers_by_author(self, author_id: str) -> List[Paper]:
        """Retrieves all papers written by a given author ID."""
        pass

    @abstractmethod
    def get_papers_by_entity(self, entity_id: str, edge_type: str) -> List[Paper]:
        """Retrieves all papers connected to a specific entity with a specific edge type."""
        pass

    @abstractmethod
    def get_distinct_targets(self, source_ids: List[str], edge_type: str) -> List[tuple[str, str]]:
        """Retrieves distinct target IDs and target properties for source IDs and an edge type."""
        pass

    @abstractmethod
    def search_papers_by_title(self, query: str, limit: int = 20) -> List[Paper]:
        """Searches papers by title (case-insensitive) using the title column."""
        pass

    @abstractmethod
    def get_notes(self) -> List[Paper]:
        """Retrieves all nodes that represent notes (source_type = 'note')."""
        pass

    @abstractmethod
    def delete_edges_by_target(self, target_id: str, edge_types: List[str]) -> None:
        """Deletes all edges pointing TO target_id with one of the given edge types."""
        pass

    @abstractmethod
    def delete_edges_by_source(self, source_id: str, edge_types: List[str]) -> None:
        """Deletes all edges originating FROM source_id with one of the given edge types."""
        pass

    @abstractmethod
    def delete_node(self, node_id: str) -> None:
        """Deletes a node (and, via foreign key cascade, its related edges and chunks)."""
        pass

    @abstractmethod
    def get_paper_ids(self) -> List[str]:
        """Retrieves all paper IDs in the graph."""
        pass

    @abstractmethod
    def get_non_placeholder_paper_ids(self) -> List[str]:
        """Retrieves all paper IDs that are not placeholders."""
        pass

    @abstractmethod
    def get_paper_source_types(self) -> Dict[str, str]:
        """Retrieves a dictionary mapping paper/note ID to its source type (from properties)."""
        pass

    @abstractmethod
    def get_browse_rows(self, table: str, page: int, limit: int, search_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Paginated search/retrieval of nodes or relations for CLI/UI browsing."""
        pass

    @abstractmethod
    def get_browse_count(self, table: str, search_query: Optional[str] = None) -> int:
        """Get the count of nodes or relations for CLI/UI browsing."""
        pass

    @abstractmethod
    def update_node_properties(self, node_id: str, properties: Dict[str, Any]) -> None:
        """Updates specific properties of a node in the graph."""
        pass

    @abstractmethod
    def get_concept_aliases(self) -> Dict[str, str]:
        """Retrieves a mapping of lowercased alias name to canonical concept name."""
        pass

    @abstractmethod
    def find_paper_by_arxiv(self, arxiv_id: str) -> Optional[Paper]:
        """Finds a paper by its arXiv ID."""
        pass

    @abstractmethod
    def find_paper_by_url(self, url: str) -> Optional[Paper]:
        """Finds a paper by its URL."""
        pass

    @abstractmethod
    def get_non_placeholder_papers(self) -> List[Paper]:
        """Retrieves all non-placeholder papers."""
        pass

    @abstractmethod
    def save_reference_corpus_stats(self, stats: List[Tuple[str, int, float, int, str]]) -> None:
        """Saves reference corpus frequency stats."""
        pass

    @abstractmethod
    def save_paper_reference_vectors(self, vectors: List[Tuple[str, str, float]]) -> None:
        """Saves paper reference vectors."""
        pass

    @abstractmethod
    def save_chunk_reference_mentions(self, mentions: List[Tuple[str, str, str, Optional[str], Optional[str], Optional[int], Optional[str], Optional[str]]]) -> None:
        """Saves chunk-level reference context mentions."""
        pass

    @abstractmethod
    def delete_derived_edges_by_types(self, edge_types: List[str]) -> None:
        """Deletes all edges of the specified types."""
        pass

    @abstractmethod
    def delete_derived_edges_for_papers(self, paper_ids: List[str], edge_types: List[str]) -> None:
        """Deletes derived edges for specific paper IDs."""
        pass

    @abstractmethod
    def get_reference_corpus_stats(self) -> Dict[str, Dict[str, Any]]:
        """Retrieves reference corpus stats mapping work_id -> stats dict."""
        pass

    @abstractmethod
    def get_paper_reference_vectors(self) -> Dict[str, Dict[str, float]]:
        """Retrieves paper reference vectors mapping paper_id -> {work_id -> weight}."""
        pass

    @abstractmethod
    def get_chunk_reference_mentions(self) -> List[Dict[str, Any]]:
        """Retrieves all chunk-level reference mentions."""
        pass

    @abstractmethod
    def remap_external_work_to_local_paper(self, ext_work_id: str, local_paper_id: str) -> None:
        """Remaps all edges and mentions pointing to ext_work_id to point to local_paper_id, and deletes ext_work_id."""
        pass

    @abstractmethod
    def delete_chunk_reference_mentions_for_paper(self, paper_id: str) -> None:
        """Deletes all chunk reference mentions for the given paper ID."""
        pass

    @abstractmethod
    def delete_chunk_nodes_for_paper(self, paper_id: str) -> None:
        """Deletes all nodes with label 'Chunk' that belong to the given paper ID."""
        pass

    @abstractmethod
    def get_nodes_by_label(self, label: str) -> List[tuple[str, Dict[str, Any]]]:
        """Retrieves all nodes with a given label as list of (node_id, properties)."""
        pass

    @abstractmethod
    def get_node_properties(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves properties for any node by its ID."""
        pass

    def save_nodes_bulk(self, nodes: List[tuple[str, str, Dict[str, Any]]]) -> None:
        """Saves multiple nodes in bulk."""
        pass

    def save_edges_bulk(self, edges: List[tuple[str, str, str, Dict[str, Any]]]) -> None:
        """Saves multiple edges in bulk."""
        for src, target, edge_type, props in edges:
            self.add_edge(src, target, edge_type, props)

    @contextmanager
    def transaction(self):
        """Context manager for running multiple operations in a single transaction."""
        yield




class VectorRepository(ABC):
    @abstractmethod
    def save_chunks(self, chunks: List[Chunk]) -> None:
        """Saves text chunks and their embeddings."""
        pass

    def save_chunks_bulk(self, chunks: List[Chunk]) -> None:
        """Saves text chunks and their embeddings in bulk."""
        self.save_chunks(chunks)

    @abstractmethod
    def search_similar_chunks(self, query_embedding: List[float], limit: int = 5, filters: Optional[dict] = None) -> List[tuple[Chunk, float]]:
        """
        Searches for similar chunks using cosine similarity.
        Returns a list of tuples (Chunk, score).
        """
        pass

    @abstractmethod
    def search_text_fts5(self, query: str, limit: int = 10, filters: Optional[dict] = None) -> List[tuple[Chunk, float]]:
        """
        Searches for chunks using SQLite FTS5 BM25 search.
        Returns a list of tuples (Chunk, score).
        """
        pass

    @abstractmethod
    def get_chunks_for_paper(self, paper_id: str) -> List[Chunk]:
        """Retrieves all text chunks belonging to a paper."""
        pass

    @abstractmethod
    def get_all_chunks(self) -> List[Chunk]:
        """Retrieves all chunks from the database."""
        pass
