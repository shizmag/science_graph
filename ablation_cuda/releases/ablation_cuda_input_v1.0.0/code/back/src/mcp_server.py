import os
import sys
import json
import logging
from typing import List, Optional
from pathlib import Path

from fastmcp import FastMCP

# Ensure we set MCP mode if not already set, so console output is redirected to stderr
os.environ["SCIENCE_GRAPH_MCP_MODE"] = "1"

# Add parent directory of src to path to ensure proper imports if run directly
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.config import config
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from src.vector_search import EmbeddingEngine
from src.indexer import DuplicateDocumentError, Indexer
from src.llm_engine import LLMEngine
from src.services.rag_service import RAGService
from src.services.note_service import NoteService

# Suppress debug logs from fastmcp/mcp which write to stdout/stderr
logging.basicConfig(level=logging.WARNING)

mcp = FastMCP("Science Graph")

# ── Service Accessors (Singletons) ──
_graph_repo = None
_vector_repo = None
_emb_engine = None

def get_graph_repo():
    global _graph_repo
    if _graph_repo is None:
        _graph_repo = SQLiteGraphRepository(config.db_path)
    return _graph_repo

def get_vector_repo():
    global _vector_repo
    if _vector_repo is None:
        _vector_repo = SQLiteVectorRepository(config.db_path)
    return _vector_repo

def get_embedding_engine():
    global _emb_engine
    if _emb_engine is None:
        _emb_engine = EmbeddingEngine()
    return _emb_engine

def get_llm_engine(use_cloud: bool = False) -> LLMEngine:
    # Use config default if use_cloud is not specified or fallback
    use_cloud_flag = use_cloud or os.environ.get("SCIENCE_GRAPH_USE_CLOUD") == "1"
    llm = LLMEngine(use_cloud=use_cloud_flag)
    return llm

def get_rag_service(use_cloud: bool = False) -> RAGService:
    return RAGService(
        get_graph_repo(),
        get_vector_repo(),
        get_embedding_engine(),
        get_llm_engine(use_cloud)
    )

def get_note_service(use_cloud: bool = False) -> NoteService:
    return NoteService(
        get_graph_repo(),
        get_vector_repo(),
        get_embedding_engine(),
        get_llm_engine(use_cloud)
    )


# ── MCP Tools ──

@mcp.tool
def get_stats() -> dict:
    """Get database statistics including counts of papers, authors, concepts, tags, and database storage details."""
    graph_repo = get_graph_repo()
    stats = graph_repo.get_stats()
    try:
        storage_stats = config.get_storage_stats()
        stats["storage"] = storage_stats
    except Exception:
        pass
    return stats

@mcp.tool
def search_papers(query: str, limit: int = 20) -> list:
    """Quick keyword search over paper/note/book titles. Returns list of matches with ID and title."""
    graph_repo = get_graph_repo()
    papers = graph_repo.search_papers_by_title(query, limit)
    results = []
    for paper in papers:
        results.append({
            "id": paper.id,
            "title": paper.title,
            "year": paper.year,
            "source_type": paper.properties.get("source_type", "paper")
        })
    return results

@mcp.tool
def query_rag(question: str, limit: int = 5, use_cloud: bool = False,
              paper_id: Optional[str] = None, filters: Optional[dict] = None) -> str:
    """
    Query the RAG pipeline.
    Retrieves relevant text chunks and graph connections, runs them through the local (or cloud) LLM,
    and returns a cited, context-aware answer in Russian.
    """
    try:
        rag_service = get_rag_service(use_cloud=use_cloud)
        response = rag_service.ask(question, limit=limit, paper_id=paper_id, filters=filters)
        return response
    except Exception as e:
        return f"Error executing RAG query: {str(e)}"

@mcp.tool
def get_paper_details(paper_id: str) -> dict:
    """Get detailed metadata, abstract, authors, concepts, tags, and citation relationships for a specific paper or node ID."""
    graph_repo = get_graph_repo()
    node = graph_repo.get_node_by_id(paper_id)
    if not node:
        return {"error": f"Node with ID '{paper_id}' not found."}

    label, properties_json = node
    props = json.loads(properties_json or "{}")

    if label == "Paper":
        paper = graph_repo.get_paper(paper_id)
        if not paper:
            return {"error": f"Paper with ID '{paper_id}' not found."}

        neighbors = graph_repo.get_neighbors(paper_id, max_depth=1)

        concept_ids = []
        tag_ids = []
        author_ids = []
        citation_ids = []
        cited_by_ids = []

        for src_id, src_label, edge_type, tgt_id, tgt_label, edge_props_json in neighbors:
            if edge_type == "MENTIONS_CONCEPT":
                concept_ids.append(tgt_id)
            elif edge_type == "HAS_TAG":
                tag_ids.append(tgt_id)
            elif edge_type == "AUTHORED":
                author_ids.append(src_id)
            elif edge_type == "CITES" and src_id == paper_id:
                citation_ids.append(tgt_id)
            elif edge_type == "CITES" and tgt_id == paper_id:
                cited_by_ids.append(src_id)

        citations_map = graph_repo.get_papers_batch(citation_ids) if citation_ids else {}
        cited_by_map = graph_repo.get_papers_batch(cited_by_ids) if cited_by_ids else {}

        concepts = []
        for cid in concept_ids:
            c_node = graph_repo.get_node_by_id(cid)
            c_name = json.loads(c_node[1] or "{}").get("name", cid) if c_node else cid
            concepts.append({"id": cid, "name": c_name})

        tags = []
        for tid in tag_ids:
            t_node = graph_repo.get_node_by_id(tid)
            t_name = json.loads(t_node[1] or "{}").get("name", tid) if t_node else tid
            tags.append({"id": tid, "name": t_name})

        authors = paper.authors or []
        if not authors and author_ids:
            author_objs = [graph_repo.get_author(aid) for aid in author_ids]
            authors = [a.name for a in author_objs if a]

        citations = [{"id": pid, "title": p.title} for pid in citation_ids if (p := citations_map.get(pid)) and p.title]
        cited_by = [{"id": pid, "title": p.title} for pid in cited_by_ids if (p := cited_by_map.get(pid)) and p.title]

        return {
            "type": "paper",
            "id": paper.id,
            "title": paper.title,
            "authors": authors,
            "year": paper.year,
            "doi": paper.doi,
            "abstract": paper.abstract,
            "source_type": paper.properties.get("source_type", "paper"),
            "concepts": concepts,
            "tags": tags,
            "citations": citations,
            "cited_by": cited_by,
            "file_path": paper.file_path,
            "summary": paper.properties.get("summary"),
            "created_at": paper.created_at,
            "properties": paper.properties,
        }
    elif label == "Author":
        papers_list = graph_repo.get_papers_by_author(paper_id)
        papers = [{"id": p.id, "title": p.title, "source_type": p.properties.get("source_type", "paper")} for p in papers_list]
        return {
            "type": "author",
            "id": paper_id,
            "name": props.get("name", paper_id),
            "papers": papers,
            "papers_count": len(papers)
        }
    elif label == "Concept":
        is_tag = props.get("is_tag", False)
        edge_type = "HAS_TAG" if is_tag else "MENTIONS_CONCEPT"
        papers_list = graph_repo.get_papers_by_entity(paper_id, edge_type)
        papers = [{"id": p.id, "title": p.title, "source_type": p.properties.get("source_type", "paper")} for p in papers_list]
        return {
            "type": "tag" if is_tag else "concept",
            "id": paper_id,
            "name": props.get("name", paper_id),
            "description": props.get("description", f"No description available for '{props.get('name', paper_id)}'."),
            "papers": papers,
        }
    return {"id": paper_id, "label": label, "properties": props}

@mcp.tool
def index_file(file_path: str, use_cloud: bool = False) -> dict:
    """Index a local document file (PDF, EPUB, or Markdown) by its path on the host. Extracted concepts and metadata will be added to the graph."""
    expanded_path = os.path.expanduser(file_path)
    if not os.path.exists(expanded_path):
        return {"status": "error", "message": f"File not found: {file_path}"}

    suffix = Path(expanded_path).suffix.lower()
    if suffix not in {".pdf", ".md", ".epub"}:
        return {"status": "error", "message": "Only PDF, Markdown (.md), and EPUB files are supported."}

    try:
        graph_repo = get_graph_repo()
        vector_repo = get_vector_repo()
        embedding_engine = get_embedding_engine()
        llm_engine = get_llm_engine(use_cloud=use_cloud)

        indexer = Indexer(graph_repo, vector_repo, embedding_engine, llm_engine)
        if suffix == ".pdf":
            paper_id = indexer.index_pdf(expanded_path)
        elif suffix == ".md":
            paper_id = indexer.index_markdown(expanded_path)
        else:
            paper_id = indexer.index_epub(expanded_path)

        return {"status": "success", "id": paper_id, "file_path": expanded_path}
    except DuplicateDocumentError as e:
        return {"status": "error", "message": f"Duplicate document: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Indexing failed: {str(e)}"}

@mcp.tool
def index_url(url: str, use_cloud: bool = False) -> dict:
    """Index a webpage or YouTube video URL. Transcribes YouTube audio locally using Whisper, extracts text and entities, and adds them to the graph."""
    try:
        graph_repo = get_graph_repo()
        vector_repo = get_vector_repo()
        embedding_engine = get_embedding_engine()
        llm_engine = get_llm_engine(use_cloud=use_cloud)

        indexer = Indexer(graph_repo, vector_repo, embedding_engine, llm_engine)
        
        import re
        urls = [u.strip() for u in re.split(r'[,;]', url) if u.strip()]
        if not urls:
            return {"status": "error", "message": "No URLs provided."}
            
        paper_ids = []
        titles = []
        
        for u in urls:
            paper_id = indexer.index_url(u)
            paper = graph_repo.get_paper(paper_id)
            title = paper.title if paper else u
            paper_ids.append(paper_id)
            titles.append(title)
            
        return {
            "status": "success", 
            "id": ", ".join(paper_ids), 
            "title": ", ".join(titles),
            "url": url
        }
    except DuplicateDocumentError as e:
        return {"status": "error", "message": f"Duplicate document: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Indexing URL failed: {str(e)}"}

@mcp.tool
def get_notes() -> list:
    """Retrieve list of all research notes."""
    note_service = get_note_service()
    notes = note_service.get_notes()
    return notes

@mcp.tool
def create_note(title: str, content: str, authors: List[str] = None, tags: List[str] = None, use_cloud: bool = False) -> dict:
    """Create a new research note with title, markdown content, optional authors, and tags. Synthesizes concepts and adds the note to the graph."""
    try:
        note_service = get_note_service(use_cloud=use_cloud)
        paper_id, note_path = note_service.create_note(
            title,
            content,
            authors or [],
            tags or []
        )
        return {"status": "success", "id": paper_id, "file_path": note_path}
    except Exception as e:
        return {"status": "error", "message": f"Failed to create note: {str(e)}"}


# ── MCP Resources ──

@mcp.resource("graph://notes")
def list_notes_resource() -> str:
    """Retrieve a list of all research notes as a markdown index."""
    try:
        note_service = get_note_service()
        notes = note_service.get_notes()
        if not notes:
            return "# Research Notes\n\nNo research notes found in the graph."
        
        lines = ["# Research Notes Index\n", "Click on any note link to read its full content:\n"]
        for note in notes:
            lines.append(f"- **[{note['title']}](graph://notes/{note['id']})** (ID: `{note['id']}`)")
            if note.get("summary"):
                lines.append(f"  *Summary:* {note['summary']}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing notes: {str(e)}"

@mcp.resource("graph://notes/{note_id}")
def get_note_resource(note_id: str) -> str:
    """Retrieve the full content of a specific research note as markdown."""
    try:
        graph_repo = get_graph_repo()
        paper = graph_repo.get_paper(note_id)
        if not paper:
            return f"Error: Note with ID '{note_id}' not found."
            
        frontmatter = {
            "id": paper.id,
            "title": paper.title,
            "authors": paper.authors or [],
            "created_at": paper.created_at,
        }
        if paper.properties:
            frontmatter.update(paper.properties)
            
        import yaml
        fm_str = yaml.dump(frontmatter, allow_unicode=True).strip()
        return f"---\n{fm_str}\n---\n\n{paper.abstract or ''}"
    except Exception as e:
        return f"Error retrieving note: {str(e)}"

@mcp.resource("graph://papers/{paper_id}/abstract")
def get_paper_abstract_resource(paper_id: str) -> str:
    """Retrieve the abstract or summary text for a specific paper."""
    try:
        graph_repo = get_graph_repo()
        paper = graph_repo.get_paper(paper_id)
        if not paper:
            return f"Error: Paper with ID '{paper_id}' not found."
        return f"# Abstract: {paper.title}\n\n{paper.abstract or 'No abstract available.'}"
    except Exception as e:
        return f"Error retrieving paper abstract: {str(e)}"


# ── MCP Prompts ──

@mcp.prompt()
def summarize_paper(paper_id: str) -> str:
    """Generate a prompt to summarize a specific paper or note."""
    return f"Please summarize the paper or research note with ID '{paper_id}'. Provide key findings, methodology, and list the core concepts mentioned."

@mcp.prompt()
def compare_papers(paper_id_1: str, paper_id_2: str) -> str:
    """Generate a prompt to compare two research documents."""
    return f"Compare the paper/note with ID '{paper_id_1}' and the paper/note with ID '{paper_id_2}'. Find common themes, contrasting viewpoints, and analyze how their concepts link together."

@mcp.prompt()
def analyze_concept(concept_id: str) -> str:
    """Generate a prompt to analyze a specific concept across the graph."""
    return f"Analyze the concept '{concept_id}'. What documents mention it? How is it defined across the graph, and what other concepts or tags is it connected to?"


# ── Extra MCP Editing & Graph Tools ──

@mcp.tool
def manage_graph(
    action: str,
    node_id: Optional[str] = None,
    source_id: Optional[str] = None,
    target_id: Optional[str] = None,
    relationship_type: Optional[str] = None,
    properties: Optional[dict] = None,
    paper_id: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> dict:
    """Unified graph management tool.
    
    Supported actions and their parameters:
    - action="delete_node" -> node_id (str)
    - action="create_edge" -> source_id (str), target_id (str), relationship_type (str), properties (Optional[dict])
    - action="delete_edge" -> source_id (str), target_id (str), relationship_type (str)
    - action="add_tags" -> paper_id (str), tags (List[str])
    """
    try:
        graph_repo = get_graph_repo()
        action_clean = action.strip()
        
        if action_clean == "delete_node":
            if not node_id:
                return {"status": "error", "message": "node_id is required for delete_node action."}
            node = graph_repo.get_node_by_id(node_id)
            if not node:
                return {"status": "error", "message": f"Node with ID '{node_id}' not found."}
                
            graph_repo.delete_node(node_id)
            return {"status": "success", "message": f"Node '{node_id}' deleted successfully."}
            
        elif action_clean == "create_edge":
            if not source_id or not target_id or not relationship_type:
                return {"status": "error", "message": "source_id, target_id, and relationship_type are required for create_edge action."}
            src = graph_repo.get_node_by_id(source_id)
            tgt = graph_repo.get_node_by_id(target_id)
            if not src:
                return {"status": "error", "message": f"Source node '{source_id}' not found."}
            if not tgt:
                return {"status": "error", "message": f"Target node '{target_id}' not found."}
                
            graph_repo.add_edge(source_id, target_id, relationship_type, properties or {})
            return {"status": "success", "message": f"Relationship '{relationship_type}' created from '{source_id}' to '{target_id}'."}
            
        elif action_clean == "delete_edge":
            if not source_id or not target_id or not relationship_type:
                return {"status": "error", "message": "source_id, target_id, and relationship_type are required for delete_edge action."}
            with graph_repo._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM edges WHERE source_id = ? AND target_id = ? AND type = ?",
                    (source_id, target_id, relationship_type)
                )
                conn.commit()
                if cursor.rowcount == 0:
                    return {"status": "error", "message": "No matching relationship found to delete."}
                    
            return {"status": "success", "message": f"Relationship '{relationship_type}' from '{source_id}' to '{target_id}' deleted."}
            
        elif action_clean == "add_tags":
            if not paper_id or tags is None:
                return {"status": "error", "message": "paper_id and tags are required for add_tags action."}
            paper = graph_repo.get_node_by_id(paper_id)
            if not paper:
                return {"status": "error", "message": f"Paper/note with ID '{paper_id}' not found."}
                
            from src.models import slugify, Concept
            
            added_tags = []
            for tag in tags:
                tag_clean = tag.strip()
                if not tag_clean:
                    continue
                tag_id = slugify(tag_clean)
                
                existing_concept = graph_repo.get_node_by_id(tag_id)
                if not existing_concept:
                    concept = Concept(
                        id=tag_id,
                        name=tag_clean,
                        properties={"is_tag": True}
                    )
                    graph_repo.save_concept(concept)
                    
                graph_repo.add_edge(paper_id, tag_id, "HAS_TAG")
                added_tags.append(tag_clean)
                
            return {"status": "success", "message": f"Tags {added_tags} added to paper '{paper_id}'."}
            
        else:
            return {"status": "error", "message": f"Unknown action: '{action}'. Allowed actions: 'delete_node', 'create_edge', 'delete_edge', 'add_tags'."}
            
    except Exception as e:
        return {"status": "error", "message": f"Failed to execute graph management action: {str(e)}"}

@mcp.tool
def update_note(
    note_id: str,
    title: Optional[str] = None,
    content: Optional[str] = None,
    authors: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    use_cloud: bool = False
) -> dict:
    """Update an existing research note. Modifies the markdown file on disk and re-indexes the document."""
    try:
        note_service = get_note_service(use_cloud=use_cloud)
        res = note_service.update_note(
            note_id=note_id,
            title=title,
            content=content,
            authors=authors,
            tags=tags
        )
        return res
    except Exception as e:
        return {"status": "error", "message": f"Failed to update note: {str(e)}"}

@mcp.tool
def search_graph(query: str, limit: int = 10) -> list:
    """Search the graph database for papers, authors, and concepts matching the query."""
    try:
        graph_repo = get_graph_repo()
        q_like = f"%{query}%"
        results = []
        with graph_repo._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, label, properties, title
                FROM nodes
                WHERE (label IN ('Paper', 'UserNote') AND title LIKE ?)
                   OR (label IN ('Author', 'Concept') AND json_extract(properties, '$.name') LIKE ?)
                LIMIT ?
                """,
                (q_like, q_like, limit)
            ).fetchall()
            
            for r in rows:
                node_id = r["id"]
                label = r["label"]
                props = json.loads(r["properties"] or "{}")
                
                if label in ('Paper', 'UserNote'):
                    title = r["title"] or props.get("title", node_id)
                    results.append({
                        "type": "paper",
                        "id": node_id,
                        "title": title
                    })
                elif label == 'Author':
                    name = props.get("name", node_id)
                    results.append({
                        "type": "author",
                        "id": node_id,
                        "name": name
                    })
                elif label == 'Concept':
                    name = props.get("name", node_id)
                    is_tag = props.get("is_tag", False)
                    results.append({
                        "type": "tag" if is_tag else "concept",
                        "id": node_id,
                        "name": name
                    })
        return results
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


if __name__ == "__main__":
    mcp.run()
