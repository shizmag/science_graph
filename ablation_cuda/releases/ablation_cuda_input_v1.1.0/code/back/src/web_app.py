"""
FastAPI Web Application for Science Graph.

Endpoints:
  GET  /              → serves web/index.html
  GET  /api/graph     → all nodes + edges for vis-network
  GET  /api/stats     → database statistics
  POST /api/query     → SSE-streamed RAG answer
  GET  /api/paper/{id}→ paper metadata + concepts/authors
  GET  /api/search    → quick title search
  POST /api/upload    → index uploaded PDF/MD/EPUB
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Optional, List, Union

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from src.config import config
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from src.vector_search import EmbeddingEngine
from src.indexer import DuplicateDocumentError
from src.llm_engine import LLMEngine, strip_thinking_tokens
from src.services.rag_service import RAGService
from src.services.note_service import NoteService
from src.schemas import (
    QueryRequest,
    NoteCreate,
    NoteResponse,
    OpenFileRequest,
    GraphResponse,
    SearchResponse,
    StatsResponse,
    PaperDetailResponse,
    AuthorDetailResponse,
    ConceptDetailResponse,
    UploadResponse,
    NoteCreateResponse,
    OpenFileResponse,
    UrlIndexRequest,
    UrlIndexResponse,
    LibraryResponse,
)


# ── Dependency Injection Providers (with caching for performance) ──

from src.services.container import container

def get_graph_repo() -> SQLiteGraphRepository:
    return container.get_graph_repo()


def get_vector_repo() -> SQLiteVectorRepository:
    return container.get_vector_repo()


def get_embedding_engine() -> EmbeddingEngine:
    return container.get_embedding_engine()


def get_llm_engine(use_cloud: bool = False) -> Optional[LLMEngine]:
    cloud = use_cloud or os.environ.get("SCIENCE_GRAPH_USE_CLOUD") == "1"
    try:
        return container.get_llm_engine(use_cloud=cloud)
    except Exception as e:
        print(f"[!] LLMEngine unavailable (cloud={cloud}): {e}")
        return None


def get_default_llm_engine() -> Optional[LLMEngine]:
    return get_llm_engine(use_cloud=False)


def get_rag_service(use_cloud: bool = False) -> Optional[RAGService]:
    cloud = use_cloud or os.environ.get("SCIENCE_GRAPH_USE_CLOUD") == "1"
    try:
        return container.get_rag_service(use_cloud=cloud)
    except Exception as e:
        print(f"[!] RAGService unavailable (cloud={cloud}): {e}")
        return None


def get_note_service(
    graph_repo: SQLiteGraphRepository = Depends(get_graph_repo),
    vector_repo: SQLiteVectorRepository = Depends(get_vector_repo),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
    llm_engine: Optional[LLMEngine] = Depends(get_default_llm_engine)
) -> NoteService:
    return container.get_note_service()


# ── App Setup ──

app = FastAPI(title="Science Graph", version="0.1.0")

# Serve static files from frontend/out in the repository root
_WEB_DIR = Path(__file__).resolve().parents[2] / "frontend" / "out"

# Mount Next.js static assets if the directory exists
if (_WEB_DIR / "_next").exists():
    app.mount("/_next", StaticFiles(directory=str(_WEB_DIR / "_next")), name="next-assets")


@app.get("/", include_in_schema=False)
async def root():
    index_file = _WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Web UI build not found. Please run 'npm run build' inside the 'frontend' directory first."
        )
    return FileResponse(str(index_file))


@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon_ico():
    fav_file = _WEB_DIR / "favicon.ico"
    if fav_file.exists():
        return FileResponse(str(fav_file))
    raise HTTPException(status_code=404, detail="favicon.ico not found.")



# ── /api/stats ──

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats(graph_repo: SQLiteGraphRepository = Depends(get_graph_repo)):
    stats = await asyncio.to_thread(graph_repo.get_stats)
    try:
        storage_stats = await asyncio.to_thread(config.get_storage_stats)
        stats["storage"] = storage_stats
    except Exception:
        pass
    return stats


# ── /api/models ──

@app.get("/api/models")
async def get_models():
    return {
        "llm_local": config.llm_local_model_path.split("/")[-1] if config.llm_local_model_path else "not set",
        "llm_cloud": config.llm_cloud_model_name,
        "llm_provider": config.llm_provider,
        "embedding": config.embedding_model_name.split("/")[-1] if config.embedding_model_name else "not set",
        "spacy": config.spacy_model_name,
        "ner": config.ner_model_name.split("/")[-1] if config.ner_model_name else "not set"
    }


# ── /api/graph ──

@app.get("/api/graph", response_model=GraphResponse)
async def get_graph(
    show_references: bool = False,
    graph_repo: SQLiteGraphRepository = Depends(get_graph_repo)
):
    """Returns all nodes and edges formatted for vis-network."""
    nodes_rows = await asyncio.to_thread(graph_repo.get_all_nodes)
    edges_rows = await asyncio.to_thread(graph_repo.get_all_edges)

    # 1. Identify indexed and mentioned (placeholder) papers
    indexed_paper_ids = set()
    placeholder_paper_ids = set()
    for node_id, label, properties_json in nodes_rows:
        if label == "Paper":
            props = json.loads(properties_json or "{}")
            is_placeholder = bool(props.get("is_placeholder") or props.get("placeholder"))
            if is_placeholder:
                placeholder_paper_ids.add(node_id)
            else:
                indexed_paper_ids.add(node_id)

    allowed_paper_ids = indexed_paper_ids
    if show_references:
        allowed_paper_ids = allowed_paper_ids.union(placeholder_paper_ids)

    # 2. Identify authors and concepts connected to the allowed papers
    connected_non_papers = set()
    for source_id, target_id, edge_type, edge_properties in edges_rows:
        if source_id in allowed_paper_ids:
            connected_non_papers.add(target_id)
        if target_id in allowed_paper_ids:
            connected_non_papers.add(source_id)

    allowed_node_ids = allowed_paper_ids.union(connected_non_papers)

    # 3. Format allowed nodes
    vis_nodes = []
    for node_id, label, properties_json in nodes_rows:
        if node_id not in allowed_node_ids:
            continue

        props = json.loads(properties_json or "{}")
        source_type = props.get("source_type", "paper")
        is_placeholder = bool(props.get("is_placeholder") or props.get("placeholder"))

        if label == "Paper":
            title = props.get("title", node_id)
            display = title if len(title) < 28 else title[:25] + "…"
            
            if is_placeholder:
                color = "#64748b" # slate/grey color for placeholder references
                size = 14
                group = "reference"
                source_type = "reference"
            else:
                color_map = {
                    "note": "#a5b4fc",
                    "book": "#818cf8",
                    "paper": "#6366f1",
                    "video": "#f43f5e",
                    "webpage": "#06b6d4"
                }
                color = color_map.get(source_type, "#6366f1")
                size = 25
                group = source_type
        elif label == "Author":
            display = props.get("name", node_id)
            color = "#cbd5e1"
            size = 18
            group = "author"
        elif label == "Concept":
            display = props.get("name", node_id)
            if props.get("is_tag"):
                color = "#ec4899"
                size = 15
                group = "tag"
            else:
                color = "#10b981"
                size = 16
                group = "concept"
        else:
            display = node_id
            color = "#475569"
            size = 14
            group = "other"

        tooltip = f"<b>{label} (Упомянутая работа)</b>: {props.get('title', props.get('name', node_id))}" if is_placeholder else f"<b>{label}</b>: {props.get('title', props.get('name', node_id))}"
        if props.get("year"):
            tooltip += f"<br>Year: {props['year']}"
        if props.get("authors"):
            tooltip += f"<br>Authors: {', '.join(props['authors'][:3])}"

        vis_nodes.append({
            "id": node_id,
            "label": display,
            "title": tooltip,
            "color": color,
            "size": size,
            "group": group,
            "shape": "dot",
            "created_at": props.get("created_at"),
            "source_type": source_type,
            "full_title": props.get("title", props.get("name", node_id)),
        })

    # 4. Format allowed edges
    vis_edges = []
    for source_id, target_id, edge_type, edge_properties in edges_rows:
        if source_id not in allowed_node_ids or target_id not in allowed_node_ids:
            continue
        vis_edges.append({
            "from": source_id,
            "to": target_id,
            "label": edge_type,
            "arrows": "to",
            "font": {"size": 8, "align": "top", "color": "#94a3b8"},
            "color": {"color": "rgba(255, 255, 255, 0.15)", "highlight": "#6366f1"},
        })

    return {"nodes": vis_nodes, "edges": vis_edges}



# ── /api/paper/{id} ──

@app.get("/api/paper/{paper_id:path}", response_model=Union[PaperDetailResponse, AuthorDetailResponse, ConceptDetailResponse])
async def get_paper(
    paper_id: str,
    graph_repo: SQLiteGraphRepository = Depends(get_graph_repo)
):
    node = await asyncio.to_thread(graph_repo.get_node_by_id, paper_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    label, properties_json = node
    props = json.loads(properties_json or "{}")

    if label == "Paper":
        paper = await asyncio.to_thread(graph_repo.get_paper, paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")

        neighbors = await asyncio.to_thread(graph_repo.get_neighbors, paper_id, max_depth=1)

        # ── Collect IDs for batch fetching ──
        concept_ids: list[str] = []
        tag_ids: list[str] = []
        author_ids: list[str] = []
        citation_ids: list[str] = []
        cited_by_ids: list[str] = []

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

        # ── Batch fetch citations and cited_by papers ──
        citations_map = await asyncio.to_thread(
            graph_repo.get_papers_batch, citation_ids
        ) if citation_ids else {}
        cited_by_map = await asyncio.to_thread(
            graph_repo.get_papers_batch, cited_by_ids
        ) if cited_by_ids else {}

        # ── Batch fetch concept and tag node info ──
        async def _get_node(nid: str):
            return await asyncio.to_thread(graph_repo.get_node_by_id, nid)

        import asyncio as _aio
        concept_nodes = await _aio.gather(*[_get_node(cid) for cid in concept_ids])
        tag_nodes = await _aio.gather(*[_get_node(tid) for tid in tag_ids])

        concepts = []
        for cid, node in zip(concept_ids, concept_nodes):
            if node:
                c_props = json.loads(node[1] or "{}")
                concepts.append({"id": cid, "name": c_props.get("name", cid)})
            else:
                concepts.append({"id": cid, "name": cid})

        tags = []
        for tid, node in zip(tag_ids, tag_nodes):
            if node:
                t_props = json.loads(node[1] or "{}")
                tags.append({"id": tid, "name": t_props.get("name", tid)})
            else:
                tags.append({"id": tid, "name": tid})

        # Resolve author names (typically a small set, so individual calls are acceptable)
        authors = paper.authors or []
        if not authors and author_ids:
            author_objs = await _aio.gather(
                *[asyncio.to_thread(graph_repo.get_author, aid) for aid in author_ids]
            )
            authors = [a.name for a in author_objs if a]

        citations = [
            {"id": pid, "title": p.title}
            for pid in citation_ids
            if (p := citations_map.get(pid)) and p.title
        ][:100]

        cited_by = [
            {"id": pid, "title": p.title}
            for pid in cited_by_ids
            if (p := cited_by_map.get(pid)) and p.title
        ][:100]

        abstract = paper.abstract
        if abstract:
            abstract = strip_thinking_tokens(abstract)
        summary = paper.properties.get("summary")
        if summary:
            summary = strip_thinking_tokens(summary)

        clean_props = {**paper.properties}
        if "summary" in clean_props and clean_props["summary"]:
            clean_props["summary"] = strip_thinking_tokens(clean_props["summary"])
        if "video_overview" in clean_props and clean_props["video_overview"]:
            clean_props["video_overview"] = strip_thinking_tokens(clean_props["video_overview"])
        if "video_themes" in clean_props and isinstance(clean_props["video_themes"], list):
            clean_props["video_themes"] = [strip_thinking_tokens(t) if isinstance(t, str) else t for t in clean_props["video_themes"]]
        if "video_outline" in clean_props and isinstance(clean_props["video_outline"], list):
            clean_props["video_outline"] = [strip_thinking_tokens(o) if isinstance(o, str) else o for o in clean_props["video_outline"]]

        return {
            "type": "paper",
            "id": paper.id,
            "title": paper.title,
            "authors": authors,
            "year": paper.year,
            "doi": paper.doi,
            "abstract": abstract,
            "source_type": paper.properties.get("source_type", "paper"),
            "concepts": concepts,
            "tags": tags,
            "citations": citations,
            "cited_by": cited_by,
            "file_path": paper.file_path,
            "summary": summary,
            "created_at": paper.created_at,
            "properties": clean_props,
        }

    elif label == "Author":
        papers_list = await asyncio.to_thread(graph_repo.get_papers_by_author, paper_id)
        papers = []
        for p in papers_list:
            papers.append({
                "id": p.id,
                "title": p.title,
                "source_type": p.properties.get("source_type", "paper")
            })

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
        papers_list = await asyncio.to_thread(graph_repo.get_papers_by_entity, paper_id, edge_type)

        papers = []
        for p in papers_list:
            papers.append({
                "id": p.id,
                "title": p.title,
                "source_type": p.properties.get("source_type", "paper")
            })

        related_entities = []
        if papers:
            paper_ids = [p["id"] for p in papers]
            related_edge_type = "HAS_TAG" if not is_tag else "MENTIONS_CONCEPT"
            targets = await asyncio.to_thread(graph_repo.get_distinct_targets, paper_ids, related_edge_type)
            for t_id, t_props_json in targets:
                t_props = json.loads(t_props_json or "{}")
                related_entities.append({
                    "id": t_id,
                    "name": t_props.get("name", t_id)
                })

        return {
            "type": "tag" if is_tag else "concept",
            "id": paper_id,
            "name": props.get("name", paper_id),
            "description": props.get("description", f"No description available for '{props.get('name', paper_id)}'."),
            "papers": papers,
            "related": related_entities
        }


@app.get("/api/paper-text")
async def get_paper_text(
    paper_id: str = Query(...),
    vector_repo: SQLiteVectorRepository = Depends(get_vector_repo)
):
    """Fetch text chunks of a paper sorted by page number."""
    chunks = await asyncio.to_thread(vector_repo.get_chunks_for_paper, paper_id)
    chunks.sort(key=lambda x: (x.page_number or 0, x.id))
    return {
        "paper_id": paper_id,
        "chunks": [
            {
                "id": c.id,
                "text_content": c.text_content,
                "page_number": c.page_number
            }
            for c in chunks
        ]
    }


@app.get("/api/paper-pdf/{paper_id:path}")
async def get_paper_pdf(
    paper_id: str,
    graph_repo: SQLiteGraphRepository = Depends(get_graph_repo)
):
    """Serve the original PDF file of a paper if available."""
    paper = await asyncio.to_thread(graph_repo.get_paper, paper_id)
    if not paper or not paper.file_path:
        raise HTTPException(status_code=404, detail="Paper or file path not found")

    expanded = os.path.expanduser(paper.file_path)
    if not os.path.exists(expanded):
        expanded = str(Path(paper.file_path).resolve())
        if not os.path.exists(expanded):
            raise HTTPException(status_code=404, detail=f"File not found on host: {paper.file_path}")

    suffix = Path(expanded).suffix.lower()
    if suffix == ".pdf":
        media_type = "application/pdf"
    elif suffix == ".md":
        media_type = "text/markdown"
    elif suffix == ".epub":
        media_type = "application/epub+zip"
    else:
        media_type = "application/octet-stream"

    headers = {
        "Content-Disposition": f"inline; filename=\"{os.path.basename(expanded)}\""
    }
    return FileResponse(expanded, media_type=media_type, headers=headers)



# ── /api/search ──

@app.get("/api/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1),
    graph_repo: SQLiteGraphRepository = Depends(get_graph_repo)
):
    """Quick keyword search over paper/note/book titles."""
    papers = await asyncio.to_thread(graph_repo.search_papers_by_title, q, 20)

    results = []
    for paper in papers:
        results.append({
            "id": paper.id,
            "title": paper.title,
            "year": paper.year,
            "source_type": paper.properties.get("source_type", "paper")
        })

    return {"results": results}


# ── /api/documents ──

@app.get("/api/documents", response_model=LibraryResponse)
async def get_documents(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    q: Optional[str] = Query(None),
    source_type: Optional[List[str]] = Query(None),
    author: Optional[List[str]] = Query(None),
    concept: Optional[List[str]] = Query(None),
    tag: Optional[List[str]] = Query(None),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    only_indexed: bool = Query(False),
    graph_repo: SQLiteGraphRepository = Depends(get_graph_repo)
):
    """Paginated list of papers/notes/books with search and concepts."""
    conditions = ["label IN ('Paper', 'UserNote')"]
    if only_indexed:
        conditions.append("is_placeholder = 0")
    params = []

    if q:
        like_pat = f"%{q}%"
        conditions.append(
            "(id IN (SELECT DISTINCT paper_id FROM chunks WHERE text_content LIKE ?) OR properties LIKE ?)"
        )
        params.extend([like_pat, like_pat])

    if source_type:
        type_conds = []
        placeholders = ", ".join("?" for _ in source_type)
        type_conds.append(f"json_extract(properties, '$.source_type') IN ({placeholders})")
        if "paper" in source_type:
            type_conds.append("(label = 'Paper' AND json_extract(properties, '$.source_type') IS NULL)")
        if "note" in source_type:
            type_conds.append("(label = 'UserNote' AND json_extract(properties, '$.source_type') IS NULL)")
        conditions.append("(" + " OR ".join(type_conds) + ")")
        params.extend(source_type)

    if from_date:
        conditions.append("substr(json_extract(properties, '$.created_at'), 1, 10) >= ?")
        params.append(from_date)

    if to_date:
        conditions.append("substr(json_extract(properties, '$.created_at'), 1, 10) <= ?")
        params.append(to_date)

    if author:
        author_placeholders = ", ".join("?" for _ in author)
        conditions.append(
            f"""
            id IN (
                SELECT target_id FROM edges 
                WHERE type = 'AUTHORED' 
                AND source_id IN (
                    SELECT id FROM nodes 
                    WHERE label = 'Author' 
                    AND (id IN ({author_placeholders}) OR json_extract(properties, '$.name') IN ({author_placeholders}))
                )
            )
            """
        )
        params.extend(author * 2)

    if concept:
        concept_placeholders = ", ".join("?" for _ in concept)
        conditions.append(
            f"""
            id IN (
                SELECT source_id FROM edges 
                WHERE type = 'MENTIONS_CONCEPT' 
                AND target_id IN (
                    SELECT id FROM nodes 
                    WHERE label = 'Concept' 
                    AND (id IN ({concept_placeholders}) OR json_extract(properties, '$.name') IN ({concept_placeholders}))
                )
            )
            """
        )
        params.extend(concept * 2)

    if tag:
        tag_placeholders = ", ".join("?" for _ in tag)
        conditions.append(
            f"""
            id IN (
                SELECT source_id FROM edges 
                WHERE type = 'MENTIONS_CONCEPT' 
                AND target_id IN (
                    SELECT id FROM nodes 
                    WHERE label = 'Concept' 
                    AND json_extract(properties, '$.is_tag') = 1
                    AND (id IN ({tag_placeholders}) OR json_extract(properties, '$.name') IN ({tag_placeholders}))
                )
            )
            """
        )
        params.extend(tag * 2)

    where_clause = " AND ".join(conditions)

    def _fetch_data():
        with graph_repo._get_connection() as conn:
            # Get total count
            count_sql = f"SELECT count(*) FROM nodes WHERE {where_clause}"
            total_count = conn.execute(count_sql, params).fetchone()[0]

            # Get paginated rows
            off = (page - 1) * limit
            rows_sql = f"SELECT id, properties FROM nodes WHERE {where_clause} LIMIT ? OFFSET ?"
            rows_params = params + [limit, off]
            rows_data = conn.execute(rows_sql, rows_params).fetchall()
            return total_count, [dict(r) for r in rows_data]

    total, rows = await asyncio.to_thread(_fetch_data)

    results = []
    for r in rows:
        paper_id = r["id"]
        props = json.loads(r["properties"] or "{}")

        def _fetch_entities():
            with graph_repo._get_connection() as conn:
                nodes = conn.execute(
                    """
                    SELECT n.id, n.properties FROM nodes n
                    JOIN edges e ON n.id = e.target_id
                    WHERE e.source_id = ? AND e.type = 'MENTIONS_CONCEPT'
                    """,
                    (paper_id,)
                ).fetchall()
                return [(n[0], n[1]) for n in nodes]

        entity_nodes = await asyncio.to_thread(_fetch_entities)

        concepts = []
        tags = []
        for nid, n_props_json in entity_nodes:
            n_props = json.loads(n_props_json or "{}")
            name = n_props.get("name", nid)
            is_tag = n_props.get("is_tag", False)
            if is_tag:
                tags.append(name)
            else:
                concepts.append(name)

        abstract = props.get("abstract")
        if abstract:
            abstract = strip_thinking_tokens(abstract)
        summary = props.get("summary")
        if summary:
            summary = strip_thinking_tokens(summary)

        results.append({
            "id": paper_id,
            "title": props.get("title", paper_id),
            "authors": props.get("authors", []),
            "year": props.get("year"),
            "doi": props.get("doi"),
            "abstract": abstract,
            "source_type": props.get("source_type", "paper"),
            "summary": summary,
            "concepts": concepts[:5],
            "tags": tags,
            "file_path": props.get("file_path"),
            "created_at": props.get("created_at")
        })

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "results": results
    }



# ── /api/query (SSE streaming) ──

@app.post("/api/query")
async def query_rag(
    body: QueryRequest
):
    """
    SSE-streamed RAG answer.
    """
    rag_service = get_rag_service(use_cloud=bool(body.cloud))
    if rag_service is None:
        raise HTTPException(status_code=503, detail="LLM engine is not available.")

    async def event_stream() -> AsyncGenerator[dict, None]:
        async for event in rag_service.generate_stream(body.question, body.limit, paper_id=body.paper_id):
            yield {"data": json.dumps(event)}

    return EventSourceResponse(event_stream())


# ── /api/open-file ──

@app.post("/api/open-file", response_model=OpenFileResponse)
async def open_file(body: OpenFileRequest):
    file_path = body.file_path
    expanded = os.path.expanduser(file_path)
    if not os.path.exists(expanded):
        expanded = str(Path(file_path).resolve())
        if not os.path.exists(expanded):
            raise HTTPException(status_code=404, detail=f"File not found on host: {file_path}")

    import subprocess
    import sys

    def _open():
        if sys.platform == "win32":
            os.startfile(expanded)
        elif sys.platform == "darwin":
            subprocess.run(["open", expanded], check=True)
        else:
            subprocess.run(["xdg-open", expanded], check=True)

    try:
        await asyncio.to_thread(_open)
        return {"status": "ok", "message": f"Opened {file_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /api/notes ──

@app.get("/api/notes", response_model=List[NoteResponse])
async def get_notes(note_service: NoteService = Depends(get_note_service)):
    notes = await asyncio.to_thread(note_service.get_notes)
    return notes


@app.post("/api/notes", response_model=NoteCreateResponse)
async def create_note(
    body: NoteCreate,
    note_service: NoteService = Depends(get_note_service)
):
    try:
        paper_id, note_path = await asyncio.to_thread(
            note_service.create_note,
            body.title,
            body.content,
            body.authors,
            body.tags
        )
        return {"status": "ok", "id": paper_id, "file_path": note_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /api/upload ──

@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    graph_repo: SQLiteGraphRepository = Depends(get_graph_repo),
    vector_repo: SQLiteVectorRepository = Depends(get_vector_repo),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
    llm_engine: Optional[LLMEngine] = Depends(get_llm_engine)
):
    """Accepts a PDF, .md or .epub file and indexes it."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".md", ".epub"}:
        raise HTTPException(status_code=400, detail="Only PDF, Markdown (.md), and EPUB files are supported.")
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM engine is not available for indexing.")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from src.indexer import Indexer
        indexer = Indexer(graph_repo, vector_repo, embedding_engine, llm_engine)

        def _index():
            if suffix == ".pdf":
                return indexer.index_pdf(tmp_path)
            elif suffix == ".md":
                return indexer.index_markdown(tmp_path)
            else:
                return indexer.index_epub(tmp_path)

        paper_id = await asyncio.to_thread(_index)
        return {"status": "ok", "id": paper_id, "filename": file.filename}

    except DuplicateDocumentError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── /api/index-url ──

@app.post("/api/index-url", response_model=UrlIndexResponse)
async def index_url_route(
    body: UrlIndexRequest,
    graph_repo: SQLiteGraphRepository = Depends(get_graph_repo),
    vector_repo: SQLiteVectorRepository = Depends(get_vector_repo),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
    llm_engine: Optional[LLMEngine] = Depends(get_llm_engine)
):
    """Indexes a webpage or YouTube video URL into the knowledge graph."""
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM engine is not available for indexing.")
    try:
        from src.indexer import Indexer
        indexer = Indexer(graph_repo, vector_repo, embedding_engine, llm_engine)
        
        import re
        urls = [u.strip() for u in re.split(r'[,;]', body.url) if u.strip()]
        if not urls:
            raise HTTPException(status_code=400, detail="No URLs provided.")
            
        paper_ids = []
        titles = []
        
        for url in urls:
            # Use asyncio.to_thread since indexing can block on network/model execution
            paper_id = await asyncio.to_thread(indexer.index_url, url)
            
            # Fetch the paper details to return the title
            paper = await asyncio.to_thread(graph_repo.get_paper, paper_id)
            title = paper.title if paper else url
            paper_ids.append(paper_id)
            titles.append(title)
            
        return {"status": "ok", "id": ", ".join(paper_ids), "title": ", ".join(titles)}
    except DuplicateDocumentError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Catch-all to serve other exported files (e.g. SVG assets, subpages) or fallback to index.html for SPA routing
@app.get("/{path_name:path}", include_in_schema=False)
async def catch_all(path_name: str):
    if not path_name:
        return await root()
        
    file_path = _WEB_DIR / path_name
    if file_path.is_file():
        return FileResponse(str(file_path))
        
    index_file = _WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
        
    raise HTTPException(status_code=404, detail="Not found")
