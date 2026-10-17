import re
import datetime
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from src.repository.base import GraphRepository, VectorRepository
from src.vector_search import EmbeddingEngine
from src.llm_engine import BaseLLMEngine
from src.config import config
from src.indexer import Indexer

class NoteService:
    def __init__(
        self,
        graph_repo: GraphRepository,
        vector_repo: VectorRepository,
        embedding_engine: EmbeddingEngine,
        llm_engine: BaseLLMEngine
    ):
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.emb_engine = embedding_engine
        self.llm_engine = llm_engine

    def get_notes(self) -> List[Dict[str, Any]]:
        papers = self.graph_repo.get_notes()
        notes = []
        for p in papers:
            notes.append({
                "id": p.id,
                "title": p.title,
                "created_at": p.created_at,
                "authors": p.authors,
                "summary": p.properties.get("summary"),
                "abstract": p.abstract
            })
        return notes

    def create_note(
        self,
        title: str,
        content: str,
        authors: List[str] = None,
        tags: List[str] = None,
        comments_on: List[str] = None,
        agrees_with: List[str] = None,
        disagrees_with: List[str] = None,
        linked_to: List[str] = None
    ) -> Tuple[str, str]:
        """
        Creates a markdown note on disk with YAML frontmatter and indexes it.
        Returns a tuple of (paper_id, note_path).
        """
        notes_dir = Path(config.data_dir) / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        
        def _safe_filename(text: str) -> str:
            text = text.lower().strip()
            text = re.sub(r'[^a-z0-9\s-]', '', text)
            return re.sub(r'[\s-]+', '_', text)
            
        filename = f"{_safe_filename(title)}.md"
        note_path = notes_dir / filename
        
        yaml_front = {
            "title": title,
            "created_at": datetime.datetime.now().isoformat(),
        }
        if authors:
            yaml_front["authors"] = authors
        if tags:
            yaml_front["tags"] = tags
        if comments_on:
            yaml_front["comments_on"] = comments_on
        if agrees_with:
            yaml_front["agrees_with"] = agrees_with
        if disagrees_with:
            yaml_front["disagrees_with"] = disagrees_with
        if linked_to:
            yaml_front["linked_to"] = linked_to
            
        frontmatter_str = yaml.dump(yaml_front, allow_unicode=True).strip()
        full_md = f"---\n{frontmatter_str}\n---\n\n{content}"
        
        note_path.write_text(full_md, encoding="utf-8")
        
        indexer = Indexer(self.graph_repo, self.vector_repo, self.emb_engine, self.llm_engine)
        paper_id = indexer.index_markdown(str(note_path))
        
        return paper_id, str(note_path)

    def update_note(
        self,
        note_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        authors: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Updates a markdown note on disk (YAML frontmatter and content) and re-indexes it.
        Returns a dict with status, the potentially new id, and the file path.
        """
        import os
        import frontmatter

        paper = self.graph_repo.get_paper(note_id)
        if not paper:
            raise ValueError(f"Note with ID '{note_id}' not found.")

        file_path = paper.file_path
        if not file_path or not os.path.exists(file_path):
            raise ValueError(f"File path for note '{note_id}' not found on disk.")

        post = frontmatter.load(file_path)

        old_title = post.metadata.get("title") or paper.title

        if title is not None:
            post.metadata["title"] = title
        if authors is not None:
            post.metadata["authors"] = authors
        if tags is not None:
            post.metadata["tags"] = tags
        if content is not None:
            post.content = content

        new_title = post.metadata.get("title") or old_title
        notes_dir = Path(config.data_dir) / "notes"
        
        def _safe_filename(text: str) -> str:
            text = text.lower().strip()
            text = re.sub(r'[^a-z0-9\s-]', '', text)
            return re.sub(r'[\s-]+', '_', text)

        new_filename = f"{_safe_filename(new_title)}.md"
        new_file_path = notes_dir / new_filename

        target_path = new_file_path
        if str(new_file_path) != str(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

        with open(target_path, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))

        self.graph_repo.delete_node(note_id)

        indexer = Indexer(self.graph_repo, self.vector_repo, self.emb_engine, self.llm_engine)
        new_paper_id = indexer.index_markdown(str(target_path))

        return {"status": "success", "id": new_paper_id, "file_path": str(target_path)}
