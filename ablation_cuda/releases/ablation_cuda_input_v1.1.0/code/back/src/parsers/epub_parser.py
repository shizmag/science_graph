"""
EPUB parser using ebooklib.
Extracts chapters as pages, metadata (title, author, language),
and returns a Paper-compatible object with source_type="book".
"""

import re
from pathlib import Path
from typing import List, Tuple

from src.models import Paper, slugify
from src.parsers.base import BaseParser


def _clean_html(html: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class EPUBParser(BaseParser):
    def parse(self, source: str) -> Tuple[Paper, List[str], str]:
        """
        Parses an EPUB file.

        Returns:
            paper       – Paper object (source_type="book" in properties)
            links       – empty list (EPUBs don't have wiki-links)
            full_text   – concatenated chapter texts
        """
        import ebooklib
        from ebooklib import epub

        path = Path(source)
        book = epub.read_epub(str(path), options={"ignore_ncx": True})

        # ── Metadata ──────────────────────────────────────────────────────────────
        def _meta(key: str) -> str:
            values = book.get_metadata("DC", key)
            return values[0][0].strip() if values else ""

        title = _meta("title") or path.stem
        author_raw = _meta("creator")
        language = _meta("language") or "en"
        identifier = _meta("identifier")

        authors: List[str] = []
        if author_raw:
            authors = [a.strip() for a in re.split(r',|;| and ', author_raw) if a.strip()]

        # ── Chapter extraction ────────────────────────────────────────────────────
        chapters: List[str] = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            raw_html = item.get_content().decode("utf-8", errors="replace")
            text = _clean_html(raw_html)
            if len(text) > 50:
                chapters.append(text)

        full_text = "\n\n".join(chapters)

        # ── Paper object ──────────────────────────────────────────────────────────
        paper_id = slugify(title)

        # Abstract: first ~600 chars of first chapter
        abstract = chapters[0][:600] if chapters else ""

        paper = Paper(
            id=paper_id,
            title=title,
            authors=authors,
            year=None,
            doi=None,
            abstract=abstract,
            file_path=source,
            properties={
                "source_type": "book",
                "language": language,
                "epub_identifier": identifier,
                "chapter_count": len(chapters),
                "original_path": str(path.resolve()),
            },
        )

        return paper, [], full_text

