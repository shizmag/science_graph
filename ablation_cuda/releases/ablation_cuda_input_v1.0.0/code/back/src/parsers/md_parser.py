"""
Markdown parser for Obsidian-style .md notes.
Extracts front-matter metadata, wiki-links, tags, and body text.
"""

import re
import datetime
from pathlib import Path
from typing import List, Tuple

from src.models import Paper, slugify
from src.parsers.base import BaseParser

# Matches [[WikiLink]] and [[WikiLink|Alias]]
WIKILINK_RE = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')
# Matches #tag words
TAG_RE = re.compile(r'(?<!\S)#([A-Za-z0-9_-]+)')


class MarkdownParser(BaseParser):
    def parse(self, source: str) -> Tuple[Paper, List[str], str]:
        """
        Parses a Markdown file (Obsidian-compatible).

        Returns:
            paper   – Paper object (source_type="note" stored in properties)
            links   – list of wiki-link targets (used as concept edges)
            body    – full body text (front-matter stripped)
        """
        import frontmatter  # python-frontmatter

        path = Path(source)
        raw = path.read_text(encoding="utf-8")

        # Parse YAML front-matter
        try:
            post = frontmatter.loads(raw)
            meta = post.metadata
            body = post.content
        except Exception:
            meta = {}
            body = raw

        # Title: front-matter > H1 > filename
        title = meta.get("title") or meta.get("Title") or ""
        if not title:
            h1 = re.search(r'^#\s+(.+)', body, re.MULTILINE)
            title = h1.group(1).strip() if h1 else path.stem

        # Authors: front-matter "author" or "authors" field
        raw_authors = meta.get("authors") or meta.get("author") or []
        if isinstance(raw_authors, str):
            raw_authors = [a.strip() for a in re.split(r',|;| and ', raw_authors) if a.strip()]
        authors: List[str] = [str(a) for a in raw_authors]

        # Year: front-matter "date" or "year"
        year = None
        date_raw = meta.get("date") or meta.get("year") or ""
        if date_raw:
            m = re.search(r'(\d{4})', str(date_raw))
            if m:
                year = int(m.group(1))

        # Tags: from front-matter "tags" list + inline #tags in body
        fm_tags: List[str] = meta.get("tags") or []
        if isinstance(fm_tags, str):
            fm_tags = [t.strip() for t in fm_tags.split(",")]
        inline_tags = TAG_RE.findall(body)
        all_tags = list({t.lower() for t in fm_tags + inline_tags})

        # Wiki-links: [[Target]] → concept edges
        links = WIKILINK_RE.findall(body)

        # Standard markdown links: [Label](Target)
        standard_links = re.findall(r'(?<!\!)\[([^\]]+)\]\(([^)\s]*)(?:\s+["\'][^"\']*["\'])?\)', body)
        resolved_links = []
        for label, target in standard_links:
            target = target.strip()
            if not target:
                continue
            target_clean = target.split('#')[0]
            if not target_clean:
                continue
            if target_clean.startswith(("http://", "https://", "ftp://", "mailto:")):
                resolved_links.append(target_clean)
            else:
                stem = Path(target_clean).stem
                if stem:
                    resolved_links.append(stem)
        
        # Combine both
        links = list(set(links + resolved_links))

        # Stable ID: slugify of the title
        paper_id = slugify(title)

        # Abstract: first non-empty paragraph of body
        paragraphs = [p.strip() for p in re.split(r'\n\n+', body) if p.strip()]
        abstract = paragraphs[0][:800] if paragraphs else ""

        created_at = None
        for field_name in ["created_at", "created", "date"]:
            val = meta.get(field_name)
            if val:
                if isinstance(val, (datetime.date, datetime.datetime)):
                    created_at = val.isoformat()
                else:
                    created_at = str(val).strip()
                break
                
        if not created_at:
            try:
                stat = path.stat()
                t = getattr(stat, 'st_birthtime', stat.st_mtime)
                created_at = datetime.datetime.fromtimestamp(t).isoformat()
            except Exception:
                created_at = datetime.datetime.now().isoformat()

        comments_on = meta.get("comments_on") or meta.get("comments-on") or []
        if isinstance(comments_on, str):
            comments_on = [c.strip() for c in comments_on.split(",") if c.strip()]
        agrees_with = meta.get("agrees_with") or meta.get("agrees-with") or []
        if isinstance(agrees_with, str):
            agrees_with = [c.strip() for c in agrees_with.split(",") if c.strip()]
        disagrees_with = meta.get("disagrees_with") or meta.get("disagrees-with") or []
        if isinstance(disagrees_with, str):
            disagrees_with = [c.strip() for c in disagrees_with.split(",") if c.strip()]
        linked_to = meta.get("linked_to") or meta.get("linked-to") or []
        if isinstance(linked_to, str):
            linked_to = [c.strip() for c in linked_to.split(",") if c.strip()]

        paper = Paper(
            id=paper_id,
            title=title,
            authors=authors,
            year=year,
            doi=None,
            abstract=abstract,
            file_path=source,
            created_at=created_at,
            properties={
                "source_type": "note",
                "tags": all_tags,
                "original_path": str(path.resolve()),
                "comments_on": comments_on,
                "agrees_with": agrees_with,
                "disagrees_with": disagrees_with,
                "linked_to": linked_to,
            },
        )

        return paper, links, body

