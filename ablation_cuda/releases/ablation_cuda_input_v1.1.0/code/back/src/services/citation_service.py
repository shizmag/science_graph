import asyncio
import copy
import logging
import re
from typing import Any, TypedDict

from src.services.extraction_service import ExtractionService

logger = logging.getLogger(__name__)


class CitationInput(TypedDict, total=False):
    """Type definition for citation dictionaries passed to the service."""

    source_id: str
    target_id: str
    title: str
    author: str
    year: int
    properties: dict[str, Any] | None


# Compiled regex constant to split sentences avoiding common academic abbreviations,
# decimals, initials, and et al. without fragmentation.
SENTENCE_SPLIT_PAT = re.compile(
    r"(?<!\b[A-Z]\.)(?<!\w\.\w\.)(?<!\b[Ee]t\s[Aa]l\.)"
    r"(?<!\b[Ff]ig\.)(?<!\b[Rr]ef\.)(?<!\b[Ee]q\.)(?<!\b[Vv]s\.)"
    r"(?<!\b[Ss]ec\.)(?<!\b[Cc]f\.)(?<!\b[Pp]p\.)(?<!\b[Vv]ol\.)"
    r"(?<!\b[Cc]h\.)(?<!\b[Nn]o\.)(?<!\b[Ee]tc\.)(?<!\b[A-Z][a-z]\.)"
    r"(?<=\.|\?|!)\s+"
)


class CitationService:
    """Service to handle citation context extraction and classification."""

    def __init__(self, extractor: ExtractionService) -> None:
        """Initialize the CitationService with an ExtractionService instance.

        Args:
            extractor: The extraction service used to classify citation intents.
        """
        self.extractor = extractor

    def _extract_primary_author(self, author: str | None) -> str | None:
        """Extracts the primary surname from an author string.

        Handles formats like 'Goodfellow, I.', 'Goodfellow, Ian',
        'I. Goodfellow', 'Ian Goodfellow', 'Vaswani et al.',
        'Vaswani and Bengio', 'Vaswani, A. and Bengio, Y.',
        and comma-separated lists like 'A. Vaswani, Y. Bengio'.

        Args:
            author: The raw author name string.

        Returns:
            The extracted surname/primary name, or None if input is invalid.
        """
        if not isinstance(author, str) or not author.strip():
            return None

        # Clean common et al. suffixes
        author_clean = re.sub(
            r"\b(et\s+al\b\.?|and\s+others\b\.?)", "", author, flags=re.IGNORECASE
        ).strip()
        if not author_clean:
            return None

        # Split multiple authors using 'and' or '&' to isolate the first author
        first_author = re.split(
            r"\b(?:and)\b|\s*&\s*", author_clean, flags=re.IGNORECASE
        )[0].strip()

        # Split comma-separated list to handle Surname, Initials vs Multiple Authors
        parts = [p.strip() for p in first_author.split(",")]
        if len(parts) > 1:
            if " " in parts[0]:
                first_author = parts[0]
            else:
                return parts[0]
        else:
            first_author = parts[0]

        # Split by whitespace
        words = first_author.split()
        if not words:
            return None

        # Remove common initial patterns like 'I.', 'I.J.', etc.
        cleaned_words = []
        for w in words:
            clean_w = w.strip(".")
            if len(clean_w) > 1:
                cleaned_words.append(w)

        if cleaned_words:
            # Surnames are often the last word if initials/first names are first
            # e.g., 'Ian Goodfellow' -> 'Goodfellow'
            return cleaned_words[-1].strip(".,")

        return words[-1].strip(".,")

    def get_citation_context(
        self,
        full_text: str,
        ref_title: str | None = None,
        ref_author: str | None = None,
        ref_year: int | str | None = None,
        sentences: list[str] | None = None,
    ) -> str:
        """Extracts a surrounding sentence window context for a reference match.

        Args:
            full_text: The full text of the document.
            ref_title: The title of the referenced work (optional).
            ref_author: The author of the referenced work (optional).
            ref_year: The publication year of the referenced work (optional).
            sentences: Optional pre-split list of sentences to optimize performance.

        Returns:
            A string containing the matched sentence and its surrounding context,
            or an empty string if no match is found or arguments are invalid.
        """
        if ref_title is not None and not isinstance(ref_title, str):
            return ""

        if sentences is None:
            if not isinstance(full_text, str) or not full_text:
                return ""
            try:
                sentences = SENTENCE_SPLIT_PAT.split(full_text)
            except Exception:
                # Handle potential regex split errors on malformed strings
                return ""
        else:
            if not isinstance(sentences, list):
                return ""

        if not sentences:
            return ""

        patterns = []
        primary_author = self._extract_primary_author(ref_author)

        def safe_bound(s: str) -> str:
            escaped = re.escape(s)
            if re.match(r"^\w", s):
                escaped = r"\b" + escaped
            if re.search(r"\w$", s):
                escaped = escaped + r"\b"
            return escaped

        # 1. Author + Year (highest specificity)
        if primary_author and ref_year is not None:
            author_pat = safe_bound(primary_author)
            year_pat = safe_bound(str(ref_year))
            patterns.append(
                re.compile(
                    rf"(?=.*{author_pat})(?=.*{year_pat})",
                    re.IGNORECASE,
                )
            )

        # 2. Title fallback
        if ref_title and len(ref_title) >= 3:
            words = ref_title.split()[:4]
            if words:
                escaped_words = [re.escape(w) for w in words]
                title_pat = r"\s+".join(escaped_words)
                if re.match(r"^\w", words[0]):
                    title_pat = r"\b" + title_pat
                if re.search(r"\w$", words[-1]):
                    title_pat = title_pat + r"\b"
                patterns.append(re.compile(title_pat, re.IGNORECASE))

        # 3. Author only (lowest specificity)
        if primary_author:
            patterns.append(
                re.compile(safe_bound(primary_author), re.IGNORECASE)
            )

        for pat in patterns:
            for idx, sent in enumerate(sentences):
                try:
                    if pat.search(sent):
                        start = max(0, idx - 1)
                        end = min(len(sentences), idx + 2)
                        return " ".join(sentences[start:end]).strip()
                except Exception:
                    continue
        return ""

    async def classify_cites_edges_async(
        self, cites_list: list[CitationInput], full_text: str
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Takes a list of citation dicts and classifies their citation intents.

        Args:
            cites_list: A list of dictionaries representing citations, each
                conforming to the CitationInput TypedDict structure.
            full_text: The full text of the document containing citations.

        Returns:
            A list of tuples of the form (source_id, target_id, "CITES", properties)
            where properties has context and intent updated.
        """
        if not isinstance(cites_list, list):
            return []
        if not isinstance(full_text, str):
            full_text = ""

        # Pre-split sentences once to avoid quadratic overhead on large papers
        try:
            sentences = (
                SENTENCE_SPLIT_PAT.split(full_text)
                if full_text
                else []
            )
        except Exception:
            sentences = []

        tasks = []
        metadata = []

        for cit in cites_list:
            if not isinstance(cit, dict):
                continue

            ref_title = cit.get("title")
            ref_author = cit.get("author")
            ref_year = cit.get("year")

            # Avoid potential KeyError if source_id or target_id are missing
            source_id = cit.get("source_id", "")
            target_id = cit.get("target_id", "")

            # Pass pre-split sentences list to optimize performance
            context = self.get_citation_context(
                full_text, ref_title, ref_author, ref_year, sentences=sentences
            )

            # Prevent TypeError crash if "properties" is None or invalid type
            cit_props = cit.get("properties")
            props = {}
            if isinstance(cit_props, dict):
                props = copy.deepcopy(cit_props)
            props["context"] = context

            if context:
                tasks.append(
                    self.extractor.classify_citation_intent_async(
                        context, ref_title or ""
                    )
                )
                metadata.append((source_id, target_id, props, True))
            else:
                props["intent"] = "BACKGROUND"
                metadata.append((source_id, target_id, props, False))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []

        edges = []
        task_idx = 0
        for src, tgt, props, is_active in metadata:
            if is_active:
                intent = results[task_idx]
                task_idx += 1
                if isinstance(intent, Exception):
                    logger.warning(f"Citation classification failed: {intent}")
                    intent = "BACKGROUND"
            else:
                intent = "BACKGROUND"
            props["intent"] = intent or "BACKGROUND"
            edges.append((src, tgt, "CITES", props))
        return edges
