import fitz
import re
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple
from src.models import Paper, slugify
from src.parsers.base import BaseParser

# Regex patterns
DOI_REGEX = re.compile(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', re.IGNORECASE)
YEAR_REGEX = re.compile(r'\b(19\d{2}|20[0-2]\d)\b')
REF_SPLIT_REGEX = re.compile(r'\[\d+\]\s+|\n(?=\d+\.\s+)|^\d+\.\s+', re.MULTILINE)

class PDFParser(BaseParser):
    def parse(self, source: str) -> Tuple[Paper, List[str], str]:
        """
        Parses a PDF file and extracts metadata, references, and full text.
        Returns:
            paper (Paper): The parsed paper domain model
            references (List[str]): List of citation strings extracted from references
            full_text (str): The entire text of the document
        """
        if not os.path.exists(source):
            raise FileNotFoundError(f"PDF file not found: {source}")

        doc = fitz.open(source)
        
        # 1. Full text extraction
        full_text_list = []
        for page in doc:
            full_text_list.append(page.get_text())
        full_text = "\n".join(full_text_list)
        
        # First page text for heuristics
        first_page_text = doc[0].get_text() if len(doc) > 0 else ""
        
        # 2. Extract Title
        title = doc.metadata.get("title", "")
        if not title or title.lower().strip() in ["untitled", "layout 1", "microsoft word", "manuscript", "pdf", ""] or ".pdf" in title.lower():
            # Fallback 1: Font size analysis on first page
            font_title = None
            if len(doc) > 0:
                try:
                    blocks = doc[0].get_text("dict")["blocks"]
                    spans = []
                    first_non_skipped_size = None
                    for block in blocks:
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    text = span["text"].strip()
                                    size = span["size"]
                                    if not text:
                                        continue
                                    
                                    text_lower = text.lower()
                                    is_skipped = any(w in text_lower for w in ["arxiv", "http", "www.", "doi:", "preprint", "proceedings", "journal", "vol.", "no.", "issn", "isbn", "copyright", "©", "all rights reserved", "permission", "grants", "reproduce", "scholarly", "journalistic", "attribution"])
                                    
                                    if is_skipped:
                                        if first_non_skipped_size is not None:
                                            # We already started collecting title, but hit a skipped span. Stop.
                                            raise StopIteration
                                        continue
                                    
                                    if first_non_skipped_size is None:
                                        first_non_skipped_size = size
                                    
                                    if size < first_non_skipped_size - 0.5:
                                        # Smaller font size. Stop.
                                        raise StopIteration
                                        
                                    if abs(size - first_non_skipped_size) < 0.5:
                                        spans.append(text)
                                    else:
                                        # Different font size. Stop.
                                        raise StopIteration
                except StopIteration:
                    pass
                except Exception:
                    pass
                
                if spans:
                    font_title = " ".join(spans).strip()

            if font_title:
                title = font_title
            else:
                # Fallback 2: Line-based heuristics using word boundaries for forbidden words
                lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
                if lines:
                    title_candidates = []
                    forbidden_exact = {"preprint", "journal", "proceedings", "permission", "google", "grants", "copyright", "license"}
                    forbidden_subs = ["arxiv", "vol.", "no.", "issn", "http", "www.", "doi:", "scholarly", "journalistic", "attribution"]
                    for line in lines[:5]:
                        line_lower = line.lower()
                        if len(line) > 15:
                            if any(sub in line_lower for sub in forbidden_subs):
                                continue
                            words = re.findall(r'\b\w+\b', line_lower)
                            if any(w in forbidden_exact for w in words):
                                continue
                            title_candidates.append(line)
                    title = " ".join(title_candidates[:2]) if title_candidates else lines[0]
                else:
                    title = os.path.basename(source).replace(".pdf", "")
        
        # 3. Extract Authors
        author_meta = doc.metadata.get("author", "")
        authors = []
        
        # PDF metadata author field — split by comma/semicolon/and
        if author_meta:
            raw = [a.strip() for a in re.split(r',|;|\band\b', author_meta, flags=re.IGNORECASE) if a.strip()]
            # Accept only plausible names: 2+ words, no digits, reasonable length
            authors = [a for a in raw if 2 <= len(a.split()) <= 5 and not any(c.isdigit() for c in a) and len(a) < 60]
        
        if not authors:
            # Heuristic: scan first-page lines for a comma-separated author block.
            lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
            
            # Find where title ends on first page
            title_idx = -1
            for idx, line in enumerate(lines[:10]):
                if len(title) > 8 and (title[:15].lower() in line.lower() or line.lower() in title.lower()):
                    title_idx = idx
                    break

            NAME_TOKEN_RE = re.compile(r'[A-Z][a-zé-]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-zé-]+){0,2}')
            
            search_start = max(0, title_idx + 1)
            for line in lines[search_start:search_start + 10]:
                clean_line = re.sub(r'[\d\*†‡§]+', '', line).strip()
                if len(clean_line) < 5 or len(clean_line) > 300:
                    continue
                if any(w in clean_line.lower() for w in ["abstract", "introduction", "arxiv", "http", "@", "university", "google", "deepmind"]):
                    continue
                tokens = NAME_TOKEN_RE.findall(clean_line)
                if len(tokens) >= 2:
                    authors = [t.strip() for t in tokens if 1 < len(t.split()) <= 5]
                    if len(authors) >= 2:
                        break
            
            # Last resort: run NER on the first 2000 chars of the first page
            if not authors:
                try:
                    from src.ner_engine import extract_persons_from_text
                    ner_names = extract_persons_from_text(first_page_text[:2000])
                    authors = [n for n in ner_names if 1 < len(n.split()) <= 5][:15]
                except Exception:
                    pass
        
        # Final cleanup
        cleaned = []
        seen = set()
        for a in authors:
            a = re.sub(r'[\d\*†‡§,]+$', '', a).strip()
            if a and a.lower() not in seen and len(a) > 3:
                seen.add(a.lower())
                cleaned.append(a)
        authors = cleaned

        # 4. Extract Year
        year = None
        creation_date = doc.metadata.get("creationDate", "")
        if creation_date and len(creation_date) >= 6:
            match = re.search(r'D:(\d{4})', creation_date)
            if match:
                year = int(match.group(1))
        
        if not year:
            matches = YEAR_REGEX.findall(first_page_text)
            if matches:
                year = int(matches[0])
            else:
                year = 2026

        # 5. Extract DOI
        doi = None
        doi_matches = DOI_REGEX.findall(full_text[:5000])
        if doi_matches:
            doi = doi_matches[0].rstrip(".,;)")
            if doi.lower().startswith("doi:"):
                doi = doi[4:]

        # 6. Extract Abstract
        abstract = None
        abstract_match = re.search(r'(?:abstract|summary)[:\-\s]+(.*?)(?:\n\s*(?:1\s+)?introduction|\n\s*(?:i\.\s+)?introduction|\n\s*background|\n\s*ii\.)', full_text, re.IGNORECASE | re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip()
        else:
            idx = full_text.lower().find("abstract")
            if idx != -1:
                abstract = full_text[idx+8:idx+1200].strip()
            else:
                abstract = full_text[:1000].strip()

        # 7. Extract References
        references = []
        ref_idx = -1
        for ref_keyword in ["references", "bibliography", "литература", "список литературы"]:
            idx = full_text.lower().rfind(ref_keyword)
            if idx > ref_idx:
                line_start = full_text.rfind("\n", 0, idx)
                line_end = full_text.find("\n", idx)
                line = full_text[line_start:line_end].strip()
                if len(line) < 30:
                    ref_idx = idx
        
        if ref_idx != -1:
            ref_section = full_text[ref_idx:]
            raw_refs = REF_SPLIT_REGEX.split(ref_section)
            for r in raw_refs:
                if not r:
                    continue
                r_clean = r.strip().replace("\n", " ")
                if len(r_clean) > 20 and not any(kw in r_clean.lower() for kw in ["references", "bibliography", "page"]):
                    r_clean = re.sub(r'\s+', ' ', r_clean)
                    references.append(r_clean)
        
        paper_id = doi if doi else slugify(title)
        
        paper = Paper(
            id=paper_id,
            title=title.strip(),
            authors=authors,
            year=year,
            doi=doi,
            abstract=abstract,
            file_path=source
        )
        paper.properties["pdf_parser"] = "fitz"
        
        return paper, references, full_text

    @staticmethod
    def compress_and_save_pdf(
        input_path: str,
        output_path: str,
        dpi_threshold: int,
        dpi_target: int,
        quality: int,
    ) -> None:
        """Recompresses images in a PDF using PyMuPDF and saves the result in a separate process to prevent segfaults."""
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(
            target=_compress_worker,
            args=(input_path, output_path, dpi_threshold, dpi_target, quality)
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"PDF compression process crashed with exit code {p.exitcode} (possible segmentation fault)"
            )


def _is_pdf_valid(path: str, expected_page_count: int) -> bool:
    import fitz
    import os
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        if hasattr(fitz, "JM_mupdf_warnings_store"):
            fitz.JM_mupdf_warnings_store.clear()
        
        with fitz.open(path) as doc:
            if doc.page_count != expected_page_count:
                return False
            for page in doc:
                page.get_text()
                page.get_drawings()
                
        if hasattr(fitz, "JM_mupdf_warnings_store"):
            critical_keywords = ["syntax error", "corrupt", "damaged", "unknown keyword", "error in content stream"]
            for warning in fitz.JM_mupdf_warnings_store:
                if any(kw in warning.lower() for kw in critical_keywords):
                    return False
        return True
    except Exception:
        return False


def _compress_worker(
    input_path: str,
    output_path: str,
    dpi_threshold: int,
    dpi_target: int,
    quality: int,
) -> None:
    # This runs in a separate process to isolate PyMuPDF segfaults from the main process
    import fitz
    import os

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Open original document to get expected page count
    doc = fitz.open(input_path)
    expected_page_count = doc.page_count
    actual_threshold = max(dpi_threshold, dpi_target + 1)

    # 2. Attempt aggressive compression (rewriting images)
    tmp_aggressive = None
    try:
        doc.rewrite_images(
            dpi_threshold=actual_threshold,
            dpi_target=dpi_target,
            quality=quality,
            lossy=True,
            lossless=True,
        )
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_aggressive = tmp.name
        
        doc.save(tmp_aggressive, garbage=4, deflate=True)
        doc.close()
        
        # Verify aggressive compression
        if _is_pdf_valid(tmp_aggressive, expected_page_count):
            shutil.move(tmp_aggressive, output_path)
            return
        else:
            raise RuntimeError("Aggressive PDF compression resulted in a corrupted file.")
    except Exception:
        # If document wasn't closed yet, close it
        try:
            doc.close()
        except Exception:
            pass
    finally:
        if tmp_aggressive and os.path.exists(tmp_aggressive):
            try:
                os.remove(tmp_aggressive)
            except Exception:
                pass

    # 3. Fallback: Safe compression (lossless, no image rewriting)
    doc = fitz.open(input_path)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_safe = tmp.name
        
    try:
        doc.save(tmp_safe, garbage=4, deflate=True)
        doc.close()
        
        # Verify safe compression
        if _is_pdf_valid(tmp_safe, expected_page_count):
            shutil.move(tmp_safe, output_path)
            return
        else:
            raise RuntimeError("Safe PDF compression resulted in a corrupted file.")
    except Exception as safe_err:
        raise RuntimeError(
            f"Failed to compress PDF: aggressive compression failed, "
            f"and safe fallback also failed ({safe_err})."
        )
    finally:
        if os.path.exists(tmp_safe):
            try:
                os.remove(tmp_safe)
            except Exception:
                pass


