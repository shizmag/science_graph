import os
import re
import logging
import sys
import types
import threading
import contextlib

# Monkeypatch huggingface_hub.dataclasses.strict and mock missing transformers module
# to ensure compatibility between marker-pdf 0.1.3 and transformers v5
if "transformers.utils.model_parallel_utils" not in sys.modules:
    mod = types.ModuleType("transformers.utils.model_parallel_utils")
    mod.get_device_map = lambda *a, **k: None
    mod.assert_device_map = lambda *a, **k: None
    sys.modules["transformers.utils.model_parallel_utils"] = mod

try:
    import huggingface_hub.dataclasses
    huggingface_hub.dataclasses.strict = lambda cls=None, *args, **kwargs: (lambda c: c) if cls is None else cls
    huggingface_hub.dataclasses.type_validator = lambda *a, **k: None
except ImportError:
    pass

# Patch torchvision functional normalize to accept numpy arrays (resolving torchvision v2 / transformers TypeError)
try:
    import torchvision.transforms.v2.functional as tvF2
    original_normalize2 = tvF2.normalize
    def patched_normalize2(inpt, *args, **kwargs):
        import torch
        import numpy as np
        if isinstance(inpt, np.ndarray):
            tensor_inpt = torch.from_numpy(inpt)
            out_tensor = original_normalize2(tensor_inpt, *args, **kwargs)
            return out_tensor.numpy()
        return original_normalize2(inpt, *args, **kwargs)
    tvF2.normalize = patched_normalize2
except Exception:
    pass

try:
    import torchvision.transforms.functional as tvF
    original_normalize = tvF.normalize
    def patched_normalize(inpt, *args, **kwargs):
        import torch
        import numpy as np
        if isinstance(inpt, np.ndarray):
            tensor_inpt = torch.from_numpy(inpt)
            out_tensor = original_normalize(tensor_inpt, *args, **kwargs)
            return out_tensor.numpy()
        return original_normalize(inpt, *args, **kwargs)
    tvF.normalize = patched_normalize
except Exception:
    pass

# Monkeypatch marker to fix ZeroDivisionErrors in code.py and equations.py
try:
    import marker.cleaners.code as m_code
    import fitz as pymupdf
    from marker.schema import Span, Line

    def patched_indent_blocks(blocks):
        span_counter = 0
        for page in blocks:
            for block in page.blocks:
                block_types = [span.block_type for line in block.lines for span in line.spans]
                if "Code" not in block_types:
                    continue

                lines = []
                min_left = 1000  # will contain x- coord of column 0
                col_width = 0  # width of 1 char
                for line in block.lines:
                    text = ""
                    min_left = min(line.bbox[0], min_left)
                    for span in line.spans:
                        if col_width == 0 and len(span.text) > 0:
                            col_width = (span.bbox[2] - span.bbox[0]) / len(span.text)
                        text += span.text
                    lines.append((pymupdf.Rect(line.bbox), text))

                if col_width <= 0:
                    col_width = 1.0

                block_text = ""
                blank_line = False
                for line in lines:
                    text = line[1]
                    prefix = " " * int((line[0].x0 - min_left) / col_width)
                    current_line_blank = len(text.strip()) == 0
                    if blank_line and current_line_blank:
                        # Don't put multiple blank lines in a row
                        continue

                    block_text += prefix + text + "\n"
                    blank_line = current_line_blank

                new_span = Span(
                    text=block_text,
                    bbox=block.bbox,
                    color=block.lines[0].spans[0].color,
                    span_id=f"{span_counter}_fix_code",
                    font=block.lines[0].spans[0].font,
                    block_type="Code"
                )
                span_counter += 1
                block.lines = [Line(spans=[new_span], bbox=block.bbox)]

    m_code.indent_blocks = patched_indent_blocks
except Exception:
    pass

try:
    import marker.cleaners.equations as m_eq
    from PIL import Image, ImageDraw

    def patched_mask_bbox(png_image, bbox, selected_bboxes):
        mask = Image.new('L', png_image.size, 0)  # 'L' mode for grayscale
        draw = ImageDraw.Draw(mask)
        first_x = bbox[0]
        first_y = bbox[1]
        bbox_height = max(bbox[3] - bbox[1], 1e-5)
        bbox_width = max(bbox[2] - bbox[0], 1e-5)

        for box in selected_bboxes:
            # Fit the box to the selected region
            new_box = (box[0] - first_x, box[1] - first_y, box[2] - first_x, box[3] - first_y)
            # Fit mask to image bounds versus the pdf bounds
            resized = (
               new_box[0] / bbox_width * png_image.size[0],
               new_box[1] / bbox_height * png_image.size[1],
               new_box[2] / bbox_width * png_image.size[0],
               new_box[3] / bbox_height * png_image.size[1]
            )
            draw.rectangle(resized, fill=255)

        result = Image.composite(png_image, Image.new('RGBA', png_image.size, 'white'), mask)
        return result

    m_eq.mask_bbox = patched_mask_bbox
except Exception:
    pass

# Suppress noisy debug logs from third-party libraries used by Marker
logging.getLogger("ocrmypdf").setLevel(logging.WARNING)
logging.getLogger("pdfminer").setLevel(logging.WARNING)

from typing import List, Tuple

from src.models import Paper, slugify
from src.parsers.base import BaseParser
from src.parsers.pdf_parser import PDFParser
from src import console as con

logger = logging.getLogger(__name__)

# Cached marker models loaded on demand
_marker_models = None
_marker_session_depth = 0
_marker_session_lock = threading.Lock()

def get_marker_models():
    global _marker_models
    if _marker_models is None:
        print("MARKER_LOAD_START", {"pid": os.getpid()})
        con.info("Loading Marker OCR/Layout models into memory (Mac Mini M4)...")
        from marker.models import load_all_models
        _marker_models = load_all_models()
        con.success("Marker models loaded successfully.")
        print("MARKER_LOAD_DONE", {"pid": os.getpid()})
    return _marker_models

def shutdown_marker():
    global _marker_models
    if _marker_models is not None:
        print("MARKER_UNLOAD", {"pid": os.getpid()})
        con.info("Unloading Marker models...")
        del _marker_models
        _marker_models = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass
        con.success("Marker models unloaded successfully.")

@contextlib.contextmanager
def marker_session():
    global _marker_session_depth
    with _marker_session_lock:
        _marker_session_depth += 1
    try:
        yield
    finally:
        with _marker_session_lock:
            _marker_session_depth -= 1
            if _marker_session_depth == 0:
                shutdown_marker()

class MarkerPDFParser(BaseParser):
    def parse(self, source: str) -> Tuple[Paper, List[str], str]:
        """
        Parses a PDF file using the Marker engine to produce high-quality Markdown text,
        falling back to Fitz if Marker fails or has issues.
        """
        with marker_session():
            if not os.path.exists(source):
                raise FileNotFoundError(f"PDF file not found: {source}")

            # 1. Use the legacy Fitz-based parser first to get clean metadata
            legacy_parser = PDFParser()
            try:
                paper, references, legacy_text = legacy_parser.parse(source)
            except Exception as e:
                logger.warning(f"Legacy parser failed to extract metadata from {source}: {e}")
                paper = Paper(
                    id=slugify(os.path.basename(source)),
                    title=os.path.basename(source).replace(".pdf", ""),
                    authors=[],
                    year=2026,
                    doi=None,
                    abstract="",
                    file_path=source
                )
                references = []
                legacy_text = ""

            # 2. Extract rich Markdown text using Marker
            try:
                models = get_marker_models()
                con.info(f"Extracting Markdown text from {os.path.basename(source)} using Marker...")
                
                from marker.convert import convert_single_pdf
                
                # Request Russian to enable eng+rus OCR via Tesseract
                full_text, out_metadata = convert_single_pdf(
                    source,
                    models,
                    metadata={"language": "Russian"},
                    parallel_factor=1
                )
                
                full_text = full_text.strip()
                
                # Extract references from the Marker-generated Markdown text
                marker_references = self._extract_references_from_markdown(full_text)
                if marker_references:
                    references = marker_references
                
                paper.properties["pdf_parser"] = "marker"
                    
            except Exception as e:
                con.warning(f"Marker PDF parsing failed for {source}: {e}. Falling back to standard PDF parser.")
                logger.warning(f"Marker PDF parsing failed for {source}: {e}. Falling back to standard PDF parser.", exc_info=True)
                full_text = legacy_text
                paper.properties["pdf_parser"] = "fitz"
                paper.properties["pdf_parser_fallback"] = True
                paper.properties["pdf_parser_error"] = str(e)

            # Update paper file path and return
            paper.file_path = source
            return paper, references, full_text

    def _extract_references_from_markdown(self, markdown_text: str) -> List[str]:
        """Heuristic to extract clean references list from Marker's markdown output."""
        references = []
        ref_idx = -1
        
        # Look for references section heading
        for ref_keyword in ["references", "bibliography", "литература", "список литературы"]:
            idx = markdown_text.lower().rfind(ref_keyword)
            if idx > ref_idx:
                # Ensure it looks like a section start (e.g. preceded by newline or #)
                line_start = markdown_text.rfind("\n", 0, idx)
                line_end = markdown_text.find("\n", idx)
                line = markdown_text[line_start:line_end].strip()
                if len(line) < 30:
                    ref_idx = idx

        if ref_idx != -1:
            ref_section = markdown_text[ref_idx:]
            # Split by markdown list items or numbering patterns
            # e.g., \n1. or \n- or \n[1]
            split_patterns = re.compile(r'\n(?:[*\-+]|\d+\.|\s*\[\d+\])\s+')
            raw_refs = split_patterns.split(ref_section)
            for r in raw_refs:
                if not r:
                    continue
                r_clean = r.strip().replace("\n", " ")
                # Clean multiple spaces
                r_clean = re.sub(r'\s+', ' ', r_clean)
                if len(r_clean) > 20 and not any(kw in r_clean.lower() for kw in ["references", "bibliography", "page"]):
                    references.append(r_clean)

        return references
