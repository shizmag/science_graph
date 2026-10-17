from typing import Optional
from src.parsers.base import BaseParser
from src.parsers.pdf_parser import PDFParser
from src.parsers.marker_parser import MarkerPDFParser
from src.parsers.md_parser import MarkdownParser
from src.parsers.url_parser import UrlParser
from src.parsers.epub_parser import EPUBParser
from src.parsers.youtube_parser import YoutubeVideoParser

class ParserFactory:
    @staticmethod
    def get_parser(source: str, pdf_parser_type: Optional[str] = None) -> BaseParser:
        """
        Inspects the source string and returns the correct parser instance.
        """
        source_lower = source.lower().strip()
        if source_lower.startswith("http://") or source_lower.startswith("https://"):
            if "youtube.com" in source_lower or "youtu.be" in source_lower:
                return YoutubeVideoParser()
            return UrlParser()
        elif source_lower.endswith(".pdf"):
            from src.config import config
            ptype = (pdf_parser_type or config.pdf_parser).lower().strip()
            if ptype == "marker":
                return MarkerPDFParser()
            else:
                return PDFParser()
        elif source_lower.endswith(".md"):
            return MarkdownParser()
        elif source_lower.endswith(".epub"):
            return EPUBParser()
        else:
            raise ValueError(f"No parser available for source: {source}")
