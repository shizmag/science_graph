from abc import ABC, abstractmethod
from typing import List, Tuple
from src.models import Paper

class BaseParser(ABC):
    @abstractmethod
    def parse(self, source: str) -> Tuple[Paper, List[str], str]:
        """
        Parses a document source (file path or URL).
        Returns:
            paper: Paper metadata domain model
            references/wiki_links: List of linked references/wiki links (Obsidian targets)
            full_text: Extracted plain text body
        """
        pass
