import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

def slugify(text: str) -> str:
    if not text:
        return ""
    # Remove HTML or other weird chars, then lowercase and replace spaces/hyphens with underscores
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    return re.sub(r'[\s-]+', '_', text).strip('_')

@dataclass
class Paper:
    id: str  # DOI or hash/slug of the title
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    file_path: Optional[str] = None
    created_at: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Author:
    id: str  # slug of name (e.g. "vladimir_kasterin")
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Concept:
    id: str  # slug of concept (e.g. "self_attention")
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    id: str  # paper_id#index
    paper_id: str
    text_content: str
    page_number: int
    embedding: Optional[List[float]] = None
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None

@dataclass
class Edge:
    source_id: str
    target_id: str
    type: str  # CITES, AUTHORED, MENTIONS_CONCEPT, RELATED_TO
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Institution:
    id: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Dataset:
    id: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CodeRepository:
    id: str
    name: str
    url: str
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class JournalConference:
    id: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UserNote:
    id: str
    title: str
    properties: Dict[str, Any] = field(default_factory=dict)
