from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    question: str
    limit: Optional[int] = 5
    cloud: Optional[bool] = False
    paper_id: Optional[str] = None

class NoteCreate(BaseModel):
    title: str
    content: str
    authors: Optional[List[str]] = Field(default_factory=list)
    tags: Optional[List[str]] = Field(default_factory=list)

class NoteResponse(BaseModel):
    id: str
    title: str
    created_at: Optional[str] = None
    authors: List[str] = []
    summary: Optional[str] = None
    abstract: Optional[str] = None

class OpenFileRequest(BaseModel):
    file_path: str

class GraphNode(BaseModel):
    id: str
    label: str
    title: str
    color: str
    size: int
    group: str
    shape: str = "dot"
    created_at: Optional[str] = None
    source_type: str = "paper"
    full_title: str

class GraphEdgeFont(BaseModel):
    size: int = 8
    align: str = "top"

class GraphEdgeColor(BaseModel):
    color: str = "#adb5bd"
    highlight: str = "#74c0fc"

class GraphEdge(BaseModel):
    from_node: str = Field(..., alias="from")
    to: str
    label: str
    arrows: str = "to"
    font: GraphEdgeFont = GraphEdgeFont()
    color: GraphEdgeColor = GraphEdgeColor()

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True
    )

class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]

class SearchResultItem(BaseModel):
    id: str
    title: str
    year: Optional[int] = None
    source_type: str = "paper"

class SearchResponse(BaseModel):
    results: List[SearchResultItem]

class ExtensionStats(BaseModel):
    extension: str
    size: int
    count: int

class SourceStats(BaseModel):
    source: str
    size: int
    count: int

class StorageStats(BaseModel):
    storage_dir: str
    total_size: int
    extensions: List[ExtensionStats]
    sources: List[SourceStats]

class StatsResponse(BaseModel):
    papers: int
    authors: int
    concepts: int
    edges: int
    indexed_papers: int = 0
    mentioned_papers: int = 0
    storage: Optional[StorageStats] = None

class ConceptItem(BaseModel):
    id: str
    name: str

class TagItem(BaseModel):
    id: str
    name: str

class CitationItem(BaseModel):
    id: str
    title: str

class PaperDetailResponse(BaseModel):
    type: str = "paper"
    id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    source_type: str = "paper"
    concepts: List[ConceptItem] = Field(default_factory=list)
    tags: List[TagItem] = Field(default_factory=list)
    citations: List[CitationItem] = Field(default_factory=list)
    cited_by: List[CitationItem] = Field(default_factory=list)
    file_path: Optional[str] = None
    summary: Optional[str] = None
    created_at: Optional[str] = None
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)

class PaperBrief(BaseModel):
    id: str
    title: str
    source_type: str = "paper"

class AuthorDetailResponse(BaseModel):
    type: str = "author"
    id: str
    name: str
    papers: List[PaperBrief] = Field(default_factory=list)
    papers_count: int

class ConceptDetailResponse(BaseModel):
    type: str
    id: str
    name: str
    description: str
    papers: List[PaperBrief] = Field(default_factory=list)
    related: List[ConceptItem] = Field(default_factory=list)

class UploadResponse(BaseModel):
    status: str
    id: str
    filename: str

class NoteCreateResponse(BaseModel):
    status: str
    id: str
    file_path: str

class OpenFileResponse(BaseModel):
    status: str
    message: str


class UrlIndexRequest(BaseModel):
    url: str


class UrlIndexResponse(BaseModel):
    status: str
    id: str
    title: str


class LibraryPaperItem(BaseModel):
    id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    source_type: str = "paper"
    summary: Optional[str] = None
    concepts: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    file_path: Optional[str] = None
    created_at: Optional[str] = None


class LibraryResponse(BaseModel):
    total: int
    page: int
    limit: int
    results: List[LibraryPaperItem]



