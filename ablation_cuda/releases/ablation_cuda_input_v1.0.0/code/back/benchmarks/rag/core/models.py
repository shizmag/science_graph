"""
Science Graph — Pydantic Models for RAG Layer Outputs.
Defines strict schemas for retrieval, generation, and evaluation results.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict


class RetrievedChunk(BaseModel):
    """Represents a single retrieved text chunk from a paper."""
    model_config = ConfigDict(extra="allow")

    id: str
    paper_id: str
    page_number: int
    text_content: str
    score: float


class ComponentPerformance(BaseModel):
    """Represents calls and latency of a single component."""
    model_config = ConfigDict(extra="allow")

    calls: int = 0
    time_sec: float = 0.0


class StageMetrics(BaseModel):
    """Represents RAG stage metrics (I/O calls, performance breakdowns)."""
    model_config = ConfigDict(extra="allow")

    components: Optional[Dict[str, ComponentPerformance]] = None
    total_io_calls: int = 0
    total_latency: Optional[float] = None


class ShannonDiagnostics(BaseModel):
    """Represents Shannon diagnostics and logit-level telemetry metrics."""
    model_config = ConfigDict(extra="allow")

    h_rank_pre_rerank: Optional[float] = 0.0
    h_rank_post_rerank: Optional[float] = 0.0
    h_lexical_pre_trim: Optional[float] = 0.0
    h_lexical_post_trim: Optional[float] = 0.0
    h_graph_relation_type: Optional[float] = 0.0
    h_graph_degree: Optional[float] = 0.0
    h_gen: Optional[float] = 0.0
    h_citation: Optional[float] = 0.0
    n_citation_tokens: Optional[int] = 0
    delta_h_gen: Optional[float] = 0.0
    msp: Optional[float] = None
    avg_msp: Optional[float] = None
    logit_margin: Optional[float] = None
    avg_logit_margin: Optional[float] = None
    first_token_margin: Optional[float] = None
    first_token_msp: Optional[float] = None
    citation_entropy: Optional[float] = None
    ll_rag: Optional[float] = None
    ll_base: Optional[float] = None
    clr: Optional[float] = None


class BaselineOutput(BaseModel):
    """Represents the results of running a single baseline on a test case."""
    model_config = ConfigDict(extra="allow")

    status: str = "success"
    latency_sec: Optional[float] = 0.0
    retrieved_papers: List[str] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    # Shannon stage inputs captured at retrieve/trim boundaries
    pre_rerank_scores: Optional[List[float]] = None
    context_text: Optional[str] = None
    context_graph: Optional[str] = None
    graph_relations: Optional[List[Dict[str, Any]]] = None
    trimmed_text: Optional[str] = ""
    trimmed_graph: Optional[str] = ""
    enrichment_block: Optional[str] = ""
    generated_answer: Optional[str] = None
    baseline_config: Optional[Dict[str, Any]] = None
    metrics: Optional[StageMetrics] = None
    eval_metrics: Optional[Dict[str, Any]] = None
    shannon_diagnostics: Optional[Dict[str, Any]] = None
    trace: Optional[Dict[str, Any]] = None
    msp: Optional[float] = None
    avg_msp: Optional[float] = None
    logit_margin: Optional[float] = None
    avg_logit_margin: Optional[float] = None
    first_token_margin: Optional[float] = None
    first_token_msp: Optional[float] = None
    citation_entropy: Optional[float] = None
    ll_rag: Optional[float] = None
    ll_base: Optional[float] = None
    clr: Optional[float] = None


class TestCaseOutput(BaseModel):
    """Represents a test case and its evaluated baselines."""
    model_config = ConfigDict(extra="allow")

    id: str
    query: str = ""
    category: str = "default"
    golden_answer: Optional[str] = ""
    expected_papers: List[str] = Field(default_factory=list)
    is_answerable: bool = True
    baselines: Dict[str, BaselineOutput] = Field(default_factory=dict)


class ReportOutput(BaseModel):
    """Unified container for full layer reports (with optional metadata and summaries)."""
    model_config = ConfigDict(extra="allow")

    metadata: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    results: List[TestCaseOutput] = Field(default_factory=list)


def parse_report(data: Any) -> ReportOutput:
    """Parses raw deserialized YAML (list or dict) into a unified ReportOutput model."""
    if isinstance(data, list):
        # retrieved_contexts.yaml is a list of test cases at the root level
        return ReportOutput(results=[TestCaseOutput.model_validate(item) for item in data])
    elif isinstance(data, dict):
        if "results" in data:
            return ReportOutput.model_validate(data)
        else:
            # Handle dictionary without explicit "results" key (e.g. metadata-only or dynamic structure)
            return ReportOutput.model_validate(data)
    else:
        raise ValueError(f"Unsupported report format: {type(data)}")


def load_report_file(file_path: Path) -> ReportOutput:
    """Loads a YAML report file and parses it into a ReportOutput model."""
    import yaml
    if not file_path.exists():
        raise FileNotFoundError(f"Report file {file_path} does not exist.")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return parse_report(data)
