import os
import copy
import yaml
from pathlib import Path
from typing import Optional, List

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "pdf-graph-analyzer"
DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "pdf-graph-analyzer"
CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.yaml"

DEFAULT_CONFIG = {
    "db_path": str(DEFAULT_DATA_DIR / "graph.db"),
    "archive_dir": str(DEFAULT_DATA_DIR / "archive"),
    "pdf_parser": "marker",
    "hf_token": "",
    "llm": {
        "provider": "mlx",
        "max_tokens": 1000,
        "model_max_context": 4096,
        "temp": 0.1,
        "request_delay": 1.0,
        "retry_backoff": 2.0,
        "chunk_pool_size": 4,
        "max_expanded_queries": 3,
        "hyde_enabled": False,
        "hyde_max_tokens": 300,
        "hyde_count": 1,
        "enable_mtp": True,
        "auto_disable_mtp_if_missing_files": True,
        # Task-specific input token limits
        "extraction_input_limit": 5000,
        "clustering_input_limit": 6000,
        "synthesis_input_limit": 5000,
        # Task-specific output token limits
        "extraction_output_limit": 2048,
        "clustering_output_limit": 1500,
        "synthesis_output_limit": 1500,
        # Nested split configurations
        "local": {
            "model_path": str(Path.home() / "models" / "llm" / "gemma-3-text-12b-it-4bit"),
            "rag_model_path": str(Path.home() / "models" / "llm" / "gemma-3-text-12b-it-4bit"),
        },
        "gguf": {
            "n_gpu_layers": -1,
            "n_ctx": 4096,
        },
        "cloud": {
            "provider": "openai",
            "model_name": "google/gemini-2.5-flash",
            "cheap_model_name": "google/gemini-2.5-flash",
            "rag_model_name": "google/gemini-2.5-flash",
            "api_key": "",
            "base_url": "https://openrouter.ai/api/v1",
        },
        "evaluation": {
            "concurrency": 1,
            "rpm": 10,
            "retries": 5
        }
    },
    "embedding": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "child_chunk_size": 300,
        "child_chunk_overlap": 50,
        "parent_chunk_size": 2500,
        "parent_chunk_overlap": 200
    },
    "spacy": {
        "model_name": "en_core_web_sm"
    },
    "ner": {
        "model_name": "dslim/bert-base-NER"
    },
    "reranker": {
        "model_name": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    },
    "pdf_compression": {
        "enabled": True,
        "dpi_threshold": 151,
        "dpi_target": 150,
        "quality": 75
    },
    "rag_components": {
        "intent_classifier": False,
        "graph_ontology_lookup": True,
        "llm_query_expansion": True,
        "hyde": True,
        "lexical_search": True,
        "dense_search": True,
        "dynamic_alpha_blending": True,
        "rrf": True,
        "graph_expansion": True,
        "reranker": True,
        "score_blending": True,
        "context_trimming": True,
        "citation_repair": True,
        "graph_neighbors_in_rrf": False,
        "graph_concept_retrieval": False,
        "graph_bridge_retrieval": False,
        "graph_selected_sources_card": False,
        "graph_retrieval_trace": False,
        "shannon_estimator_enabled": True,
        "deduplicate_parent_chunks": True,
    },
    "graph_retrieval": {
        "enabled": False,
        "concept_retrieval_enabled": False,
        "bridge_retrieval_enabled": False,
        "selected_sources_card_enabled": False,
        "trace_enabled": False,
        "chunks_per_graph_paper": 1,
        "max_graph_candidate_papers": "auto",
        "candidate_budget_mode": "fixed",
        "max_graph_chunk_candidates": None,
        "use_concept_idf": True,
        "single_token_concept_policy": "idf_if_no_multitoken",
        "use_author_edges_for_retrieval": False,
        "allowed_retrieval_edge_types": [
            "CITES",
            "CITED_BY",
            "MENTIONS_CONCEPT",
            "RELATED_TO",
            "HAS_TAG"
        ]
    },
    "hyperparameters": {
        "rag": {
            # Weight of Cross-Encoder reranker score in final score blending
            "score_blend_reranker_weight": 0.7,
            # Weight of Reciprocal Rank Fusion (RRF) score in final score blending
            "score_blend_rrf_weight": 0.3,
            # Smoothing constant k used in Reciprocal Rank Fusion (RRF)
            "rrf_k": 60.0,
            # BM25 score threshold below which low alpha (lexical search weight) is applied
            "dynamic_alpha_threshold_low": 1.0,
            # Alpha value (lexical search weight) when BM25 score is below the low threshold
            "dynamic_alpha_val_low": 0.2,
            # BM25 score threshold below which mid alpha (lexical search weight) is applied
            "dynamic_alpha_threshold_mid": 3.0,
            # Alpha value (lexical search weight) when BM25 score is below the mid threshold
            "dynamic_alpha_val_mid": 0.5,
            # Alpha value (lexical search weight) when BM25 score is high
            "dynamic_alpha_val_high": 1.0,
        },
        "graph": {
            # Base probability of transitioning to/crawling a neighboring node in graph expansion
            "p_base": 0.75,
            # Decay factor gamma for graph crawling limits and authority aging
            "gamma": 0.5,
            # Convergence stop threshold: graph crawling halts when expected transition count K_n < threshold
            "crawl_stop_threshold": 1.0,
            # Minimum semantic score required to crawl a neighboring non-note node
            "semantic_score_threshold": 0.4,
            # Top-P threshold for crawling a neighboring non-note node (nucleus filtering)
            "semantic_score_top_p": 0.9,
            # Minimum sigmoid-scaled score required for a newly fetched text chunk to be relevant
            "sigmoid_score_threshold": 0.4,
            # Top-P threshold for a newly fetched text chunk to be relevant (nucleus filtering)
            "sigmoid_score_top_p": 0.9,
            # Minimum sigmoid score required to classify a gathered graph fact as essential
            "essential_fact_threshold": 0.5,
            # Sigmoid scaling slope (factor) for Cross-Encoder reranker score logit calibration
            "sigmoid_slope": -25.0,
            # Sigmoid scaling center offset for Cross-Encoder reranker score logit calibration
            "sigmoid_center": 0.5,
            # Weight for AUTHORED relationship type
            "weight_authored": 0.8,
            # Weight for CITES relationship type
            "weight_cites": 0.7,
            # Weight for MENTIONS_CONCEPT relationship type
            "weight_mentions_concept": 0.6,
            # Default weight for other relationship types
            "weight_default": 0.5,
            # Order/depth of graph neighbors to retrieve for Baseline 6 RRF/cross-encoding
            "b6_graph_neighbors_order": 2,
        },
        "bm25": {
            # BM25 term frequency saturation parameter k1
            "k1": 1.5,
            # BM25 document length normalization parameter b
        }
    }
}


def resolve_existing_model_path(path_str: str) -> str:
    """Fallback resolver checking external volume paths (/Volumes/happy-disk/models) or MODELS_DIR if missing."""
    if not path_str:
        return path_str
    p = Path(path_str)
    if p.exists():
        return str(p)

    # 1. Check environment variable override
    models_dir = os.environ.get("MODELS_DIR")
    if models_dir:
        name = p.name
        candidate = Path(models_dir) / name
        if candidate.exists():
            return str(candidate)
        parent_name = p.parent.name
        if parent_name:
            candidate = Path(models_dir) / parent_name / name
            if candidate.exists():
                return str(candidate)

    # 2. Check /Volumes/happy-disk/models
    happy_disk = Path("/Volumes/happy-disk/models")
    if happy_disk.exists():
        name = p.name
        for sub in ["", "llm", "embeddings"]:
            candidate = (happy_disk / sub / name) if sub else (happy_disk / name)
            if candidate.exists():
                return str(candidate)

    return path_str


class Config:
    def __init__(self):
        self.config_dir = DEFAULT_CONFIG_DIR
        self.data_dir = DEFAULT_DATA_DIR
        self.config_file = CONFIG_FILE
        
        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = self._load_or_create_config()
        
        # Apply HF token to environment if set
        hf_token = self.data.get("hf_token", "")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        
        # Create archive dir if defined
        Path(self.data["archive_dir"]).mkdir(parents=True, exist_ok=True)

    def _load_or_create_config(self) -> dict:
        if not self.config_file.exists():
            config_template = f"""# Configuration for PDF Graph Analyzer

# Path to the SQLite database file
db_path: "{DEFAULT_CONFIG['db_path']}"

# Directory where local archives of websites/PDFs are stored
archive_dir: "{DEFAULT_CONFIG['archive_dir']}"

# PDF parser to use: 'marker' (modern OCR/Markdown converter) or 'fitz' (legacy PyMuPDF)
pdf_parser: "{DEFAULT_CONFIG['pdf_parser']}"

# HuggingFace token for downloading gated models/embeddings (optional)
hf_token: ""

# Large Language Model (LLM) configuration
llm:
  # Provider: 'mlx' (for local Apple Silicon), 'gguf' (for local GGUF models via llama.cpp) or 'openai' (for OpenAI / OpenRouter / compatible APIs)
  provider: "mlx"

  # Global default maximum output tokens for LLM response
  max_tokens: 1000

  # Maximum context window size for the LLM
  model_max_context: 4096

  # Default temperature (0.0 = deterministic, 1.0 = creative)
  temp: 0.1

  # Delay (in seconds) between requests to LLM / API providers
  request_delay: 1.0

  # Wait timeout (in seconds) before retrying a failed provider request
  retry_backoff: 2.0

  # Number of concurrent chunks to process in parallel via LLM
  chunk_pool_size: 4

  # Maximum number of expanded queries for search (including original, 1 = disabled)
  max_expanded_queries: 3

  # Enable Hypothetical Document Embeddings (HyDE) for retrieval improvement
  hyde_enabled: false

  # Maximum tokens generated for the hypothetical answer
  hyde_max_tokens: 300

  # Number of hypothetical answers to generate
  hyde_count: 1

  # Enable speculative decoding (MTP) mode for local OpenAI-compatible backends
  enable_mtp: true

  # Automatically disable MTP mode if speculative decoding files (e.g. mtp.safetensors) are missing
  auto_disable_mtp_if_missing_files: true

  # Local model settings (used if provider is 'mlx' or 'gguf')
  # Note: For 'mlx', model_path is a directory; for 'gguf', model_path is a file path to the .gguf model.
  local:
    model_path: "{DEFAULT_CONFIG['llm']['local']['model_path']}"
    # Separate fine-tuned model path used specifically for RAG generation (optional)
    rag_model_path: "{DEFAULT_CONFIG['llm']['local']['rag_model_path']}"

  # GGUF-specific settings (used if provider is 'gguf')
  gguf:
    # Number of model layers to offload to GPU (-1 offloads all layers to Metal/CUDA)
    n_gpu_layers: -1
    # Context size for GGUF model (overrides model_max_context if set)
    n_ctx: 4096

  # Cloud model settings (used if provider is 'openai')
  cloud:
    provider: "openai"
    model_name: "google/gemini-2.5-flash"
    # Separate model name used specifically for RAG generation (optional)
    rag_model_name: "google/gemini-2.5-flash"
    api_key: ""
    base_url: "https://openrouter.ai/api/v1"

  # Evaluation / LLM-as-a-judge settings
  evaluation:
    concurrency: 1
    rpm: 10
    retries: 5

  # Task-specific input token limits (used to dynamically truncate inputs to fit context)
  extraction_input_limit: 5000
  clustering_input_limit: 6000
  synthesis_input_limit: 5000

  # Task-specific output token limits (passed to model during generation)
  extraction_output_limit: 2048
  clustering_output_limit: 1500
  synthesis_output_limit: 1500

# Embedding model configuration (used for vector search and indexing)
embedding:
  # HuggingFace model name for sentence embeddings
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

  # Number of characters per text chunk
  chunk_size: 1000

  # Number of characters overlap between consecutive chunks
  chunk_overlap: 200

# spaCy model configuration (used for lemmatization)
spacy:
  # spaCy model name (e.g. "en_core_web_sm") or path
  model_name: "en_core_web_sm"

# NER model configuration (used for name extraction)
ner:
  # NER model name or HuggingFace repo ID or local path
  model_name: "dslim/bert-base-NER"

# Reranker model configuration (used for Cross-Encoder reranking)
reranker:
  # Cross-Encoder model name or HuggingFace repo ID or local path
  model_name: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# PDF compression settings (used to downsample high-DPI scanned PDFs)
pdf_compression:
  enabled: true
  dpi_threshold: 151
  dpi_target: 150
  quality: 75

# RAG components configuration for benchmarking (Scenario 1, 2, 3)
rag_components:
  intent_classifier: false
  graph_ontology_lookup: true
  llm_query_expansion: true
  hyde: true
  lexical_search: true
  dense_search: true
  dynamic_alpha_blending: true
  rrf: true
  graph_expansion: true
  reranker: true
  score_blending: true
  context_trimming: true
  citation_repair: true

# Deterministic Graph Retrieval configuration
graph_retrieval:
  enabled: false
  concept_retrieval_enabled: false
  bridge_retrieval_enabled: false
  selected_sources_card_enabled: false
  trace_enabled: false
  chunks_per_graph_paper: 1
  max_graph_candidate_papers: "auto"
  candidate_budget_mode: "mirror_base"
  max_graph_chunk_candidates: null

# Fine-grained hyperparameters for RAG, graph crawling, and search algorithms
hyperparameters:
  rag:
    # Weight of Cross-Encoder reranker score in final score blending
    score_blend_reranker_weight: 0.7
    # Weight of Reciprocal Rank Fusion (RRF) score in final score blending
    score_blend_rrf_weight: 0.3
    # Smoothing constant k used in Reciprocal Rank Fusion (RRF)
    rrf_k: 60.0
    # BM25 score threshold below which low alpha (lexical search weight) is applied
    dynamic_alpha_threshold_low: 1.0
    # Alpha value (lexical search weight) when BM25 score is below the low threshold
    dynamic_alpha_val_low: 0.2
    # BM25 score threshold below which mid alpha (lexical search weight) is applied
    dynamic_alpha_threshold_mid: 3.0
    # Alpha value (lexical search weight) when BM25 score is below the mid threshold
    dynamic_alpha_val_mid: 0.5
    # Alpha value (lexical search weight) when BM25 score is high
    dynamic_alpha_val_high: 1.0
  graph:
    # Base probability of transitioning to/crawling a neighboring node in graph expansion
    p_base: 0.75
    # Decay factor gamma for graph crawling limits and authority aging
    gamma: 0.5
    # Convergence stop threshold: graph crawling halts when expected transition count K_n < threshold
    crawl_stop_threshold: 1.0
    # Minimum semantic score required to crawl a neighboring non-note node
    semantic_score_threshold: 0.4
    # Top-P threshold for crawling a neighboring non-note node (nucleus filtering)
    semantic_score_top_p: 0.9
    # Minimum sigmoid-scaled score required for a newly fetched text chunk to be relevant
    sigmoid_score_threshold: 0.4
    # Top-P threshold for a newly fetched text chunk to be relevant (nucleus filtering)
    sigmoid_score_top_p: 0.9
    # Minimum sigmoid score required to classify a gathered graph fact as essential
    essential_fact_threshold: 0.5
    # Sigmoid scaling slope (factor) for Cross-Encoder reranker score logit calibration
    sigmoid_slope: -25.0
    # Sigmoid scaling center offset for Cross-Encoder reranker score logit calibration
    sigmoid_center: 0.5
    # Heuristic weight for AUTHORED relationship type
    weight_authored: 0.8
    # Heuristic weight for CITES relationship type
    weight_cites: 0.7
    # Heuristic weight for MENTIONS_CONCEPT relationship type
    weight_mentions_concept: 0.6
    # Default heuristic weight for other relationship types
    weight_default: 0.5
  bm25:
    # BM25 term frequency saturation parameter k1
    k1: 1.5
    # BM25 document length normalization parameter b
    b: 0.75
"""
            with open(self.config_file, "w", encoding="utf-8") as f:
                f.write(config_template)
            return DEFAULT_CONFIG
        
        with open(self.config_file, "r", encoding="utf-8") as f:
            try:
                loaded = yaml.safe_load(f) or {}
                # Merge defaults for missing keys
                merged = {**DEFAULT_CONFIG}
                for k, v in loaded.items():
                    if isinstance(v, dict) and k in merged:
                        merged[k] = {**merged[k], **v}
                    else:
                        merged[k] = v
                return merged
            except Exception:
                return DEFAULT_CONFIG

    def save(self) -> None:
        """Saves current configuration to the config.yaml file."""
        with open(self.config_file, "w", encoding="utf-8") as f:
            yaml.dump(self.data, f, default_flow_style=False, allow_unicode=True)

    def init_config(self) -> None:
        """Brings the config.yaml file up-to-date with DEFAULT_CONFIG, preserving user values."""
        loaded = {}
        if self.config_file.exists():
            with open(self.config_file, "r", encoding="utf-8") as f:
                try:
                    loaded = yaml.safe_load(f) or {}
                except Exception:
                    pass

        def sync_dict(template: dict, current: dict) -> dict:
            result = {}
            for k, v in template.items():
                if k in current:
                    if isinstance(v, dict) and isinstance(current[k], dict):
                        result[k] = sync_dict(v, current[k])
                    else:
                        result[k] = current[k]
                else:
                    result[k] = v
            return result

        self.data = sync_dict(DEFAULT_CONFIG, loaded)
        self.save()

    @property
    def hf_token(self) -> str:
        return self.data.get("hf_token", "")

    @property
    def db_path(self) -> str:
        return self.data["db_path"]

    @property
    def archive_dir(self) -> str:
        return self.data["archive_dir"]

    @property
    def pdf_parser(self) -> str:
        return self.data.get("pdf_parser", "marker")

    @property
    def llm_provider(self) -> str:
        return self.data["llm"].get("provider", "mlx")

    @property
    def llm_api_key(self) -> str:
        return self.llm_cloud_api_key

    @property
    def llm_base_url(self) -> str:
        return self.llm_cloud_base_url

    @property
    def llm_model_path(self) -> str:
        return self.llm_local_model_path

    @property
    def llm_local_model_path(self) -> str:
        local_cfg = self.data["llm"].get("local", {})
        path_val = ""
        if isinstance(local_cfg, dict):
            path_val = local_cfg.get("model_path") or ""
        if not path_val:
            path_val = self.data["llm"].get("model_path", "")
        return resolve_existing_model_path(path_val)

    @property
    def llm_local_rag_model_path(self) -> str:
        local_cfg = self.data["llm"].get("local", {})
        if isinstance(local_cfg, dict):
            rag_path = local_cfg.get("rag_model_path")
            model_path = local_cfg.get("model_path")
            default_path = str(Path.home() / "models" / "llm" / "gemma-3-text-12b-it-4bit")
            if not rag_path or rag_path == model_path or rag_path == default_path:
                return self.llm_local_model_path
            return resolve_existing_model_path(rag_path)
        return self.llm_local_model_path

    @property
    def llm_local_base_url(self) -> str:
        local_cfg = self.data["llm"].get("local", {})
        if isinstance(local_cfg, dict):
            nested_url = local_cfg.get("base_url")
            if nested_url:
                return nested_url
        top_url = self.data["llm"].get("base_url")
        if top_url:
            return top_url
        return self.llm_local_model_path

    @property
    def llm_cloud_model_name(self) -> str:
        cloud_cfg = self.data["llm"].get("cloud", {})
        if isinstance(cloud_cfg, dict):
            nested_name = cloud_cfg.get("model_name")
            if nested_name:
                return nested_name
        top_name = self.data["llm"].get("model_name")
        if top_name:
            return top_name
        return self.data["llm"].get("model_path", "")

    @property
    def llm_cloud_rag_model_name(self) -> str:
        cloud_cfg = self.data["llm"].get("cloud", {})
        if isinstance(cloud_cfg, dict):
            rag_name = cloud_cfg.get("rag_model_name")
            model_name = cloud_cfg.get("model_name")
            default_name = "google/gemini-2.5-flash"
            if not rag_name or rag_name == model_name or rag_name == default_name:
                return self.llm_cloud_model_name
            return cloud_cfg.get("rag_model_name", self.llm_cloud_model_name)
        return self.llm_cloud_model_name

    @property
    def llm_cloud_api_key(self) -> str:
        cloud_cfg = self.data["llm"].get("cloud", {})
        if isinstance(cloud_cfg, dict):
            nested_key = cloud_cfg.get("api_key")
            if nested_key:
                return nested_key
        return self.data["llm"].get("api_key", "")

    @property
    def llm_cloud_base_url(self) -> str:
        cloud_cfg = self.data["llm"].get("cloud", {})
        if isinstance(cloud_cfg, dict):
            nested_url = cloud_cfg.get("base_url")
            if nested_url:
                return nested_url
        return self.data["llm"].get("base_url", "")

    @property
    def llm_cheap_model_name(self) -> str:
        cloud_cfg = self.data["llm"].get("cloud", {})
        if isinstance(cloud_cfg, dict):
            return cloud_cfg.get("cheap_model_name", "google/gemini-2.5-flash")
        return "google/gemini-2.5-flash"

    @property
    def llm_enable_mtp(self) -> bool:
        llm_cfg = self.data.get("llm", {})
        if not isinstance(llm_cfg, dict):
            return True
        if "enable_mtp" in llm_cfg:
            return bool(llm_cfg["enable_mtp"])
        if "mtp" in llm_cfg:
            return bool(llm_cfg["mtp"])
        return True

    @property
    def llm_auto_disable_mtp_if_missing_files(self) -> bool:
        return bool(self.data["llm"].get("auto_disable_mtp_if_missing_files", True))

    @property
    def llm_mtp_file_found(self) -> bool:
        model_path = self.llm_model_path
        if not model_path:
            return False
        import os
        try:
            if os.path.exists(os.path.join(model_path, "mtp.safetensors")) or os.path.exists(os.path.join(model_path, "mtp.safetensor")):
                return True
            
            if os.path.isdir(model_path):
                for f in os.listdir(model_path):
                    if f.startswith("mtp") and (f.endswith(".safetensors") or f.endswith(".safetensor")):
                        return True
            return False
        except Exception:
            return False

    @property
    def llm_effective_mtp_mode(self) -> bool:
        requested = self.llm_enable_mtp
        if not requested:
            return False

        provider = self.llm_provider
        is_local = provider in ("mlx", "gguf")
        
        if is_local or self.llm_auto_disable_mtp_if_missing_files:
            is_local_server = False
            base_url = self.llm_cloud_base_url
            if base_url:
                from urllib.parse import urlparse
                try:
                    parsed = urlparse(base_url)
                    hostname = parsed.hostname
                    if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
                        is_local_server = True
                except Exception:
                    pass
            
            model_path = self.llm_model_path
            path_exists = False
            if model_path:
                try:
                    import os
                    path_exists = os.path.exists(model_path)
                except Exception:
                    pass

            if is_local or is_local_server or path_exists:
                found = self.llm_mtp_file_found
                if not found:
                    return False
        return True

    @property
    def llm_expected_launch_command(self) -> str:
        model_path = self.llm_model_path or ""
        if self.llm_effective_mtp_mode:
            return f"optiq serve --model {model_path} --mtp"
        else:
            return f"optiq serve --model {model_path}"

    @property
    def llm_max_tokens(self) -> int:
        return self.data["llm"].get("max_tokens", 1000)

    @property
    def llm_model_max_context(self) -> int:
        val = self.data["llm"].get("model_max_context")
        if val is not None:
            return int(val)
        
        provider = self.llm_provider
        if provider in ("openai", "openai-compatible"):
            model_name = self.llm_cloud_model_name.lower()
            if "gpt-4o" in model_name:
                return 128000
            elif "gpt-4" in model_name:
                return 8192
            elif "gpt-3.5" in model_name:
                return 16385
            elif "gemini" in model_name:
                return 32768
            elif "claude" in model_name:
                return 200000
        
        return 4096

    @property
    def llm_temp(self) -> float:
        return self.data["llm"].get("temp", 0.1)

    @property
    def llm_request_delay(self) -> float:
        return float(self.data["llm"].get("request_delay", 1.0))

    @property
    def llm_retry_backoff(self) -> float:
        return float(self.data["llm"].get("retry_backoff", 2.0))

    @property
    def llm_chunk_pool_size(self) -> int:
        return int(self.data["llm"].get("chunk_pool_size", 4))

    @property
    def llm_evaluation_concurrency(self) -> int:
        eval_cfg = self.data["llm"].get("evaluation", {})
        if isinstance(eval_cfg, dict):
            return int(eval_cfg.get("concurrency", 1))
        return 1

    @property
    def llm_evaluation_rpm(self) -> int:
        eval_cfg = self.data["llm"].get("evaluation", {})
        if isinstance(eval_cfg, dict):
            return int(eval_cfg.get("rpm", 10))
        return 10

    @property
    def llm_evaluation_retries(self) -> int:
        eval_cfg = self.data["llm"].get("evaluation", {})
        if isinstance(eval_cfg, dict):
            return int(eval_cfg.get("retries", 5))
        return 5

    @property
    def max_expanded_queries(self) -> int:
        return int(self.data["llm"].get("max_expanded_queries", 3))

    @property
    def hyde_enabled(self) -> bool:
        return bool(self.data["llm"].get("hyde_enabled", False))

    @property
    def hyde_max_tokens(self) -> int:
        return int(self.data["llm"].get("hyde_max_tokens", 300))

    @property
    def hyde_count(self) -> int:
        return int(self.data["llm"].get("hyde_count", 1))

    @property
    def llm_extraction_input_limit(self) -> int:
        val = self.data["llm"].get("extraction_input_limit", 5000)
        return max(val, self.llm_max_tokens)

    @property
    def llm_extraction_output_limit(self) -> int:
        return self.data["llm"].get("extraction_output_limit", 2048)

    @property
    def llm_clustering_input_limit(self) -> int:
        val = self.data["llm"].get("clustering_input_limit", 6000)
        return max(val, self.llm_max_tokens)

    @property
    def llm_clustering_output_limit(self) -> int:
        return self.data["llm"].get("clustering_output_limit", 1500)

    @property
    def llm_synthesis_input_limit(self) -> int:
        val = self.data["llm"].get("synthesis_input_limit", 5000)
        return max(val, self.llm_max_tokens)

    @property
    def llm_synthesis_output_limit(self) -> int:
        return self.data["llm"].get("synthesis_output_limit", 1500)

    @property
    def embedding_model_name(self) -> str:
        return resolve_existing_model_path(self.data["embedding"]["model_name"])

    @property
    def reranker_model_name(self) -> str:
        return resolve_existing_model_path(self.data.get("reranker", {}).get("model_name", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))

    @property
    def chunk_size(self) -> int:
        return self.data["embedding"]["chunk_size"]

    @property
    def chunk_overlap(self) -> int:
        return self.data["embedding"]["chunk_overlap"]

    @property
    def child_chunk_size(self) -> int:
        return self.data["embedding"].get("child_chunk_size", 300)

    @property
    def child_chunk_overlap(self) -> int:
        return self.data["embedding"].get("child_chunk_overlap", 50)

    @property
    def parent_chunk_size(self) -> int:
        return self.data["embedding"].get("parent_chunk_size", 2500)

    @property
    def parent_chunk_overlap(self) -> int:
        return self.data["embedding"].get("parent_chunk_overlap", 200)

    @property
    def score_blend_reranker_weight(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("rag", {}).get("score_blend_reranker_weight", 0.7))

    @property
    def score_blend_rrf_weight(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("rag", {}).get("score_blend_rrf_weight", 0.3))

    @property
    def rrf_k(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("rag", {}).get("rrf_k", 60.0))

    @property
    def dynamic_alpha_threshold_low(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("rag", {}).get("dynamic_alpha_threshold_low", 1.0))

    @property
    def dynamic_alpha_val_low(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("rag", {}).get("dynamic_alpha_val_low", 0.2))

    @property
    def dynamic_alpha_threshold_mid(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("rag", {}).get("dynamic_alpha_threshold_mid", 3.0))

    @property
    def dynamic_alpha_val_mid(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("rag", {}).get("dynamic_alpha_val_mid", 0.5))

    @property
    def dynamic_alpha_val_high(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("rag", {}).get("dynamic_alpha_val_high", 1.0))

    @property
    def graph_p_base(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("graph", {}).get("p_base", 0.75))

    @property
    def graph_gamma(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("graph", {}).get("gamma", 0.5))

    @property
    def graph_crawl_stop_threshold(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("graph", {}).get("crawl_stop_threshold", 1.0))

    @property
    def graph_semantic_score_threshold(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("graph", {}).get("semantic_score_threshold", 0.4))

    @property
    def graph_semantic_score_top_p(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("graph", {}).get("semantic_score_top_p", 0.9))

    @property
    def graph_sigmoid_score_threshold(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("graph", {}).get("sigmoid_score_threshold", 0.4))

    @property
    def graph_sigmoid_score_top_p(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("graph", {}).get("sigmoid_score_top_p", 0.9))

    @property
    def graph_essential_fact_threshold(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("graph", {}).get("essential_fact_threshold", 0.5))

    @property
    def graph_sigmoid_slope(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("graph", {}).get("sigmoid_slope", -25.0))

    @property
    def graph_sigmoid_center(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("graph", {}).get("sigmoid_center", 0.5))

    @property
    def graph_weight_authored(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("graph", {}).get("weight_authored", 0.8))

    @property
    def graph_weight_cites(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("graph", {}).get("weight_cites", 0.7))

    @property
    def graph_weight_mentions_concept(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("graph", {}).get("weight_mentions_concept", 0.6))

    @property
    def graph_weight_default(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("graph", {}).get("weight_default", 0.5))

    @property
    def b6_graph_neighbors_order(self) -> int:
        return int(self.data.get("hyperparameters", {}).get("graph", {}).get("b6_graph_neighbors_order", 2))

    @property
    def bm25_k1(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("bm25", {}).get("k1", 1.5))

    @property
    def bm25_b(self) -> float:
        return float(self.data.get("hyperparameters", {}).get("bm25", {}).get("b", 0.75))


    @property
    def spacy_model_name(self) -> str:
        return self.data.get("spacy", {}).get("model_name", "en_core_web_sm")

    @property
    def ner_model_name(self) -> str:
        return self.data.get("ner", {}).get("model_name", "dslim/bert-base-NER")

    @property
    def pdf_compression_enabled(self) -> bool:
        return self.data.get("pdf_compression", {}).get("enabled", True)

    @property
    def pdf_compression_dpi_threshold(self) -> int:
        return self.data.get("pdf_compression", {}).get("dpi_threshold", 151)

    @property
    def pdf_compression_dpi_target(self) -> int:
        return self.data.get("pdf_compression", {}).get("dpi_target", 150)

    @property
    def pdf_compression_quality(self) -> int:
        return self.data.get("pdf_compression", {}).get("quality", 75)

    @property
    def graph_retrieval_enabled(self) -> bool:
        env_val = os.environ.get("RAG_GRAPH_RETRIEVAL")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.data.get("graph_retrieval", {}).get("enabled", False))

    @property
    def graph_retrieval_candidate_budget_mode(self) -> str:
        mode = self.data.get("graph_retrieval", {}).get("candidate_budget_mode", "mirror_base")
        if mode not in ("mirror_base", "fixed"):
            import logging
            logging.getLogger(__name__).warning(f"Invalid candidate_budget_mode '{mode}' in config. Falling back to 'mirror_base'.")
            return "mirror_base"
        return str(mode)

    @property
    def graph_retrieval_max_graph_chunk_candidates(self) -> Optional[int]:
        val = self.data.get("graph_retrieval", {}).get("max_graph_chunk_candidates", None)
        if val is None or val == "":
            return None
        return int(val)

    @property
    def graph_concept_retrieval_enabled(self) -> bool:
        return self.is_component_enabled("graph_concept_retrieval") or self.data.get("graph_retrieval", {}).get("concept_retrieval_enabled", False)

    @property
    def graph_bridge_retrieval_enabled(self) -> bool:
        return self.is_component_enabled("graph_bridge_retrieval") or self.data.get("graph_retrieval", {}).get("bridge_retrieval_enabled", False)

    @property
    def graph_selected_sources_card_enabled(self) -> bool:
        return self.is_component_enabled("graph_selected_sources_card") or self.data.get("graph_retrieval", {}).get("selected_sources_card_enabled", False)

    @property
    def graph_retrieval_trace_enabled(self) -> bool:
        return self.is_component_enabled("graph_retrieval_trace") or self.data.get("graph_retrieval", {}).get("trace_enabled", False)

    @property
    def graph_retrieval_chunks_per_graph_paper(self) -> int:
        return int(self.data.get("graph_retrieval", {}).get("chunks_per_graph_paper", 1))

    @property
    def graph_retrieval_max_graph_candidate_papers(self):
        val = self.data.get("graph_retrieval", {}).get("max_graph_candidate_papers", "auto")
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)
        return "auto"

    @property
    def graph_retrieval_use_concept_idf(self) -> bool:
        return bool(self.data.get("graph_retrieval", {}).get("use_concept_idf", True))

    @property
    def graph_retrieval_single_token_concept_policy(self) -> str:
        return str(self.data.get("graph_retrieval", {}).get("single_token_concept_policy", "idf_if_no_multitoken"))

    @property
    def graph_retrieval_use_author_edges_for_retrieval(self) -> bool:
        return bool(self.data.get("graph_retrieval", {}).get("use_author_edges_for_retrieval", False))

    @property
    def graph_retrieval_allowed_retrieval_edge_types(self) -> List[str]:
        return list(self.data.get("graph_retrieval", {}).get("allowed_retrieval_edge_types", ["CITES", "CITED_BY", "MENTIONS_CONCEPT", "RELATED_TO", "HAS_TAG"]))

    def is_component_enabled(self, name: str) -> bool:
        if name == "intent_classifier":
            return False
        # Check environment variable first (e.g. RAG_HYDE=false)
        env_val = os.environ.get(f"RAG_{name.upper()}")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        # Fallback to config file
        components = self.data.get("rag_components", {})
        return bool(components.get(name, DEFAULT_CONFIG["rag_components"].get(name, True)))

    @property
    def rag_components(self) -> dict:
        defaults = DEFAULT_CONFIG.get("rag_components", {})
        return {name: self.is_component_enabled(name) for name in defaults}

    @property
    def taxonomy(self) -> dict:
        """
        Lazy-loads and caches config/taxonomy.yaml from the project root.
        Returns a dict with keys: concepts, topics, descriptions.
        Falls back to empty dicts if the file is missing or malformed.
        """
        if hasattr(self, "_taxonomy"):
            return self._taxonomy  # type: ignore[attr-defined]

        # Project root is two levels above this file (src/config.py → src → project root)
        project_root = Path(__file__).parent.parent
        taxonomy_path = project_root / "config" / "taxonomy.yaml"

        try:
            with open(taxonomy_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            self._taxonomy = {  # type: ignore[attr-defined]
                "concepts": raw.get("concepts", {}),
                "topics": raw.get("topics", {}),
                "descriptions": raw.get("descriptions", {}),
            }
        except Exception:
            self._taxonomy = {"concepts": {}, "topics": {}, "descriptions": {}}  # type: ignore[attr-defined]

        return self._taxonomy

    def get_storage_stats(self) -> dict:
        import os
        from src.repository.sqlite_impl import SQLiteGraphRepository
        
        storage_dir = Path(self.data_dir)
        archive_dir = Path(self.archive_dir)
        
        total_size = 0
        extension_sizes = {}
        extension_counts = {}
        
        archive_sizes_by_source = {}
        archive_counts_by_source = {}
        
        # Get mapping of paper ID to source_type from SQLiteGraphRepository
        paper_source_types = {}
        db_path = self.db_path
        if os.path.exists(db_path):
            try:
                repo = SQLiteGraphRepository(db_path)
                paper_source_types = repo.get_paper_source_types()
            except Exception:
                pass
                
        if storage_dir.exists():
            for root, _, files in os.walk(storage_dir):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.is_symlink() or not file_path.is_file():
                        continue
                    sz = file_path.stat().st_size
                    total_size += sz
                    
                    ext = file_path.suffix.lower()
                    if not ext:
                        ext = "(no extension)"
                    
                    extension_sizes[ext] = extension_sizes.get(ext, 0) + sz
                    extension_counts[ext] = extension_counts.get(ext, 0) + 1
                    
                    # Check if inside archive
                    try:
                        is_in_archive = archive_dir.resolve() in file_path.resolve().parents
                    except Exception:
                        is_in_archive = str(file_path.resolve()).startswith(str(archive_dir.resolve()))
                        
                    if is_in_archive:
                        paper_id = file_path.stem
                        stype = paper_source_types.get(paper_id)
                        if not stype:
                            if ext == ".pdf":
                                stype = "paper"
                            elif ext == ".md":
                                stype = "note"
                            elif ext == ".epub":
                                stype = "book"
                            else:
                                stype = "other"
                        archive_sizes_by_source[stype] = archive_sizes_by_source.get(stype, 0) + sz
                        archive_counts_by_source[stype] = archive_counts_by_source.get(stype, 0) + 1
                        
        # Format results
        ext_list = [
            {"extension": k, "size": v, "count": extension_counts[k]}
            for k, v in sorted(extension_sizes.items(), key=lambda x: x[1], reverse=True)
        ]
        
        src_list = [
            {"source": k, "size": v, "count": archive_counts_by_source[k]}
            for k, v in sorted(archive_sizes_by_source.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return {
            "storage_dir": str(storage_dir),
            "total_size": total_size,
            "extensions": ext_list,
            "sources": src_list
        }

config = Config()
