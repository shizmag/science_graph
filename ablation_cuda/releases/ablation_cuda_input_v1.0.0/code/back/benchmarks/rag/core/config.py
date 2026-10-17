"""
Science Graph — Configuration Management, Presets, and Dataset Loading.
Contains baseline definitions, configuration presets, argument parsers, dataset loaders,
and run directory creation utilities.
"""

import copy
import argparse
import hashlib
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import yaml

from src.config import config

# Map of baselines and their descriptions
BASELINES_INFO: Dict[str, str] = {
    "B0": "Zero-Shot (Pure generation) — evaluating basic LLM knowledge without context.",
    "B1": "Pure Lexical (Lexical only) — keyword search strictly via SQLite FTS5.",
    "B2": "Pure Dense (Vectors only) — classical semantic search over embeddings.",
    "B3": "Dense + HyDE (Vectors + Hypothetical Document) — semantic search with hypothetical answer.",
    "B4": "Standard Hybrid + Reranker (Basic hybrid with reranker) — FTS5 + Vectors combination via RRF + reranker without graphs.",
    "B5": "Hybrid + Graph + Reranker (Basic Graph-RAG with reranker) — hybrid search + static graph traversal + reranker.",
    "B6": "Full Pipeline (Full run) — all 12 components enabled (graph, reranker, LLM query expansion, etc., without HyDE).",
    "CUSTOM": "Custom Run (Custom parameters configuration) — for testing impact of settings on search."
}

# Standard baselines excluding CUSTOM (used for default 'all' runs)
STANDARD_BASELINES: List[str] = ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]

DEFAULT_LIMIT = 50


def database_file_sha256(database_path: Path | str) -> str:
    """Hash a resolved SQLite database file without opening it through the repository layer."""
    target_db = Path(database_path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with target_db.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

# Store default config settings for reference and rollback
DEFAULT_COMPONENTS = copy.deepcopy(config.data.get("rag_components", {}))
DEFAULT_HYPERPARAMS = copy.deepcopy(config.data.get("hyperparameters", {}))

# Hardcoded CUSTOM preset configuration (components and hyperparameters).
CUSTOM_PRESET_COMPONENTS = {
    "intent_classifier": False,
    "graph_ontology_lookup": True,
    "llm_query_expansion": False,
    "hyde": False,
    "lexical_search": True,
    "dense_search": True,
    "dynamic_alpha_blending": False,
    "rrf": True,
    "graph_expansion": False,
    "reranker": True,
    "score_blending": False,
    "context_trimming": True,
    "citation_repair": True,
    "deduplicate_parent_chunks": True,
}


class RAGPreset(NamedTuple):
    score_blend_reranker_weight: float
    score_blend_rrf_weight: float
    rrf_k: float
    dynamic_alpha_threshold_low: float
    dynamic_alpha_val_low: float
    dynamic_alpha_threshold_mid: float
    dynamic_alpha_val_mid: float
    dynamic_alpha_val_high: float


class GraphPreset(NamedTuple):
    p_base: float
    gamma: float
    crawl_stop_threshold: float
    semantic_score_threshold: float
    semantic_score_top_p: float
    sigmoid_score_threshold: float
    sigmoid_score_top_p: float
    essential_fact_threshold: float
    sigmoid_slope: float
    sigmoid_center: float
    weight_authored: float
    weight_cites: float
    weight_mentions_concept: float
    weight_default: float


class BM25Preset(NamedTuple):
    k1: float
    b: float


class CustomPresetHyperparams(NamedTuple):
    rag: RAGPreset
    graph: GraphPreset
    bm25: BM25Preset


CUSTOM_PRESET_HYPERPARAMS_NT = CustomPresetHyperparams(
    rag=RAGPreset(
        score_blend_reranker_weight=0.75,
        score_blend_rrf_weight=0.25,
        rrf_k=60.0,
        dynamic_alpha_threshold_low=1.2,
        dynamic_alpha_val_low=1.0,
        dynamic_alpha_threshold_mid=3.0,
        dynamic_alpha_val_mid=0.5,
        dynamic_alpha_val_high=1.0,
    ),
    graph=GraphPreset(
        p_base=0.0,
        gamma=0.0,
        crawl_stop_threshold=1.0,
        semantic_score_threshold=0.35,
        semantic_score_top_p=0.9,
        sigmoid_score_threshold=0.4,
        sigmoid_score_top_p=0.9,
        essential_fact_threshold=0.5,
        sigmoid_slope=0.0,
        sigmoid_center=0.5,
        weight_authored=1.0,
        weight_cites=1.0,
        weight_mentions_concept=1.0,
        weight_default=1.0,
    ),
    bm25=BM25Preset(
        k1=1.5,
        b=0.75,
    )
)

CUSTOM_PRESET_HYPERPARAMS = {
    "rag": CUSTOM_PRESET_HYPERPARAMS_NT.rag._asdict(),
    "graph": CUSTOM_PRESET_HYPERPARAMS_NT.graph._asdict(),
    "bm25": CUSTOM_PRESET_HYPERPARAMS_NT.bm25._asdict(),
}


def get_custom_preset_weights(preset_hype: CustomPresetHyperparams) -> dict:
    """Returns the custom preset weights configured for edge-type heuristics from the provided NamedTuple."""
    return {
        "weight_authored": preset_hype.graph.weight_authored,
        "weight_cites": preset_hype.graph.weight_cites,
        "weight_mentions_concept": preset_hype.graph.weight_mentions_concept,
        "weight_default": preset_hype.graph.weight_default,
    }


def add_custom_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds custom config command line arguments to the parser."""
    parser.add_argument(
        "--config-file", type=str, default=None,
        help="Path to a custom YAML configuration file containing overrides."
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Path to SQLite database file (defaults to project science_graph.db)."
    )
    parser.add_argument(
        "--db-sha256", type=str, default=None,
        help="Expected SHA-256 for --db-path; fail if the database identity differs."
    )
    parser.add_argument(
        "--custom", action="store_true",
        help="Apply the hardcoded custom preset components and hyperparameters."
    )

    # Component overrides (Boolean Optional actions)
    parser.add_argument("--intent-classifier", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--graph-ontology-lookup", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--llm-query-expansion", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hyde", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lexical-search", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dense-search", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dynamic-alpha-blending", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rrf", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--graph-expansion", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reranker", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--score-blending", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--context-trimming", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--citation-repair", action=argparse.BooleanOptionalAction, default=None)

    # RAG Hyperparameters overrides
    parser.add_argument("--score-blend-reranker-weight", type=float, default=None)
    parser.add_argument("--score-blend-rrf-weight", type=float, default=None)
    parser.add_argument("--rrf-k", type=float, default=None)
    parser.add_argument("--dynamic-alpha-threshold-low", type=float, default=None)
    parser.add_argument("--dynamic-alpha-val-low", type=float, default=None)
    parser.add_argument("--dynamic-alpha-threshold-mid", type=float, default=None)
    parser.add_argument("--dynamic-alpha-val-mid", type=float, default=None)
    parser.add_argument("--dynamic-alpha-val-high", type=float, default=None)

    # Graph Hyperparameters overrides
    parser.add_argument("--graph-p-base", type=float, default=None)
    parser.add_argument("--graph-gamma", type=float, default=None)
    parser.add_argument("--graph-crawl-stop-threshold", type=float, default=None)
    parser.add_argument("--graph-semantic-score-threshold", type=float, default=None)
    parser.add_argument("--graph-semantic-score-top-p", type=float, default=None)
    parser.add_argument("--graph-sigmoid-score-threshold", type=float, default=None)
    parser.add_argument("--graph-sigmoid-score-top-p", type=float, default=None)
    parser.add_argument("--graph-essential-fact-threshold", type=float, default=None)
    parser.add_argument("--graph-sigmoid-slope", type=float, default=None)
    parser.add_argument("--graph-sigmoid-center", type=float, default=None)
    parser.add_argument("--graph-weight-authored", type=float, default=None)
    parser.add_argument("--graph-weight-cites", type=float, default=None)
    parser.add_argument("--graph-weight-mentions-concept", type=float, default=None)
    parser.add_argument("--graph-weight-default", type=float, default=None)

    # BM25 Hyperparameters overrides
    parser.add_argument("--bm25-k1", type=float, default=None)
    parser.add_argument("--bm25-b", type=float, default=None)


def build_custom_config(args: Any, file_config: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Builds custom components and hyperparameters dictionaries by merging
    defaults, hardcoded preset (if args.custom), file config (if provided), and CLI overrides.
    """
    custom_comp = copy.deepcopy(DEFAULT_COMPONENTS)
    custom_hype = copy.deepcopy(DEFAULT_HYPERPARAMS)

    if getattr(args, "custom", False):
        custom_comp.update(CUSTOM_PRESET_COMPONENTS)
        for section, params in CUSTOM_PRESET_HYPERPARAMS.items():
            if section not in custom_hype:
                custom_hype[section] = {}
            custom_hype[section].update(params)

    if file_config:
        if "rag_components" in file_config:
            custom_comp.update(file_config["rag_components"])
        if "hyperparameters" in file_config:
            for section, params in file_config["hyperparameters"].items():
                if section not in custom_hype:
                    custom_hype[section] = {}
                if params:
                    custom_hype[section].update(params)

    comp_fields = [
        "intent_classifier", "graph_ontology_lookup", "llm_query_expansion", 
        "hyde", "lexical_search", "dense_search", "dynamic_alpha_blending", 
        "rrf", "graph_expansion", "reranker", "score_blending", 
        "context_trimming", "citation_repair"
    ]
    for field in comp_fields:
        val = getattr(args, field, None)
        if val is not None:
            custom_comp[field] = val

    rag_hype_fields = [
        "score_blend_reranker_weight", "score_blend_rrf_weight", "rrf_k",
        "dynamic_alpha_threshold_low", "dynamic_alpha_val_low",
        "dynamic_alpha_threshold_mid", "dynamic_alpha_val_mid",
        "dynamic_alpha_val_high"
    ]
    for field in rag_hype_fields:
        val = getattr(args, field, None)
        if val is not None:
            if "rag" not in custom_hype:
                custom_hype["rag"] = {}
            custom_hype["rag"][field] = val

    graph_hype_fields = [
        ("graph_p_base", "p_base"),
        ("graph_gamma", "gamma"),
        ("graph_crawl_stop_threshold", "crawl_stop_threshold"),
        ("graph_semantic_score_threshold", "semantic_score_threshold"),
        ("graph_semantic_score_top_p", "semantic_score_top_p"),
        ("graph_sigmoid_score_threshold", "sigmoid_score_threshold"),
        ("graph_sigmoid_score_top_p", "sigmoid_score_top_p"),
        ("graph_essential_fact_threshold", "essential_fact_threshold"),
        ("graph_sigmoid_slope", "sigmoid_slope"),
        ("graph_sigmoid_center", "sigmoid_center"),
        ("graph_weight_authored", "weight_authored"),
        ("graph_weight_cites", "weight_cites"),
        ("graph_weight_mentions_concept", "weight_mentions_concept"),
        ("graph_weight_default", "weight_default")
    ]
    for arg_name, conf_name in graph_hype_fields:
        val = getattr(args, arg_name, None)
        if val is not None:
            if "graph" not in custom_hype:
                custom_hype["graph"] = {}
            custom_hype["graph"][conf_name] = val

    bm25_hype_fields = [
        ("bm25_k1", "k1"),
        ("bm25_b", "b")
    ]
    for arg_name, conf_name in bm25_hype_fields:
        val = getattr(args, arg_name, None)
        if val is not None:
            if "bm25" not in custom_hype:
                custom_hype["bm25"] = {}
            custom_hype["bm25"][conf_name] = val

    return custom_comp, custom_hype


_BASELINE_CONFIG_PATCH_STATE: Dict[str, Any] | None = None


def restore_baseline_config_patch() -> None:
    """Restores get_baseline_config after patch_config_for_custom."""
    global _BASELINE_CONFIG_PATCH_STATE
    if not _BASELINE_CONFIG_PATCH_STATE:
        return

    orig = _BASELINE_CONFIG_PATCH_STATE["core.config"]
    # Reassign local module reference
    globals()["get_baseline_config"] = orig

    core_retrieval = _BASELINE_CONFIG_PATCH_STATE.get("core.retrieval")
    if core_retrieval is not None:
        import core.retrieval as core_retrieval_mod
        core_retrieval_mod.get_baseline_config = core_retrieval

    core_generation = _BASELINE_CONFIG_PATCH_STATE.get("core.generation")
    if core_generation is not None:
        import core.generation as core_generation_mod
        core_generation_mod.get_baseline_config = core_generation

    _BASELINE_CONFIG_PATCH_STATE = None


def patch_config_for_custom(custom_comp: dict, custom_hype: dict):
    """Dynamically patches core config and retrieval functions to support CUSTOM baseline."""
    restore_baseline_config_patch()

    orig_get_baseline_config = globals()["get_baseline_config"]

    def custom_get_baseline_config(baseline: str, config_rag_components: dict) -> dict:
        if baseline == "CUSTOM":
            config.data["hyperparameters"] = copy.deepcopy(custom_hype)
            merged = orig_get_baseline_config("CUSTOM", config_rag_components)
            merged.update(copy.deepcopy(custom_comp))
            return merged

        config.data["hyperparameters"] = copy.deepcopy(DEFAULT_HYPERPARAMS)
        return orig_get_baseline_config(baseline, config_rag_components)

    global _BASELINE_CONFIG_PATCH_STATE
    _BASELINE_CONFIG_PATCH_STATE = {"core.config": orig_get_baseline_config}

    globals()["get_baseline_config"] = custom_get_baseline_config

    try:
        import core.retrieval as core_retrieval
        _BASELINE_CONFIG_PATCH_STATE["core.retrieval"] = core_retrieval.get_baseline_config
        core_retrieval.get_baseline_config = custom_get_baseline_config
    except ImportError:
        pass

    try:
        import core.generation as core_generation
        _BASELINE_CONFIG_PATCH_STATE["core.generation"] = core_generation.get_baseline_config
        core_generation.get_baseline_config = custom_get_baseline_config
    except ImportError:
        pass


def get_baseline_config(baseline: str, config_rag_components: dict) -> Dict[str, bool]:
    """Returns the RAG components configuration dictionary for a given baseline.
    
    Accepts config_rag_components keys to initialize the default state.
    """
    components = {k: False for k in config_rag_components.keys()}
    
    if baseline == "B0":
        pass
    elif baseline == "B1":
        components["lexical_search"] = True
    elif baseline == "B2":
        components["dense_search"] = True
    elif baseline == "B3":
        components["dense_search"] = True
        components["hyde"] = True
    elif baseline == "B4":
        components["dense_search"] = True
        components["lexical_search"] = True
        components["rrf"] = True
        components["dynamic_alpha_blending"] = True
        components["reranker"] = True
    elif baseline == "B5":
        components["dense_search"] = True
        components["lexical_search"] = True
        components["rrf"] = True
        components["dynamic_alpha_blending"] = True
        components["graph_expansion"] = True
        components["context_trimming"] = True
        components["citation_repair"] = True
        components["reranker"] = True
    elif baseline == "B6":
        components = {k: config_rag_components.get(k, True) for k in config_rag_components.keys()}
        components["hyde"] = False
        components["reranker"] = True
        components["graph_neighbors_in_rrf"] = True
    elif baseline == "CUSTOM":
        components = {k: config_rag_components.get(k, True) for k in config_rag_components.keys()}
        components["hyde"] = False
        components["citation_repair"] = bool(config_rag_components.get("citation_repair", True))

    if baseline not in ["B6", "CUSTOM"]:
        components["intent_classifier"] = False
    else:
        components["intent_classifier"] = config_rag_components.get("intent_classifier", False)
        
    if baseline != "B6":
        components["graph_neighbors_in_rrf"] = False

    if "shannon_estimator_enabled" in config_rag_components:
        components["shannon_estimator_enabled"] = bool(config_rag_components["shannon_estimator_enabled"])
    if baseline != "B0":
        components["deduplicate_parent_chunks"] = bool(config_rag_components.get("deduplicate_parent_chunks", True))
    return components


def get_safe_model_name(model_name: str) -> str:
    """Safely normalizes LLM/embedder model path/name to be used as directory name."""
    name = Path(model_name).name
    name = name.replace("/", "_").replace(":", "_").replace(" ", "_")
    return name


def load_benchmark_dataset(
    dataset_path: Path,
    limit: int = None,
    seed: int = 42,
    unanswerable_limit: int = None,
) -> list:
    """Loads a dataset YAML file, formats it if it's SciQ format,
    and applies random sampling based on limit and unanswerable_limit count.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        return []

    is_sciq = False
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict) and "question" in data[0]:
            is_sciq = True

    if is_sciq:
        converted = []
        for idx, item in enumerate(data):
            q_data = item["question"]
            q_id = q_data.get("id", idx + 1)
            converted.append({
                "id": f"sciq_{q_id}",
                "category": "sciq",
                "query": q_data["q"],
                "golden_answer": q_data["a"],
                "is_answerable": True,
                "c": q_data["c"]
            })

        unique_contexts = []
        context_to_id = {}
        for c in [case["c"] for case in converted]:
            if c not in context_to_id:
                unique_contexts.append(c)
                context_to_id[c] = len(unique_contexts)

        for case in converted:
            c_id = context_to_id[case["c"]]
            case["expected_papers"] = [f"sciq_paper_{c_id}"]
            del case["c"]

        data = converted

        if limit is None:
            limit = DEFAULT_LIMIT
        elif limit == -1:
            limit = None

    rng = random.Random(seed)

    if unanswerable_limit is not None and isinstance(unanswerable_limit, int):
        from core.metrics import get_is_answerable
        ans_cases = [item for item in data if get_is_answerable(item)]
        unans_cases = [item for item in data if not get_is_answerable(item)]

        if limit is not None and isinstance(limit, int) and limit != -1 and len(ans_cases) > limit:
            ans_cases = rng.sample(ans_cases, limit)

        if unanswerable_limit != -1 and len(unans_cases) > unanswerable_limit:
            unans_cases = rng.sample(unans_cases, unanswerable_limit)

        data = ans_cases + unans_cases
        try:
            data.sort(key=lambda x: str(x.get("id", "")))
        except Exception:
            pass
    elif limit is not None and isinstance(limit, int) and limit != -1 and len(data) > limit:
        data = rng.sample(data, limit)
        try:
            data.sort(key=lambda x: str(x.get("id", "")))
        except Exception:
            pass

    return data


def create_graph_run_dir(
    base_dir: Path,
    run_name: Optional[str] = None,
    model_name: Optional[str] = None,
    timestamp: Any = None,
) -> Path:
    """Creates a unified benchmark run directory inside base_dir with traces/ and parsed/ subfolders."""
    if timestamp is None:
        timestamp = datetime.now()

    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")

    if run_name:
        dir_name = run_name
    elif model_name:
        safe_model = get_safe_model_name(model_name)
        dir_name = f"run_{ts_str}_{safe_model}"
    else:
        dir_name = f"run_{ts_str}"

    run_dir = base_dir / dir_name

    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    (run_dir / "parsed").mkdir(parents=True, exist_ok=True)

    return run_dir


def resolve_project_db_path(
    db_path_arg: Optional[str] = None,
    expected_sha256: Optional[str] = None,
) -> Path:
    """Resolves and configures the SQLite database path in global config.
    Checks explicit CLI argument, local project science_graph.db, or defaults.
    """
    from src.config import config
    from src.services.container import container

    file_path = Path(__file__).resolve()
    target_db = None

    if db_path_arg:
        target_db = Path(db_path_arg).expanduser().resolve()
        if not target_db.is_file():
            raise FileNotFoundError(
                f"Explicit SQLite database path does not name an existing file: {target_db}"
            )
    else:
        back_dir = file_path.parents[3]
        repo_dir = file_path.parents[4] if len(file_path.parents) > 4 else back_dir
        candidates = [
            back_dir / "science_graph.db",
            repo_dir / "back" / "science_graph.db",
            repo_dir / "science_graph.db",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.stat().st_size > 1000000:
                target_db = candidate
                break
        if not target_db:
            for candidate in candidates:
                if candidate.exists():
                    target_db = candidate
                    break

    if target_db and target_db.is_file():
        if expected_sha256:
            actual_sha256 = database_file_sha256(target_db)
            if actual_sha256.lower() != expected_sha256.lower():
                raise ValueError(
                    "Explicit SQLite database SHA-256 does not match the run identity: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
        config.data["db_path"] = str(target_db)
        container._graph_repo = None
        container._vector_repo = None
    elif expected_sha256:
        raise FileNotFoundError("A database SHA-256 was provided without a resolvable database file")

    return Path(config.db_path)


def database_path_cli_args(
    database_path: Path | str,
    expected_sha256: Optional[str] = None,
) -> list[str]:
    """Return an absolute database path and optional immutable identity for child processes."""
    target_db = Path(database_path).expanduser().resolve(strict=True)
    if not target_db.is_file():
        raise FileNotFoundError(f"SQLite database path is not a regular file: {target_db}")
    args = ["--db-path", str(target_db)]
    if expected_sha256:
        args.extend(["--db-sha256", expected_sha256])
    return args

