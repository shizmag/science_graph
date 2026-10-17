#!/usr/bin/env python3
"""
Science Graph — Custom RAG Staged Retrieval Runner.
Allows running the retrieval pipeline with a custom configuration of components
and hyperparameters, and compares results against baselines.
"""

import sys
import argparse
import copy
import yaml
from pathlib import Path

# Set up python path to resolve src and core imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.services.container import container
from src.config import config
from src import console as con
from src.prompts import prompts
from core.config import (
    DEFAULT_COMPONENTS,
    DEFAULT_HYPERPARAMS,
    add_custom_config_arguments,
    build_custom_config,
    patch_config_for_custom as patch_retrieval_for_custom,
    resolve_project_db_path,
    database_file_sha256,
)
from core.retrieval import (
    run_staged_retrieval,
    evaluate_and_compare,
    save_markdown_report,
)



def main():
    parser = argparse.ArgumentParser(description="Run RAG retrieval with custom configurations and analyze their impact")
    
    # Standard options
    parser.add_argument(
        "--dataset", "-d", type=str, default=None,
        help="Path to golden dataset YAML file."
    )
    parser.add_argument(
        "--output", "-o", type=str, default="reports/custom_retrieved_contexts.yaml",
        help="Path to save retrieved contexts YAML."
    )
    parser.add_argument(
        "--cloud", action="store_true",
        help="Use cloud LLM engine for retrieval/generation instead of local one."
    )
    parser.add_argument(
        "--pipeline", "--baselines", dest="baselines", type=str, default="B4,B6",
        help="Comma-separated baselines to compare against (e.g. B4,B6) or 'all'."
    )
    parser.add_argument(
        "--no-unique-dir", action="store_true",
        help="Save output directly to the specified path without timestamp subdirectory."
    )
    parser.add_argument(
        "--artifact-dir", type=str, default=None,
        help="Optional path to write a user-facing markdown report artifact."
    )
    add_custom_config_arguments(parser)
    parser.add_argument(
        "--limit", "-l", type=int, default=-1,
        help="Limit the number of questions to evaluate (default: -1 which means no limit/all)."
    )
    parser.add_argument(
        "--unanswerable-limit", "-u", type=int, default=None,
        help="Limit the number of unanswerable questions (is_answerable: false) to include from the dataset."
    )

    args = parser.parse_args()

    active_db = resolve_project_db_path(args.db_path, args.db_sha256)
    con.info(
        f"Using database path: {active_db} "
        f"(sha256={database_file_sha256(active_db)})"
    )

    # Load file config if specified
    file_config = None
    if args.config_file:
        config_path = Path(args.config_file)
        if not config_path.exists():
            con.error(f"Config file not found: {config_path}")
            sys.exit(1)
        with open(config_path, "r", encoding="utf-8") as f:
            file_config = yaml.safe_load(f)

    # Build final custom configuration overrides
    custom_comp, custom_hype = build_custom_config(args, file_config)

    # Apply dynamic patch
    patch_retrieval_for_custom(custom_comp, custom_hype)

    # Determine baselines to run
    raw_baselines = getattr(args, "pipeline", None) or getattr(args, "baselines", "B4,B6")
    if raw_baselines.lower() == "all":
        from core.config import STANDARD_BASELINES
        req_baselines = list(STANDARD_BASELINES)
    else:
        req_baselines = [b.strip().upper() for b in raw_baselines.split(",") if b.strip()]

    has_custom_overrides = args.custom or file_config or any(getattr(args, field, None) is not None for field in [
        "intent_classifier", "graph_ontology_lookup", "llm_query_expansion", 
        "hyde", "lexical_search", "dense_search", "dynamic_alpha_blending", 
        "rrf", "graph_expansion", "reranker", "score_blending", 
        "context_trimming", "citation_repair"
    ])
    if has_custom_overrides and "CUSTOM" not in req_baselines:
        req_baselines.append("CUSTOM")

    args.baselines = ",".join(req_baselines)

    con.info("Custom retrieval config constructed.")
    con.info(f"Enabled components: {sorted([k for k, v in custom_comp.items() if v])}")
    con.blank()

    # Run retrieval
    try:
        config.data["rag_components"] = copy.deepcopy(custom_comp)
        run_staged_retrieval(args, config, prompts, container, con)
    except Exception as e:
        con.error(f"Failed to execute retrieval: {e}")
        # Restore configuration before exit
        config.data["hyperparameters"] = copy.deepcopy(DEFAULT_HYPERPARAMS)
        config.data["rag_components"] = copy.deepcopy(DEFAULT_COMPONENTS)
        sys.exit(1)

    # Restore original config hyperparameters and components
    config.data["hyperparameters"] = copy.deepcopy(DEFAULT_HYPERPARAMS)
    config.data["rag_components"] = copy.deepcopy(DEFAULT_COMPONENTS)

    # Evaluate metrics and display comparison
    output_path = Path(args.output)
    if not output_path.exists():
        # Find path if timestamp subdirectory was added
        candidates = list(output_path.parent.glob(f"**/{output_path.name}"))
        if candidates:
            # Sort by modification time to get the newest
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            output_path = candidates[0]
        else:
            con.error(f"Could not locate output file at {output_path}")
            sys.exit(1)

    metrics_summary = evaluate_and_compare(output_path)

    # Save reports
    local_report = Path("reports/retrieval_custom_report.md")
    local_report.parent.mkdir(parents=True, exist_ok=True)
    save_markdown_report(metrics_summary, custom_comp, custom_hype, local_report)
    con.success(f"Detailed Markdown report saved to: {local_report.resolve()}")

    if args.artifact_dir:
        art_report = Path(args.artifact_dir) / "retrieval_comparison_report.md"
        save_markdown_report(metrics_summary, custom_comp, custom_hype, art_report)
        con.success(f"Artifact Markdown report saved to: {art_report.resolve()}")


if __name__ == "__main__":
    main()
