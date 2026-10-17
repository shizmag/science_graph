#!/usr/bin/env python3
"""
Science Graph — RAG Quality Benchmarking Runner.
Runs a golden dataset against baseline configurations (delegated to core).
"""

import sys
import argparse
from pathlib import Path

# Set up python path to resolve src and core imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
from src.services.container import container
from src.config import config
from src import console as con
from src.prompts import prompts

from config_creator import (
    build_custom_config,
    patch_config_for_custom,
    restore_baseline_config_patch,
    add_custom_config_arguments
)

# Backward compatibility exports in case other files import them from here
from typing import Any, Tuple
from core.config import get_baseline_config as _get_baseline_config
from core.config import resolve_project_db_path, database_file_sha256
from core.generation import run_benchmarking
from core.generation import run_query_on_baseline as _run_query_on_baseline

def get_baseline_config(baseline: str, config_rag_components: dict = None) -> dict:
    if config_rag_components is None:
        config_rag_components = config.rag_components
    return _get_baseline_config(baseline, config_rag_components)

def run_query_on_baseline(
    rag_service: Any, 
    query: str, 
    baseline: str, 
    use_cloud: bool = False,
    cfg: Any = None
) -> Tuple[str, list, dict, list]:
    if cfg is None:
        cfg = config
    return _run_query_on_baseline(rag_service, query, baseline, use_cloud, cfg)


def main():
    parser = argparse.ArgumentParser(description="Science Graph RAG Baselines Benchmarking Runner")
    parser.add_argument(
        "--dataset", "-d", type=str, default=None,
        help="Path to golden dataset YAML file. Defaults to golden_dataset.yaml or golden_dataset.example.yaml"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="graphs/evaluation_results.yaml",
        help="Path to save evaluation output results."
    )
    parser.add_argument(
        "--cloud", action="store_true",
        help="Use cloud LLM engine instead of local one."
    )
    parser.add_argument(
        "--baselines", type=str, default="all",
        help="Comma-separated baselines to run (e.g. B0,B2,B6) or 'all'."
    )
    parser.add_argument(
        "--no-unique-dir", action="store_true",
        help="Save reports directly to the output path without creating a unique timestamped subdirectory (enables merging of results)."
    )
    parser.add_argument(
        "--consume-contexts", type=str, default=None,
        help="Path to pre-retrieved contexts YAML file to consume and bypass retrieval stages."
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None,
        help="Limit the number of questions to evaluate (for testing/SciQ default)."
    )
    parser.add_argument(
        "--unanswerable-limit", "-u", type=int, default=None,
        help="Limit the number of unanswerable questions (is_answerable: false) to include from the dataset."
    )
    parser.add_argument(
        "--logit-save", action="store_true",
        help="Save raw logits / tokens_info to raw_logits.yaml in the run directory."
    )
    add_custom_config_arguments(parser)
    args = parser.parse_args()

    active_db = resolve_project_db_path(args.db_path, args.db_sha256)
    con.info(
        f"Using database path: {active_db} "
        f"(sha256={database_file_sha256(active_db)})"
    )

    # Determine project root and resolve output
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (project_root / output_path).resolve()
    args.output = str(output_path)

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

    patch_config_for_custom(custom_comp, custom_hype)
    try:
        run_benchmarking(args, config, prompts, container, con)
    finally:
        restore_baseline_config_patch()


if __name__ == "__main__":
    main()
