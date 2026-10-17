#!/usr/bin/env python3
"""
Science Graph — End-to-End RAG Quality Benchmarking Pipeline.
Orchestrates the individual stages via subprocesses for optimal VRAM isolation.
"""

import sys
import argparse
import hashlib
import time
from datetime import datetime
from pathlib import Path

# Set up python path to resolve src and core imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
from src.config import config
from src import console as con

from core.config import (
    build_custom_config,
    add_custom_config_arguments,
    resolve_project_db_path,
    database_path_cli_args,
    database_file_sha256,
)


from core.subprocess_runner import run_command_with_progress

from core.pipelined import run_pipelined_stage_async

def main():
    parser = argparse.ArgumentParser(description="End-to-End RAG Benchmarking Pipeline")
    parser.add_argument(
        "--dataset", "-d", type=str, default=None,
        help="Path to golden dataset YAML file."
    )
    parser.add_argument(
        "--pipeline", "--baselines", "-b", dest="baselines", type=str, default="all",
        help="Comma-separated baselines to run (e.g. B0,B2,B6 or CUSTOM,B4,B6) or 'all'."
    )
    parser.add_argument(
        "--cloud", action="store_true",
        help="Use cloud LLM engine for generation instead of local one."
    )
    parser.add_argument(
        "--concurrency", "-c", type=int, default=config.llm_evaluation_concurrency,
        help=f"Max concurrent API calls to the cloud provider during evaluation. Defaults to config ({config.llm_evaluation_concurrency})."
    )
    parser.add_argument(
        "--rpm", "-r", type=int, default=config.llm_evaluation_rpm,
        help=f"Rate limit in requests per minute (RPM) during evaluation. Defaults to config ({config.llm_evaluation_rpm})."
    )
    parser.add_argument(
        "--retries", type=int, default=config.llm_evaluation_retries,
        help=f"Max number of API retries on error during evaluation. Defaults to config ({config.llm_evaluation_retries})."
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None,
        help="Limit the number of questions to evaluate (for testing)."
    )
    parser.add_argument(
        "--unanswerable-limit", "-u", type=int, default=None,
        help="Limit the number of unanswerable questions (is_answerable: false) to include from the dataset."
    )
    parser.add_argument(
        "--clear-checkpoint", action="store_true",
        help="Ignore existing evaluation checkpoints and restart from scratch."
    )
    parser.add_argument(
        "--output-dir", type=str, default="graphs",
        help="Directory to save all run outputs."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Specific run directory to use, e.g., graphs/run_XYZ. If specified, overrides --output-dir."
    )
    parser.add_argument(
        "--no-unique-dir", action="store_true",
        help="Save results directly into output-dir without creating a unique timestamped subdirectory."
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip the LLM-as-a-Judge evaluation stage and metrics parsing."
    )
    parser.add_argument(
        "--pipelined", action="store_true",
        help="Run generation and evaluation concurrently in a pipelined fashion, writing results immediately."
    )
    parser.add_argument(
        "--in-process", "-i", action="store_true",
        help="Run retrieval, generation, and evaluation in-process within a single Python runtime to avoid subprocess startup overhead and model reload cycles."
    )
    parser.add_argument(
        "--logit-save", action="store_true",
        help="Save raw logits / tokens_info to raw_logits.yaml in the run directory."
    )
    add_custom_config_arguments(parser)
    args = parser.parse_args()

    active_db = resolve_project_db_path(args.db_path, args.db_sha256)
    active_db_sha256 = database_file_sha256(active_db)
    con.info(f"Using database path: {active_db} (sha256={active_db_sha256})")


    # Determine paths relative to this script
    script_dir = Path(__file__).resolve().parent
    
    # 1. Resolve dataset
    dataset_path = args.dataset
    if not dataset_path:
        dataset_path = script_dir / "golden_dataset.yaml"
        if not dataset_path.exists():
            dataset_path = script_dir / "golden_dataset.example.yaml"
    
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        con.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    is_already_retrieved = False
    if dataset_path.name in ["retrieved_contexts.yaml", "custom_retrieved_contexts.yaml"]:
        is_already_retrieved = True
    else:
        # fallback check: read a small part
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                head_lines = [f.readline() for _ in range(50)]
                head_text = "".join(head_lines)
                if "baselines:" in head_text:
                    is_already_retrieved = True
        except Exception:
            pass

    # 2. Determine target output directory
    project_root = script_dir.parents[2]
    
    if args.output:
        run_dir = Path(args.output)
        if not run_dir.is_absolute():
            run_dir = (project_root / run_dir).resolve()
    elif is_already_retrieved:
        run_dir = dataset_path.parent
        con.info("Detected pre-retrieved context dataset. Skipping retrieval stage.")
        con.info(f"Outputs will be saved directly to: {run_dir}")
        try:
            con.blank()
            from core.retrieval import evaluate_and_compare
            evaluate_and_compare(dataset_path)
        except Exception as e:
            con.warning(f"Could not generate retrieval metrics table: {e}")
    elif args.no_unique_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = (project_root / output_dir).resolve()
        run_dir = output_dir
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = (project_root / output_dir).resolve()
            
        if args.cloud:
            cloud_val = getattr(config, "llm_cloud_rag_model_name", None)
            llm_model = cloud_val if isinstance(cloud_val, str) else config.data["llm"]["cloud"]["model_name"]
        else:
            local_val = getattr(config, "llm_local_rag_model_path", None)
            llm_model = local_val if isinstance(local_val, str) else config.data["llm"]["local"]["model_path"]
            
        from core.config import create_graph_run_dir
        run_dir = create_graph_run_dir(output_dir, model_name=llm_model)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    (run_dir / "parsed").mkdir(parents=True, exist_ok=True)
    con.info(f"All run reports will be saved to: {run_dir}")

    # Build final custom configuration overrides if user requested it or specified a config file or overrides
    temp_config_file = None
    # Check if any component or hyperparameter override is set (or --custom / --config-file)
    has_overrides = args.custom or args.config_file or any(getattr(args, field, None) is not None for field in [
        "intent_classifier", "graph_ontology_lookup", "llm_query_expansion", 
        "hyde", "lexical_search", "dense_search", "dynamic_alpha_blending", 
        "rrf", "graph_expansion", "reranker", "score_blending", 
        "context_trimming", "citation_repair", "score_blend_reranker_weight",
        "score_blend_rrf_weight", "rrf_k", "dynamic_alpha_threshold_low",
        "dynamic_alpha_val_low", "dynamic_alpha_threshold_mid",
        "dynamic_alpha_val_mid", "dynamic_alpha_val_high"
    ])
    if has_overrides:
        file_config = None
        if args.config_file:
            config_path = Path(args.config_file)
            if not config_path.exists():
                con.error(f"Config file not found: {config_path}")
                sys.exit(1)
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f)
        
        custom_comp, custom_hype = build_custom_config(args, file_config)
        temp_config_file = run_dir / "temp_custom_config.yaml"
        with open(temp_config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump({
                "rag_components": custom_comp,
                "hyperparameters": custom_hype
            }, f)
        con.info(f"Generated pipeline custom configuration overrides at: {temp_config_file}")

    # Load dataset to determine total questions and total runs
    test_cases = []
    try:
        from core.config import load_benchmark_dataset
        test_cases = load_benchmark_dataset(
            dataset_path,
            limit=args.limit,
            unanswerable_limit=args.unanswerable_limit
        )
        num_cases = len(test_cases)
    except Exception as e:
        num_cases = max(getattr(args, "limit", None) or 1, 1)
        con.warning(
            f"Could not load dataset for progress totals ({e}); "
            f"using num_cases={num_cases}."
        )

    # Replicate baselines list
    raw_baselines = getattr(args, "pipeline", None) or getattr(args, "baselines", "all")
    if raw_baselines.lower() == "all":
        from core.config import STANDARD_BASELINES
        baselines_to_run = list(STANDARD_BASELINES)
        if has_overrides:
            baselines_to_run.append("CUSTOM")
        if is_already_retrieved:
            present_baselines = {"B0"}
            try:
                for case in test_cases:
                    for b_name in case.get("baselines", {}).keys():
                        present_baselines.add(b_name.upper())
                baselines_to_run = [b for b in baselines_to_run if b in present_baselines]
            except Exception:
                pass
    else:
        baselines_to_run = [b.strip().upper() for b in raw_baselines.split(",") if b.strip()]
        if has_overrides and "CUSTOM" not in baselines_to_run:
            baselines_to_run.append("CUSTOM")

    num_baselines = len(baselines_to_run)
    baselines_with_retrieval = [b for b in baselines_to_run if b != "B0"]
    num_baselines_with_retrieval = len(baselines_with_retrieval)

    total_retrieval_steps = max(num_cases * num_baselines_with_retrieval, 1)
    total_generation_steps = max(num_cases * num_baselines, 1)
    total_evaluation_steps = max(num_cases * num_baselines, 1)

    # Save config snapshot and manifest
    import os
    import yaml
    import hashlib
    
    # 1. Config snapshot
    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config.data, f, default_flow_style=False, allow_unicode=True)
    con.info(f"Saved configuration snapshot to {run_dir}/config_snapshot.yaml")
    
    # 2. git info for manifest
    git_commit = None
    git_branch = None
    working_tree_dirty = None
    try:
        import subprocess
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        status_out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        working_tree_dirty = bool(status_out.strip())
        # Sanitize mock objects during unit testing
        if not isinstance(git_commit, str) or "Mock" in type(git_commit).__name__:
            git_commit = "unknown"
        if not isinstance(git_branch, str) or "Mock" in type(git_branch).__name__:
            git_branch = "unknown"
        if not isinstance(working_tree_dirty, bool) or "Mock" in type(working_tree_dirty).__name__:
            working_tree_dirty = False
    except Exception:
        pass
        
    # dataset hash
    dataset_hash = None
    try:
        if dataset_path.exists():
            dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    except Exception:
        pass
        
    cloud_val = getattr(config, "llm_cloud_rag_model_name", None)
    local_val = getattr(config, "llm_local_rag_model_path", None)
    
    manifest = {
        "run_id": run_dir.name,
        "created_at": datetime.now().isoformat(),
        "git_commit": git_commit,
        "git_branch": git_branch,
        "working_tree_dirty": working_tree_dirty,
        "baselines": baselines_to_run,
        "model": {
            "provider": config.data["llm"]["provider"],
            "local_model_path": local_val if isinstance(local_val, str) else config.data["llm"]["local"]["model_path"],
            "model_max_context": config.llm_model_max_context,
            "max_tokens": config.data["llm"].get("max_tokens")
        },
        "embedding": {
            "model_name": config.data["embedding"]["model_name"]
        },
        "reranker": {
            "model_name": config.reranker_model_name if config.data["rag_components"].get("reranker", True) else "disabled"
        },
        "config_profile": getattr(config, "profile", None),
        "config_snapshot_path": "config_snapshot.yaml",
        "dataset": {
            "name": dataset_path.name,
            "query_count": num_cases,
            "dataset_hash": dataset_hash
        },
        "database": {
            "path": str(active_db.resolve()),
            "sha256": active_db_sha256,
        },
        "output_dir": str(run_dir),
        "trace_dir": str(run_dir / "traces"),
        "env_overrides": {
            "RAG_GRAPH_RETRIEVAL": os.environ.get("RAG_GRAPH_RETRIEVAL")
        }
    }
    
    with open(run_dir / "run_manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, default_flow_style=False, allow_unicode=True)
    con.info(f"Saved run manifest to {run_dir}/run_manifest.yaml")

    # Output filenames
    eval_results = run_dir / "evaluation_results.yaml"
    metrics_results = run_dir / "result_metrics.yaml"
    summary_md = run_dir / "metrics_summary.md"
    summary_csv = run_dir / "metrics_summary.csv"
    details_csv = run_dir / "metrics_details.csv"
    
    # Subprocesses python interpreter
    python_bin = sys.executable

    retrieved_contexts_file = dataset_path if is_already_retrieved else run_dir / "retrieved_contexts.yaml"

    # STEP 1a: Pre-Retrieval Stage
    if not is_already_retrieved:
        con.blank()
        con.section("STEP 1a · Pre-Retrieval")
        con.info(
            f"Planned units: {total_retrieval_steps} "
            f"({num_cases} cases × {num_baselines_with_retrieval} baselines excl. B0)"
        )
        if args.in_process:
            import copy
            t0_ret = time.perf_counter()
            from core.retrieval import run_staged_retrieval, evaluate_and_compare
            from src.services.container import container
            from src.prompts import prompts
            ret_args = copy.deepcopy(args)
            ret_args.output = str(retrieved_contexts_file)
            ret_args.baselines = ",".join(baselines_to_run)
            try:
                run_staged_retrieval(ret_args, config, prompts, container, con)
                elapsed_ret = time.perf_counter() - t0_ret
                con.success(f"Pre-Retrieval completed in-process in {elapsed_ret:.2f} seconds.")
                try:
                    con.blank()
                    evaluate_and_compare(retrieved_contexts_file)
                except Exception as e:
                    con.warning(f"Could not generate retrieval metrics table: {e}")
            except Exception as e:
                con.error(f"Pre-Retrieval in-process failed: {e}")
                if temp_config_file and temp_config_file.exists():
                    try:
                        temp_config_file.unlink()
                    except Exception:
                        pass
                sys.exit(1)
        else:
            retrieve_cmd = [
                python_bin, str(script_dir / "run_custom_retrieve.py"),
                "--dataset", str(dataset_path),
                "--output", str(retrieved_contexts_file),
                "--baselines", args.baselines,
                "--no-unique-dir"
            ]
            retrieve_cmd.extend(["--limit", str(num_cases)])
            retrieve_cmd.extend(database_path_cli_args(active_db, active_db_sha256))
            if args.unanswerable_limit is not None:
                retrieve_cmd.extend(["--unanswerable-limit", str(args.unanswerable_limit)])
            if args.cloud:
                retrieve_cmd.append("--cloud")
            if temp_config_file:
                retrieve_cmd.extend(["--config-file", str(temp_config_file)])

            con.dim(f"Running command: {' '.join(retrieve_cmd)}")
            try:
                elapsed_ret = run_command_with_progress(
                    retrieve_cmd,
                    f"Pre-Retrieval · {total_retrieval_steps} units (case×baseline excl. B0)",
                    total_retrieval_steps,
                    "retrieval",
                )
                con.success(f"Pre-Retrieval completed in {elapsed_ret:.2f} seconds.")
                try:
                    con.blank()
                    from run_custom_retrieve import evaluate_and_compare
                    evaluate_and_compare(retrieved_contexts_file)
                except Exception as e:
                    con.warning(f"Could not generate retrieval metrics table: {e}")
            except subprocess.CalledProcessError as e:
                con.error(f"Pre-Retrieval failed with exit code {e.returncode}.")
                if temp_config_file and temp_config_file.exists():
                    try:
                        temp_config_file.unlink()
                    except Exception:
                        pass
                sys.exit(e.returncode)

    if args.pipelined:
        import asyncio
        con.blank()
        con.section("Pipelined · Generation + Judge Eval")
        con.info(
            f"Planned units: {total_generation_steps} "
            f"({num_cases} cases × {num_baselines} baselines)"
        )
        
        # Apply custom config overrides to this process if we have them
        if has_overrides:
            from config_creator import patch_config_for_custom
            patch_config_for_custom(custom_comp, custom_hype)
            
        t0 = time.perf_counter()
        try:
            asyncio.run(
                run_pipelined_stage_async(
                    args,
                    config,
                    run_dir,
                    dataset_path,
                    baselines_to_run,
                    eval_results,
                    metrics_results,
                    retrieved_contexts_file,
                    total_generation_steps
                )
            )
            con.success(f"Pipelined execution completed in {time.perf_counter() - t0:.2f} seconds.")
        except Exception as e:
            con.error(f"Pipelined run failed: {e}")
            if temp_config_file and temp_config_file.exists():
                try:
                    temp_config_file.unlink()
                except Exception:
                    pass
            sys.exit(1)
            
        if not args.skip_eval:
            # STEP 3: Quality Metrics Parsing & Exporting CSVs
            con.blank()
            con.section("STEP 3 · Metrics Parsing")
            parse_cmd = [
                python_bin, str(script_dir / "parse_metrics.py"),
                str(run_dir)
            ]
            con.dim(f"Running command: {' '.join(parse_cmd)}")
            try:
                subprocess.run(parse_cmd, check=True)
                con.success("Metrics parsing and CSV exports completed successfully.")
            except subprocess.CalledProcessError as e:
                con.error(f"Metrics parsing/reporting failed with exit code {e.returncode}.")
                sys.exit(e.returncode)
        else:
            con.blank()
            con.info("=== Skipping LLM-as-a-Judge Evaluation & Metrics Parsing (Pipeline Optimized) ===")
    else:
        # STEP 1b: RAG Generation Stage (consuming pre-retrieved contexts)
        con.blank()
        con.section("STEP 1b · RAG Generation")
        con.info(
            f"Planned units: {total_generation_steps} "
            f"({num_cases} cases × {num_baselines} baselines)"
        )
        if args.in_process:
            import copy
            t0_gen = time.perf_counter()
            from core.generation import run_benchmarking
            from src.services.container import container
            from src.prompts import prompts
            gen_args = copy.deepcopy(args)
            gen_args.output = str(eval_results)
            gen_args.baselines = ",".join(baselines_to_run)
            gen_args.consume_contexts = str(retrieved_contexts_file)
            gen_args.no_unique_dir = True
            try:
                run_benchmarking(gen_args, config, prompts, container, con)
                elapsed_gen = time.perf_counter() - t0_gen
                con.success(f"RAG Generation completed in-process in {elapsed_gen:.2f} seconds.")
            except Exception as e:
                con.error(f"RAG Generation in-process failed: {e}")
                if temp_config_file and temp_config_file.exists():
                    try:
                        temp_config_file.unlink()
                    except Exception:
                        pass
                sys.exit(1)
        else:
            gen_cmd = [
                python_bin, str(script_dir / "run_benchmarks.py"),
                "--dataset", str(dataset_path),
                "--output", str(eval_results),
                "--baselines", ",".join(baselines_to_run),
                "--consume-contexts", str(retrieved_contexts_file),
                "--no-unique-dir"
            ]
            gen_cmd.extend(["--limit", str(num_cases)])
            gen_cmd.extend(database_path_cli_args(active_db, active_db_sha256))
            if args.unanswerable_limit is not None:
                gen_cmd.extend(["--unanswerable-limit", str(args.unanswerable_limit)])
            if args.cloud:
                gen_cmd.append("--cloud")
            if args.logit_save:
                gen_cmd.append("--logit-save")
            if temp_config_file:
                gen_cmd.extend(["--config-file", str(temp_config_file)])

            con.dim(f"Running command: {' '.join(gen_cmd)}")
            try:
                elapsed_gen = run_command_with_progress(
                    gen_cmd,
                    f"Generation · {total_generation_steps} units (case×baseline)",
                    total_generation_steps,
                    "generation",
                )
                con.success(f"RAG Generation completed in {elapsed_gen:.2f} seconds.")
            except subprocess.CalledProcessError as e:
                con.error(f"RAG Generation failed with exit code {e.returncode}.")
                if temp_config_file and temp_config_file.exists():
                    try:
                        temp_config_file.unlink()
                    except Exception:
                        pass
                sys.exit(e.returncode)

        if not args.skip_eval:
            # STEP 2: LLM-as-a-Judge Evaluation
            con.blank()
            con.section("STEP 2 · LLM-as-a-Judge Evaluation")
            con.info(
                f"Planned units: {total_evaluation_steps} "
                f"({num_cases} cases × {num_baselines} baselines)"
            )
            eval_cmd = [
                python_bin, str(script_dir / "run_evaluator.py"),
                "--input", str(eval_results),
                "--output", str(metrics_results),
                "--baselines", ",".join(baselines_to_run),
                "--concurrency", str(args.concurrency),
                "--rpm", str(args.rpm),
                "--retries", str(args.retries)
            ]
            eval_cmd.extend(["--limit", str(num_cases)])
            if args.unanswerable_limit is not None:
                eval_cmd.extend(["--unanswerable-limit", str(args.unanswerable_limit)])
            if args.clear_checkpoint:
                eval_cmd.append("--clear-checkpoint")

            con.dim(f"Running command: {' '.join(eval_cmd)}")
            try:
                elapsed_eval = run_command_with_progress(
                    eval_cmd,
                    f"Judge Eval · {total_evaluation_steps} units (case×baseline)",
                    total_evaluation_steps,
                    "evaluation",
                )
                con.success(f"LLM-as-a-Judge Evaluation completed in {elapsed_eval:.2f} seconds.")
            except subprocess.CalledProcessError as e:
                con.error(f"LLM-as-a-Judge Evaluation failed with exit code {e.returncode}.")
                sys.exit(e.returncode)

            # STEP 3: Quality Metrics Parsing & Exporting CSVs
            con.blank()
            con.section("STEP 3 · Metrics Parsing")
            parse_cmd = [
                python_bin, str(script_dir / "parse_metrics.py"),
                str(run_dir)
            ]
            con.dim(f"Running command: {' '.join(parse_cmd)}")
            try:
                subprocess.run(parse_cmd, check=True)
                con.success("Metrics parsing and CSV exports completed successfully.")
            except subprocess.CalledProcessError as e:
                con.error(f"Metrics parsing/reporting failed with exit code {e.returncode}.")
                sys.exit(e.returncode)
        else:
            con.blank()
            con.info("=== Skipping LLM-as-a-Judge Evaluation & Metrics Parsing (Pipeline Optimized) ===")

    # Clean up temporary configuration file if created
    if temp_config_file and temp_config_file.exists():
        try:
            temp_config_file.unlink()
        except Exception:
            pass

    con.blank()
    con.success("=== PIPELINE RUN COMPLETE ===")
    if not args.skip_eval:
        con.info(f"Markdown Summary: {summary_md}")
        con.info(f"Wide CSV Summary (Typst): {summary_csv}")
        con.info(f"Detailed CSV (Pandas): {details_csv}")
    else:
        con.info(f"Evaluation results yaml saved to: {eval_results}")


if __name__ == "__main__":
    main()
