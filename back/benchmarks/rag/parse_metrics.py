#!/usr/bin/env python3
"""
Science Graph — RAG Quality Metrics & Trace Parser and Aggregator.
Parses result_metrics.yaml and trace files inside the run directory.
"""

import sys
import argparse
import csv
import json
import statistics
import math
from typing import Literal

import yaml
from pathlib import Path

# Increase CSV field size limit to handle large context blocks safely
max_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_limit)
        break
    except OverflowError:
        max_limit = int(max_limit / 2)


# Set up python path to resolve src and core imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.analytics import analyze_metrics, GRAPH_ENABLED_BASELINES
from core.reporting import (
    print_rich_tables,
    generate_markdown_report,
    export_wide_csv,
    export_detailed_csv
)
from core.models import load_report_file
from core.statistics import StatsConfig
from metrics_stats_connector import export_stats_json, run_statistical_pipeline


from core.traces import (
    load_graph_retrieval_trace,
    parse_graph_retrieval_trace,
    parse_eval_trace,
    parse_all_traces,
)



def format_val(val: float | None, is_pct: bool = False, digits: int = 1) -> str:
    """Format float values consistently using standard rounding."""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "N/A"
    if is_pct:
        return f"{round(val * 100, digits):.{digits}f}%"
    return f"{round(val, 4):.4f}"


def print_confusion_matrix_and_metrics_tables(data):
    """
    Computes and prints confusion matrix and classification quality metrics
    for all baselines based on model refusals and semantic accuracy scores.
    """
    if not data or "results" not in data or not data["results"]:
        print("Warning: No results data available to compute confusion matrix.")
        return

    import re
    import math

    results = data["results"]
    baselines = sorted(list(results[0].get("baselines", {}).keys()))

    def is_refusal(text):
        text_lower = text.lower()
        refusal_keywords = [
            "информация отсутствует", "информации отсутствует", "отсутствует информация", "отсутствуют сведения", 
            "не содержит информации", "не содержит сведений", "не содержат информации", "не содержится информации",
            "не содержится сведений", "нет сведений", "нет информации", "нет данных", "нет описания", "нет упоминания",
            "невозможно ответить", "невозможно определить", "не представляется возможным",
            "отказываемся от", "не может быть описано", "не приводятся", "не упоминаются", "не упоминается",
            "не приводится", "не описывается", "не сообщается", "не указано", "не указана", "не указаны",
            "в предоставленном тексте нет", "в предоставленном контексте нет", "в предоставленных материалах нет",
            "в статье нет", "в тексте не содержится", "в контексте не содержится", "в материалах не содержится",
            "не удается найти", "не удалось найти", "не удается определить", "не удалось определить",
            "нельзя сделать вывод", "нельзя ответить", "не представляется возможным ответить",
            "no information", "does not contain", "cannot answer", "not mention", "do not have",
            "not available", "unable to answer", "insufficient information", "not specify", "not specified",
            "not described", "not defined", "not found", "unanswerable"
        ]
        for kw in refusal_keywords:
            if kw in text_lower:
                return True
                
        if re.search(r"не\s+(?:упоминает|упомяну|приводит|описывает|содержит|находится|обнаружено|указано)", text_lower):
            if any(w in text_lower for w in ["текст", "контекст", "источник", "стать", "материал", "информац", "сведен", "данн"]):
                return True
        if "отсутств" in text_lower:
            if any(w in text_lower for w in ["текст", "контекст", "источник", "стать", "материал", "информац", "сведен", "данн"]):
                return True
        if "нет" in text_lower:
            if re.search(r"\bнет\s+(?:упоминания|описания|сведений|информации|данных|деталей|подробностей|сведений|указания)\b", text_lower):
                return True
        if "не упоминается" in text_lower and any(w in text_lower for w in ["предоставлен", "контекст", "текст", "стать"]):
            return True
            
        return False

    classification = {}
    for b in baselines:
        classification[b] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        
    for r in results:
        is_ans = r.get("is_answerable")
        if is_ans is None:
            is_ans = True
        else:
            is_ans = str(is_ans).lower() == "true"
            
        for b in baselines:
            b_data = r.get("baselines", {}).get(b, {})
            gen_ans = b_data.get("generated_answer", "")
            
            # Check for outcome/predicted_abstained first
            outcome = b_data.get("answerability_outcome")
            if not outcome:
                # check eval_metrics too
                eval_metrics = b_data.get("eval_metrics", {})
                outcome = eval_metrics.get("answerability_outcome") if isinstance(eval_metrics, dict) else None
                
            if outcome in ("TP", "FP", "TN", "FN"):
                classification[b][outcome] += 1
            else:
                pred_abst = b_data.get("predicted_abstained")
                if pred_abst is None and isinstance(b_data.get("eval_metrics"), dict):
                    pred_abst = b_data.get("eval_metrics", {}).get("predicted_abstained")
                
                if pred_abst is None:
                    try:
                        from core.metrics import detect_abstention
                        pred_abst = detect_abstention(gen_ans)
                    except Exception:
                        pred_abst = is_refusal(gen_ans)
                
                if is_ans:
                    if pred_abst:
                        classification[b]["FN"] += 1
                    else:
                        classification[b]["TP"] += 1
                else:
                    if pred_abst:
                        classification[b]["TN"] += 1
                    else:
                        classification[b]["FP"] += 1

    # Print Confusion Matrix Table
    print("\nConfusion Matrix:")
    cm_header = "| {:<10} | {:<18} | {:<19} | {:<18} | {:<18} |"
    cm_row = "| {:<10} | {:<18d} | {:<19d} | {:<18d} | {:<18d} |"
    cm_sep = "+" + "-"*12 + "+" + "-"*20 + "+" + "-"*21 + "+" + "-"*20 + "+" + "-"*20 + "+"
    print(cm_sep)
    print(cm_header.format("Baseline", "True Positive (TP)", "False Positive (FP)", "True Negative (TN)", "False Negative (FN)"))
    print(cm_sep)
    for b in baselines:
        stats_b = classification[b]
        print(cm_row.format(b, stats_b["TP"], stats_b["FP"], stats_b["TN"], stats_b["FN"]))
    print(cm_sep)

    # Print Classification Metrics Table
    print("\nClassification Quality Metrics:")
    metrics_header = "| {:<10} | {:<8} | {:<9} | {:<8} | {:<8} | {:<11} | {:<8} | {:<8} | {:<18} | {:<11} | {:<16} |"
    metrics_row = "| {:<10} | {:<8} | {:<9} | {:<8} | {:<8} | {:<11} | {:<8} | {:<8} | {:<18} | {:<11} | {:<16} |"
    metrics_sep = "+" + "-"*12 + "+" + "-"*10 + "+" + "-"*11 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*20 + "+" + "-"*13 + "+" + "-"*18 + "+"
    print(metrics_sep)
    print(metrics_header.format(
        "Baseline", "Accuracy", "Precision", "Recall", "F1 Score", 
        "Specificity", "FPR", "FNR", "Hallucination Rate", "Ans Rate", "Abstention Rate"
    ))
    print(metrics_sep)

    total_q = len(results)
    for b in baselines:
        stats_b = classification[b]
        tp = stats_b["TP"]
        fp = stats_b["FP"]
        tn = stats_b["TN"]
        fn = stats_b["FN"]
        
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        hallucination_rate = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
        answer_rate = (tp + fp) / total_q
        abstention_rate = (tn + fn) / total_q

        print(metrics_row.format(
            b,
            format_val(accuracy, True),
            format_val(precision, True),
            format_val(recall, True),
            format_val(f1, False),
            format_val(specificity, True),
            format_val(fpr, True),
            format_val(fnr, True),
            format_val(hallucination_rate, True),
            format_val(answer_rate, True),
            format_val(abstention_rate, True)
        ))
    print(metrics_sep)


class MetricsParser:
    """Configurable parser for RAG benchmark metrics, traces, and statistical analysis."""

    def __init__(
        self,
        run_dir: Path,
        *,
        input_path: Path | None = None,
        output_md_path: Path | None = None,
        csv_summary_path: Path | None = None,
        csv_details_path: Path | None = None,
        traces_only: bool = False,
        confusion: bool = False,
        enable_stats: bool = True,
        n_bootstraps: int = 10000,
        alpha: float = 0.05,
        ci_method: Literal["percentile", "bca"] = "percentile",
        correction_method: Literal["holm", "bonferroni", "none"] = "holm",
        enable_plots: bool = False,
        random_seed: int = 42,
    ):
        self.run_dir = Path(run_dir)
        self.input_path = input_path
        self.output_md_path = output_md_path or self.run_dir / "metrics_summary.md"
        self.csv_summary_path = csv_summary_path or self.run_dir / "metrics_summary.csv"
        self.csv_details_path = csv_details_path or self.run_dir / "metrics_details.csv"
        self.traces_only = traces_only
        self.confusion = confusion
        self.stats_config = StatsConfig(
            enable_stats=enable_stats,
            n_bootstraps=n_bootstraps,
            alpha=alpha,
            ci_method=ci_method,
            correction_method=correction_method,
            enable_plots=enable_plots,
            random_seed=random_seed,
            plots_dir=str(self.run_dir / "parsed" / "stats_plots"),
        )
        self.parsed_dir = self.run_dir / "parsed"
        self.traces_dir = self.run_dir / "traces"
        self.data = None
        self.stats = None
        self.stats_analysis = None

    def run(self) -> dict:
        """Execute the full parse → aggregate → report → statistics pipeline."""
        self.parsed_dir.mkdir(parents=True, exist_ok=True)

        trace_path = self.run_dir / "graph_retrieval_trace.jsonl"
        if not trace_path.exists():
            trace_path = self.run_dir / "traces" / "graph_retrieval_trace.jsonl"
        trace_map = load_graph_retrieval_trace(trace_path)

        input_path = self._resolve_input_path()
        if not self.traces_only and input_path.exists():
            try:
                report = load_report_file(input_path)
                self.data = report.model_dump()
                self.stats = analyze_metrics(self.data, trace_map)
                print_rich_tables(self.stats)
            except Exception as e:
                raise RuntimeError(f"Error processing {input_path}: {e}") from e
        elif not self.traces_only:
            print(f"Warning: result_metrics.yaml not found at {input_path}")

        if self.stats and self.data:
            try:
                generate_markdown_report(self.stats, self.output_md_path)
                export_wide_csv(self.stats, self.csv_summary_path)
                export_detailed_csv(self.data, self.stats, self.csv_details_path)
                with (self.parsed_dir / "metrics_summary.parsed.json").open("w", encoding="utf-8") as f:
                    json.dump(self.stats, f, ensure_ascii=False, indent=2)
            except OSError as exc:
                raise RuntimeError(f"Unable to write metrics reports under {self.parsed_dir}: {exc}") from exc

        graph_rows = None
        eval_rows = None
        has_traces = (
            self.traces_dir.exists()
            or (self.run_dir / "graph_retrieval_trace.jsonl").exists()
            or (self.run_dir / "eval_trace.jsonl").exists()
        )
        if has_traces:
            graph_rows, _ = parse_graph_retrieval_trace(self.traces_dir, self.parsed_dir)
            eval_rows, _ = parse_eval_trace(self.traces_dir, self.parsed_dir)

        metrics_rows = self._collect_metrics_rows()
        self._write_metrics_details_parsed(metrics_rows)

        joined_data = self._build_joined_data(metrics_rows, graph_rows, eval_rows)
        run_summary = self._build_run_summary(joined_data)
        self._write_joined_csv(joined_data)
        self._write_run_summary(run_summary)
        self._print_console_report(run_summary, joined_data)

        if self.confusion and self.data:
            print_confusion_matrix_and_metrics_tables(self.data)

        if self.stats_config.enable_stats and self.data:
            joined_rows = list(joined_data.values())
            plots_dir = Path(self.stats_config.plots_dir) if self.stats_config.plots_dir else self.parsed_dir
            self.stats_analysis = run_statistical_pipeline(
                self.data,
                config=self.stats_config,
                joined_rows=joined_rows,
                output_dir=plots_dir,
            )
            export_stats_json(self.stats_analysis, self.parsed_dir / "statistical_analysis.json")
            if self.stats:
                generate_markdown_report(
                    self.stats,
                    self.output_md_path,
                    stats_analysis=self.stats_analysis,
                )
            if self.stats_analysis.get("markdown_sections"):
                print("\n[+] Statistical analysis appended to markdown report.")

        return {
            "data": self.data,
            "stats": self.stats,
            "stats_analysis": self.stats_analysis,
            "run_summary": run_summary,
        }

    def _resolve_input_path(self) -> Path:
        if self.input_path:
            return Path(self.input_path)
        preferred = [
            "result_metrics.yaml",
            "result_metrics_judge.yaml",
            "evaluation_results.yaml",
            "evaluation_results_judge.yaml",
        ]
        for name in preferred:
            candidate = self.run_dir / name
            if candidate.exists():
                if name == "evaluation_results.yaml":
                    alt = self.run_dir / "result_metrics.yaml"
                    if alt.exists():
                        return alt
                return candidate
        return self.run_dir / "result_metrics.yaml"

    def _collect_metrics_rows(self) -> list[dict]:
        """Collect per-query metrics rows from YAML results or metrics_details.csv."""
        metrics_rows: list[dict] = []
        if self.data and "results" in self.data:
            baselines = self.stats["baselines"] if self.stats else list(self.data["results"][0].get("baselines", {}).keys())
            for r in self.data["results"]:
                q_id = r.get("id", "UNKNOWN")
                category = r.get("category", "general")
                for b in baselines:
                    b_data = r.get("baselines", {}).get(b, {})
                    if not b_data:
                        continue
                    eval_metrics = b_data.get("eval_metrics", {})
                    is_ans = r.get("is_answerable")
                    if is_ans is None:
                        is_ans = True
                    else:
                        is_ans = str(is_ans).lower() == "true"
                    sh_diag = b_data.get("shannon_diagnostics", {}) if isinstance(b_data.get("shannon_diagnostics"), dict) else {}
                    def get_telemetry(key: str, alt_key: str | None = None):
                        v = b_data.get(key)
                        if v is None and alt_key:
                            v = b_data.get(alt_key)
                        if v is None:
                            v = sh_diag.get(key)
                        if v is None and alt_key:
                            v = sh_diag.get(alt_key)
                        if v is None and isinstance(eval_metrics, dict):
                            v = eval_metrics.get(key) or (eval_metrics.get(alt_key) if alt_key else None)
                        return v

                    metrics_rows.append({
                        "query_id": q_id,
                        "category": category,
                        "baseline": b,
                        "status": b_data.get("status", "success"),
                        "is_answerable": is_ans,
                        "predicted_abstained": eval_metrics.get("predicted_abstained", False),
                        "answerability_outcome": eval_metrics.get("answerability_outcome", "TP"),
                        "retrieval_recall": eval_metrics.get("retrieval_recall"),
                        "context_precision": eval_metrics.get("context_precision"),
                        "faithfulness": eval_metrics.get("faithfulness"),
                        "answer_relevance": eval_metrics.get("answer_relevance"),
                        "citation_fidelity": eval_metrics.get("citation_fidelity"),
                        "semantic_accuracy": eval_metrics.get("semantic_accuracy"),
                        "context_fillness": eval_metrics.get("context_fillness"),
                        "ar_sa_f1": eval_metrics.get("ar_sa_f1"),
                        "latency_sec": b_data.get("latency_sec"),
                        "msp": get_telemetry("msp", "avg_msp"),
                        "avg_msp": get_telemetry("avg_msp", "msp"),
                        "logit_margin": get_telemetry("logit_margin", "avg_logit_margin"),
                        "avg_logit_margin": get_telemetry("avg_logit_margin", "logit_margin"),
                        "first_token_margin": get_telemetry("first_token_margin"),
                        "first_token_msp": get_telemetry("first_token_msp"),
                        "citation_entropy": get_telemetry("citation_entropy", "h_citation"),
                        "ll_rag": get_telemetry("ll_rag"),
                        "ll_base": get_telemetry("ll_base"),
                        "clr": get_telemetry("clr"),
                    })
        elif self.csv_details_path.exists():
            try:
                with open(self.csv_details_path, "r", encoding="utf-8") as f:
                    metrics_rows = list(csv.DictReader(f))
            except Exception as e:
                print(f"Warning: Could not read metrics_details.csv: {e}")
        return metrics_rows

    def _write_metrics_details_parsed(self, metrics_rows: list[dict]) -> None:
        if not metrics_rows:
            return
        path = self.parsed_dir / "metrics_details.parsed.csv"
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=metrics_rows[0].keys())
                writer.writeheader()
                for r in metrics_rows:
                    writer.writerow(r)
        except OSError as exc:
            raise RuntimeError(f"Unable to write parsed metrics rows to {path}: {exc}") from exc

    def _build_joined_data(self, metrics_rows, graph_rows, eval_rows) -> dict:
        """Build per-query joined data from YAML and trace sources."""
        joined_data = {}
        for r in metrics_rows:
            q_id = str(r.get("query_id") or "")
            base = str(r.get("baseline") or "")
            if not q_id or not base:
                continue
            is_ans = r.get("is_answerable")
            if is_ans is None or is_ans == "":
                is_ans = True
            else:
                is_ans = str(is_ans).lower() == "true"
            joined_data[(q_id, base)] = {
                "query_id": q_id,
                "baseline": base,
                "category": r.get("category", "general"),
                "is_answerable": is_ans,
                "predicted_abstained": r.get("predicted_abstained", False),
                "answerability_outcome": r.get("answerability_outcome", "TP"),
                "retrieval_recall": r.get("retrieval_recall"),
                "context_precision": r.get("context_precision"),
                "faithfulness": r.get("faithfulness"),
                "answer_relevance": r.get("answer_relevance"),
                "citation_fidelity": r.get("citation_fidelity"),
                "semantic_accuracy": r.get("semantic_accuracy"),
                "context_fillness": r.get("context_fillness"),
                "ar_sa_f1": r.get("ar_sa_f1"),
                "latency_sec": r.get("latency_sec"),
                "msp": r.get("msp"),
                "avg_msp": r.get("avg_msp"),
                "logit_margin": r.get("logit_margin"),
                "avg_logit_margin": r.get("avg_logit_margin"),
                "first_token_margin": r.get("first_token_margin"),
                "first_token_msp": r.get("first_token_msp"),
                "citation_entropy": r.get("citation_entropy"),
                "ll_rag": r.get("ll_rag"),
                "ll_base": r.get("ll_base"),
                "clr": r.get("clr"),
            }

        if graph_rows:
            for r in graph_rows:
                key = (str(r.get("query_id") or ""), str(r.get("baseline") or ""))
                if not key[0] or not key[1] or key[1] not in GRAPH_ENABLED_BASELINES:
                    continue
                if key not in joined_data:
                    joined_data[key] = {
                        "query_id": key[0],
                        "baseline": key[1],
                        "category": r.get("category", "general"),
                    }
                joined_data[key].update({
                    "graph_retrieval_enabled": r.get("graph_retrieval_enabled"),
                    "graph_retrieval_skip_reason": r.get("graph_retrieval_skip_reason"),
                    "base_candidates_count": r.get("base_candidates_count"),
                    "graph_neighbor_paper_ids_count": r.get("graph_neighbor_paper_ids_count"),
                    "graph_chunk_candidates_count": r.get("graph_chunk_candidates_count"),
                    "merged_candidates_count_before_reranker": r.get("merged_candidates_count_before_reranker"),
                    "graph_chunks_survived_final_context_count": r.get("graph_chunks_survived_final_context_count"),
                    "graph_chunks_survived_final_context": r.get("graph_chunks_survived_final_context"),
                    "graph_survival_rate": r.get("graph_survival_rate"),
                    "distinct_papers_in_final_context": r.get("distinct_papers_in_final_context"),
                })

        if eval_rows:
            for r in eval_rows:
                key = (str(r.get("query_id") or ""), str(r.get("baseline") or ""))
                if not key[0] or not key[1]:
                    continue
                if key not in joined_data:
                    joined_data[key] = {
                        "query_id": key[0],
                        "baseline": key[1],
                        "category": r.get("category", "general"),
                    }
                for m in [
                    "retrieval_recall", "context_precision", "faithfulness", "answer_relevance",
                    "citation_fidelity", "semantic_accuracy", "context_fillness", "ar_sa_f1",
                    "is_answerable", "predicted_abstained", "answerability_outcome", "latency_sec",
                    "msp", "avg_msp", "logit_margin", "avg_logit_margin", "first_token_margin",
                    "first_token_msp", "citation_entropy", "ll_rag", "ll_base", "clr",
                ]:
                    if r.get(m) is not None and (joined_data[key].get(m) is None or joined_data[key].get(m) == ""):
                        joined_data[key][m] = r[m]

        return joined_data

    JOINED_HEADERS = [
        "query_id", "baseline", "category", "is_answerable", "predicted_abstained", "answerability_outcome",
        "retrieval_recall", "context_precision", "faithfulness", "answer_relevance",
        "citation_fidelity", "semantic_accuracy", "context_fillness", "ar_sa_f1", "latency_sec",
        "msp", "avg_msp", "logit_margin", "avg_logit_margin", "first_token_margin", "first_token_msp",
        "citation_entropy", "ll_rag", "ll_base", "clr",
        "graph_retrieval_enabled", "graph_retrieval_skip_reason",
        "base_candidates_count", "graph_neighbor_paper_ids_count",
        "graph_chunk_candidates_count", "merged_candidates_count_before_reranker",
        "graph_chunks_survived_final_context_count", "graph_chunks_survived_final_context",
        "graph_survival_rate", "distinct_papers_in_final_context",
    ]

    def _write_joined_csv(self, joined_data: dict) -> None:
        if not joined_data:
            return
        path = self.parsed_dir / "per_query_joined.csv"
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.JOINED_HEADERS)
                writer.writeheader()
                for key in sorted(joined_data.keys()):
                    row = joined_data[key]
                    writer.writerow({h: row.get(h, "") for h in self.JOINED_HEADERS})
        except OSError as exc:
            raise RuntimeError(f"Unable to write joined metrics to {path}: {exc}") from exc

    def _build_run_summary(self, joined_data: dict) -> dict:
        baselines = sorted({row["baseline"] for row in joined_data.values() if row.get("baseline")})
        run_summary = {
            "run_id": self.run_dir.name,
            "baselines": baselines,
            "query_count": len({row["query_id"] for row in joined_data.values()}),
            "metrics": {},
            "graph_retrieval": {},
            "by_category": {},
        }

        for b in baselines:
            b_rows = [row for row in joined_data.values() if row["baseline"] == b]
            run_summary["metrics"][b] = {}
            for m in [
                "semantic_accuracy", "faithfulness", "latency_sec", "retrieval_recall",
                "context_precision", "answer_relevance", "ar_sa_f1",
                "msp", "avg_msp", "logit_margin", "avg_logit_margin", "first_token_margin",
                "first_token_msp", "citation_entropy", "ll_rag", "ll_base", "clr",
            ]:
                vals = []
                for row in b_rows:
                    is_ans = row.get("is_answerable")
                    if is_ans is None:
                        is_ans = True
                    else:
                        is_ans = str(is_ans).lower() == "true"
                    if not is_ans:
                        continue
                    val = row.get(m)
                    if val is not None and val != "":
                        try:
                            vals.append(float(val))
                        except ValueError:
                            pass
                run_summary["metrics"][b][f"{m}_mean"] = round(statistics.mean(vals), 4) if vals else 0.0

            g_rows = [row for row in b_rows if row.get("graph_retrieval_enabled") is not None]
            if g_rows:
                queries_with_graph_neighbors = sum(
                    1 for r in g_rows if float(r.get("graph_neighbor_paper_ids_count") or 0) > 0
                )
                queries_with_graph_chunks = sum(
                    1 for r in g_rows if float(r.get("graph_chunk_candidates_count") or 0) > 0
                )
                queries_with_graph_survival = sum(
                    1 for r in g_rows if float(r.get("graph_chunks_survived_final_context_count") or 0) > 0
                )
                surv_vals = []
                for r in g_rows:
                    s_rate = r.get("graph_survival_rate")
                    if s_rate is not None and s_rate != "":
                        try:
                            surv_vals.append(float(s_rate))
                        except ValueError:
                            pass
                run_summary["graph_retrieval"][b] = {
                    "queries_with_graph_neighbors": queries_with_graph_neighbors,
                    "queries_with_graph_chunks": queries_with_graph_chunks,
                    "queries_with_graph_survival": queries_with_graph_survival,
                    "avg_graph_survival_rate": round(statistics.mean(surv_vals), 4) if surv_vals else 0.0,
                }

        categories = {row.get("category", "general") for row in joined_data.values() if row.get("category")}
        for cat in categories:
            run_summary["by_category"][cat] = {}
            cat_rows = [row for row in joined_data.values() if row.get("category") == cat]
            for b in baselines:
                b_cat_rows = [row for row in cat_rows if row["baseline"] == b]
                if not b_cat_rows:
                    continue
                run_summary["by_category"][cat][b] = {}
                sem_vals = []
                for row in b_cat_rows:
                    s_acc = row.get("semantic_accuracy")
                    if s_acc is not None and s_acc != "":
                        try:
                            sem_vals.append(float(s_acc))
                        except ValueError:
                            pass
                if sem_vals:
                    run_summary["by_category"][cat][b]["semantic_accuracy_mean"] = round(statistics.mean(sem_vals), 4)
                g_cat_rows = [row for row in b_cat_rows if row.get("graph_retrieval_enabled") is not None]
                if g_cat_rows:
                    run_summary["by_category"][cat][b]["queries_with_graph_survival"] = sum(
                        1 for r in g_cat_rows if float(r.get("graph_chunks_survived_final_context_count") or 0) > 0
                    )

        def safe_float(v):
            if v is None or v == "":
                return None
            try:
                return float(v)
            except ValueError:
                return None

        highlights = {"low_faithfulness": [], "high_latency": [], "graph_successes": []}
        for row in joined_data.values():
            faith = safe_float(row.get("faithfulness"))
            if faith is not None and faith < 0.8:
                highlights["low_faithfulness"].append({
                    "query_id": row["query_id"],
                    "baseline": row["baseline"],
                    "faithfulness": faith,
                    "category": row.get("category", "general"),
                })
            lat = safe_float(row.get("latency_sec"))
            if lat is not None and lat > 4.0:
                highlights["high_latency"].append({
                    "query_id": row["query_id"],
                    "baseline": row["baseline"],
                    "latency_sec": lat,
                    "category": row.get("category", "general"),
                })
            surv = safe_float(row.get("graph_survival_rate"))
            if surv is not None and surv >= 0.5:
                highlights["graph_successes"].append({
                    "query_id": row["query_id"],
                    "baseline": row["baseline"],
                    "survival_rate": surv,
                    "survived_count": int(safe_float(row.get("graph_chunks_survived_final_context_count")) or 0),
                    "category": row.get("category", "general"),
                })

        highlights["low_faithfulness"].sort(key=lambda x: x["faithfulness"])
        highlights["low_faithfulness"] = highlights["low_faithfulness"][:3]
        highlights["high_latency"].sort(key=lambda x: x["latency_sec"], reverse=True)
        highlights["high_latency"] = highlights["high_latency"][:3]
        highlights["graph_successes"].sort(key=lambda x: (x["survival_rate"], x["survived_count"]), reverse=True)
        highlights["graph_successes"] = highlights["graph_successes"][:3]
        run_summary["highlights"] = highlights
        return run_summary

    def _write_run_summary(self, run_summary: dict) -> None:
        json_path = self.parsed_dir / "run_summary.json"
        yaml_path = self.parsed_dir / "run_summary.yaml"
        try:
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(run_summary, f, ensure_ascii=False, indent=2)
            with yaml_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(run_summary, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        except OSError as exc:
            raise RuntimeError(f"Unable to write run summary under {self.parsed_dir}: {exc}") from exc

    def _print_console_report(self, run_summary: dict, joined_data: dict) -> None:
        print(f"\nParsed run: {self.run_dir}")
        print(f"Queries: {run_summary['query_count']}")
        print(f"Baselines: {', '.join(run_summary['baselines'])}")
        joined_csv_path = self.parsed_dir / "per_query_joined.csv"
        if joined_csv_path.exists():
            print(f"\nJoined file:\n  - {joined_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse RAG quality metrics and generate reports")
    parser.add_argument(
        "run_dir", type=str, nargs="?", default=None,
        help="Path to benchmark run directory (e.g. graphs/run_XYZ)."
    )
    parser.add_argument(
        "--file", "-f", type=str, default=None,
        help="Path to result_metrics.yaml file (backward compatibility)."
    )
    parser.add_argument(
        "--output-md", "-m", type=str, default=None,
        help="Path to save summary markdown report."
    )
    parser.add_argument(
        "--csv-summary", type=str, default=None,
        help="Path to save wide-format summary CSV."
    )
    parser.add_argument(
        "--csv-details", type=str, default=None,
        help="Path to save detailed case-by-case CSV."
    )
    parser.add_argument(
        "--traces-only", action="store_true",
        help="Analyze only trace files and skip result_metrics.yaml."
    )
    parser.add_argument(
        "--include-traces", action="store_true",
        help="Included for CLI compatibility (traces are parsed by default if present)."
    )
    parser.add_argument(
        "--confusion", "--confusion-matrix", action="store_true",
        help="Calculate and print confusion matrix and classification quality metrics."
    )
    parser.add_argument(
        "--no-stats", action="store_true",
        help="Disable statistical analysis (enabled by default)."
    )
    parser.add_argument(
        "--n-bootstraps", type=int, default=10000,
        help="Number of bootstrap resamples (default: 10000)."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level (default: 0.05)."
    )
    parser.add_argument(
        "--ci-method", choices=["percentile", "bca"], default="percentile",
        help="Bootstrap CI method: percentile or BCa (default: percentile)."
    )
    parser.add_argument(
        "--correction-method", choices=["holm", "bonferroni", "none"], default="holm",
        help="Multiple-comparison correction (default: holm)."
    )
    parser.add_argument(
        "--stats-plots", action="store_true",
        help="Generate optional statistical plots (boxplots, p-value heatmaps)."
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Random seed for reproducible bootstrap resampling (default: 42)."
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]

    def _resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path.resolve()
        for base in (Path.cwd(), script_dir, project_root):
            candidate = (base / path).resolve()
            if candidate.exists():
                return candidate
        return (script_dir / path).resolve()

    # 1. Resolve run directory
    run_dir = None
    if args.run_dir:
        run_dir = _resolve_path(args.run_dir)
        if run_dir.is_file():
            run_dir = run_dir.parent
    elif args.file:
        file_path = _resolve_path(args.file)
        run_dir = file_path if file_path.is_dir() else file_path.parent
    else:
        # Fallback default: project_root / graphs or reports
        run_dir = project_root / "graphs"
        if not run_dir.exists():
            run_dir = project_root / "reports"

    if not run_dir.exists():
        print(f"Error: run directory does not exist: {run_dir}")
        sys.exit(1)

    print(f"Processing run directory: {run_dir}")

    input_path = None
    if args.file:
        file_path = _resolve_path(args.file)
        if file_path.is_file():
            input_path = file_path
    elif args.run_dir:
        run_candidate = _resolve_path(args.run_dir)
        if run_candidate.is_file() and run_candidate.suffix in (".yaml", ".yml"):
            input_path = run_candidate

    output_md_path = Path(args.output_md) if args.output_md else run_dir / "metrics_summary.md"
    csv_summary_path = Path(args.csv_summary) if args.csv_summary else run_dir / "metrics_summary.csv"
    csv_details_path = Path(args.csv_details) if args.csv_details else run_dir / "metrics_details.csv"

    metrics_parser = MetricsParser(
        run_dir,
        input_path=input_path,
        output_md_path=output_md_path,
        csv_summary_path=csv_summary_path,
        csv_details_path=csv_details_path,
        traces_only=args.traces_only,
        confusion=args.confusion,
        enable_stats=not args.no_stats,
        n_bootstraps=args.n_bootstraps,
        alpha=args.alpha,
        ci_method=args.ci_method,
        correction_method=args.correction_method,
        enable_plots=args.stats_plots,
        random_seed=args.random_seed,
    )

    try:
        metrics_parser.run()
    except Exception as e:
        print(f"Error processing benchmark run: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
