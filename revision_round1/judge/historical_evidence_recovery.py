"""Recover July-run B3/B5 generator-visible evidence for paired judge review.

Evidence is reconstructed from the archived post-trim fields and the prompt
routing code at the run's recorded source commit. This does not recreate the
byte-exact full prompt and does not validate historical scores; it identifies
the exact archived evidence payload inserted into the historical prompt path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

PAPER_TO_CODE = {"B3": "B4", "B5": "B6"}
DEFAULT_RUN = Path(__file__).resolve().parents[2] / "graphs/run_20260729_211324_OCC-RAG-1.7B"
DEFAULT_OUTPUT = Path(__file__).resolve().parent
EXPECTED_SOURCE_COMMIT = "b4f3322d8cae3395bcbf0b5a1f2e47a904d1591a"
SOURCE_FILES = (
    "back/benchmarks/rag/core/config.py",
    "back/benchmarks/rag/core/retrieval.py",
    "back/benchmarks/rag/core/generation.py",
    "back/src/prompts/rag/ask_no_expander.txt",
    "back/src/prompts/rag/ask_expander.txt",
    "back/src/services/rag_service.py",
    "back/src/services/graph_expander.py",
    "back/src/prompts/manager.py",
    "back/benchmarks/rag/core/pipelined.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return yaml.load(stream, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


def _git_blob(repo: Path, revision: str, path: str) -> tuple[str, str]:
    content = subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{path}"])
    return content.decode("utf-8"), hashlib.sha256(content).hexdigest()


def verify_source_contract(repo: Path, source_commit: str) -> dict[str, Any]:
    if source_commit != EXPECTED_SOURCE_COMMIT:
        raise ValueError(f"run source commit mismatch: expected {EXPECTED_SOURCE_COMMIT}, got {source_commit}")
    config, config_hash = _git_blob(repo, source_commit, SOURCE_FILES[0])
    retrieval, retrieval_hash = _git_blob(repo, source_commit, SOURCE_FILES[1])
    generation, generation_hash = _git_blob(repo, source_commit, SOURCE_FILES[2])
    no_expander, no_expander_hash = _git_blob(repo, source_commit, SOURCE_FILES[3])
    expander_prompt, expander_prompt_hash = _git_blob(repo, source_commit, SOURCE_FILES[4])
    expander, expander_hash = _git_blob(repo, source_commit, SOURCE_FILES[6])
    prompt_manager, manager_hash = _git_blob(repo, source_commit, SOURCE_FILES[7])
    pipelined, pipelined_hash = _git_blob(repo, source_commit, SOURCE_FILES[8])

    required = {
        "B4": ('elif baseline == "B4"',),
        "B6": ('elif baseline == "B6"',),
    }
    if not all(text in config for text in required["B4"] + required["B6"]):
        raise ValueError("historical baseline configuration branches not found at run commit")
    b4_start, b5_start, b6_start = config.index('elif baseline == "B4"'), config.index('elif baseline == "B5"'), config.index('elif baseline == "B6"')
    b4_config = config[b4_start:b5_start]
    b6_config = config[b6_start:config.find('elif baseline == "CUSTOM"', b6_start)]
    if 'components = {k: False for k in config_rag_components.keys()}' not in config[:b4_start] or 'components["graph_expansion"]' in b4_config:
        raise ValueError("code B4 graph-expansion default-off semantics not verified")
    if 'components = {k: config_rag_components.get(k, True) for k in config_rag_components.keys()}' not in b6_config:
        raise ValueError("code B6 dynamic snapshot-inheritance semantics not verified")
    for literal in ('"trimmed_text": trimmed_text', '"trimmed_graph": trimmed_graph', '"enrichment_block": enrichment_block'):
        if literal not in retrieval:
            raise ValueError(f"retrieval archive writer lacks expected field {literal}")
    if 'context_text=trimmed_text, context_graph=trimmed_graph' not in generation:
        raise ValueError("generation source no longer confirms B4 post-trim evidence insertion")
    if 'enrichment_block=enrichment_block' not in generation:
        raise ValueError("generation source no longer confirms B6 enrichment insertion")
    fallback_tokens = ('ask_no_expander', 'context_text=trimmed_text', 'context_graph=trimmed_graph')
    if not all(token in generation for token in fallback_tokens) or "No essential knowledge graph enrichment found." not in generation:
        raise ValueError("B6 empty-enrichment fallback to trimmed text/graph is not verified")
    if "def get_prompt" not in prompt_manager:
        raise ValueError("historical prompt manager implementation is missing")
    if "run_query_on_baseline" not in pipelined and "generate_response" not in pipelined:
        raise ValueError("historical pipeline generation entrypoint is not verified")
    if "{{context_text}}" not in no_expander or "{{context_graph}}" not in no_expander:
        raise ValueError("B4 prompt template context placeholders are missing")
    if "{{enrichment_block}}" not in expander_prompt:
        raise ValueError("B6 prompt template enrichment placeholder is missing")
    if "return \"\\n\".join(enrichment_lines)" not in expander:
        raise ValueError("expander source does not confirm archived enrichment block serialization")
    return {
        "source_commit": source_commit,
        "paper_to_code": PAPER_TO_CODE,
        "source_sha256": {
            SOURCE_FILES[0]: config_hash, SOURCE_FILES[1]: retrieval_hash,
            SOURCE_FILES[2]: generation_hash, SOURCE_FILES[3]: no_expander_hash,
            SOURCE_FILES[4]: expander_prompt_hash, SOURCE_FILES[6]: expander_hash,
            SOURCE_FILES[7]: manager_hash, SOURCE_FILES[8]: pipelined_hash,
        },
        "verified_routes": {
            "B3/B4": "ask_no_expander inserts archived post-trim trimmed_text and trimmed_graph; config disables graph expansion.",
            "B5/B6": "ask_expander inserts archived enrichment_block when nonempty; block is the exact final expansion serialization, not the intermediate retrieved chunks or graph trace.",
        },
        "citation_representation": {
            "B3/code_B4": "Evidence contains explicit <|source_id|>N and Block N markers; citation references can be checked against these visible IDs.",
            "B5/code_B6": "Enrichment evidence contains source labels such as [Chunk], PubMed identifiers, and pages, but no numbered Block N mapping. A generated [Block N] reference is unresolved from the generator-visible payload and must be recorded as an unresolvable target, not silently mapped by retrieved_chunks order.",
        },
        "limitations": [
            "The original fully serialized prompt bytes are not archived.",
            "The archived evidence strings and historical routing branch are verified; exact tokenizer-rendered prompt reconstruction is not claimed.",
            "The B6 enrichment block is an archived output of graph expansion and is not regenerated here.",
        ],
    }


def _citation_metadata(evidence: str, code: str) -> list[str]:
    # Preserve visible source identifiers, not an inferred mapping to omitted chunks.
    patterns = [r"<\|source_id\|>\s*\d+", r"\bPubMed\s+\d+\b", r"\bPage\s+\d+\b", r"\[Chunk\]"]
    found: list[str] = []
    for pattern in patterns:
        found.extend(re.findall(pattern, evidence, flags=re.IGNORECASE))
    return list(dict.fromkeys(found))


def evidence_for_baseline(query: dict[str, Any], code: str) -> tuple[str, list[str], str]:
    baseline = query.get("baselines", {}).get(code)
    if not isinstance(baseline, dict) or baseline.get("status") != "success":
        return "", [], "UNAVAILABLE"
    if code == "B4":
        text = baseline.get("trimmed_text")
        graph = baseline.get("trimmed_graph")
        if not isinstance(text, str) or not text.strip() or not isinstance(graph, str) or not graph.strip():
            return "", [], "PARTIAL"
        # These exact post-trimming fields are inserted in ask_no_expander.
        evidence = text + "\n\n### KNOWLEDGE GRAPH CONNECTIONS:\n" + graph
        components = ["trimmed_text", "trimmed_graph"]
    elif code == "B6":
        enrichment = baseline.get("enrichment_block")
        if not isinstance(enrichment, str) or not enrichment.strip() or enrichment.strip() == "No essential knowledge graph enrichment found.":
            # Generation takes a fallback ask_no_expander branch; the archived fields
            # permit an exact evidence reconstruction if both post-trim fields exist.
            text, graph = baseline.get("trimmed_text"), baseline.get("trimmed_graph")
            if isinstance(text, str) and text.strip() and isinstance(graph, str) and graph.strip():
                return text + "\n\n### KNOWLEDGE GRAPH CONNECTIONS:\n" + graph, ["trimmed_text", "trimmed_graph"], "EXACT"
            return "", [], "PARTIAL"
        evidence = enrichment
        components = ["enrichment_block"]
    else:
        raise ValueError(f"unsupported code baseline {code}")
    return evidence, components, "EXACT"


def _metrics(baseline: dict[str, Any]) -> dict[str, Any]:
    values = baseline.get("eval_metrics") or {}
    return {metric: values.get(metric) for metric in ("semantic_accuracy", "faithfulness", "citation_fidelity")}


def _source_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def load_historical_records(run_dir: Path, repo: Path) -> tuple[list[dict], dict]:
    manifest = _yaml(run_dir / "run_manifest.yaml")
    source_commit = str(manifest.get("git_commit", ""))
    source_contract = verify_source_contract(repo, source_commit)
    contexts_path = run_dir / "retrieved_contexts.yaml"
    context_sha256 = sha256_file(contexts_path)
    expected_context_hash = str((manifest.get("dataset") or {}).get("dataset_hash", ""))
    if expected_context_hash != context_sha256:
        raise ValueError(f"retrieved_contexts.yaml hash does not match run manifest: {context_sha256}")
    expected_query_count = int((manifest.get("dataset") or {}).get("query_count", 0))
    if expected_query_count != 375:
        raise ValueError(f"expected historical 375-query run manifest, found {expected_query_count}")
    run_config = _yaml(run_dir / "config_snapshot.yaml")
    if (run_config.get("rag_components") or {}).get("graph_expansion") is not True:
        raise ValueError("run snapshot does not enable graph expansion for code B6")
    manifest_baselines = set(manifest.get("baselines", []))
    if not set(PAPER_TO_CODE.values()).issubset(manifest_baselines):
        raise ValueError(f"run manifest lacks expected B4/B6 code baselines: {manifest_baselines}")
    contexts = _yaml(contexts_path)
    by_id: dict[str, dict] = {}
    for item in contexts:
        qid = str(item.get("id", ""))
        if not qid or qid in by_id:
            raise ValueError(f"missing or duplicate retrieved-context ID: {qid!r}")
        by_id[qid] = item

    answers: dict[str, dict[str, dict]] = {}
    results_paths = {}
    for code in PAPER_TO_CODE.values():
        path = run_dir / "baselines" / f"result_metrics_judge_{code.lower()}.yaml"
        results_paths[code] = path
        payload = _yaml(path)
        rows = payload.get("results", [])
        if len(rows) != 375:
            raise ValueError(f"expected 375 output records for {code}, found {len(rows)}")
        for item in rows:
            qid = str(item.get("id", ""))
            if qid not in by_id:
                raise ValueError(f"answer output query missing from retrieved contexts: {qid}")
            if code not in item.get("baselines", {}):
                raise ValueError(f"missing {code} output under query {qid}")
            answers.setdefault(qid, {})[code] = item

    hist_scores_path = run_dir / "traces" / "eval_trace.jsonl"
    historical_scores: dict[tuple[str, str], dict[str, Any]] = {}
    if hist_scores_path.exists():
        for line in hist_scores_path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            code = str(row.get("baseline", ""))
            paper = {"B4": "B3", "B6": "B5"}.get(code)
            if paper:
                historical_scores[(str(row.get("query_id")), paper)] = row
    pilot_path = Path(__file__).resolve().parent / "second_judge_scores.jsonl"
    pilot = {}
    if pilot_path.exists():
        for line in pilot_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "scored":
                    pilot[(str(row.get("query_id")), str(row.get("paper_baseline")))] = row

    raw_context_sha = sha256_file(contexts_path)
    manifest_sha = sha256_file(run_dir / "run_manifest.yaml")
    config_sha = sha256_file(run_dir / "config_snapshot.yaml")
    records = []
    for qid in sorted(by_id):
        query = by_id[qid]
        query_records = {}
        for paper, code in PAPER_TO_CODE.items():
            context_record = query.get("baselines", {}).get(code) or {}
            output_row = answers[qid][code]
            output = output_row["baselines"][code]
            answer = output.get("generated_answer")
            reference = output_row.get("golden_answer")
            evidence, components, classification = evidence_for_baseline(query, code)
            if context_record.get("status") != "success": classification = "UNAVAILABLE"
            if not answer or not isinstance(answer, str): answer = ""
            if not reference or not isinstance(reference, str): reference = ""
            citations = _citation_metadata(evidence, code)
            answer_citations = re.findall(r"\[(?:Block\s*\d+|[^\]]*PubMed[^\]]*|[^\]]*PMID[^\]]*)\]", answer or "", flags=re.I)
            block_ids = re.findall(r"<\|source_id\|>\s*(\d+)", evidence)
            if code == "B4":
                citation_mapping_status = "explicit_numbered_source_ids" if set(block_ids) else "missing_numbered_source_ids"
                unresolvable = []
                valid_ids = set(block_ids)
                for citation in answer_citations:
                    match = re.search(r"Block\s*(\d+)", citation, flags=re.I)
                    if match and match.group(1) not in valid_ids:
                        unresolvable.append(citation)
            else:
                citation_mapping_status = "source_metadata_without_numbered_block_map" if citations else "source_metadata_missing"
                unresolvable = [c for c in answer_citations if re.search(r"Block\s*\d+", c, flags=re.I)]
            hist = historical_scores.get((qid, paper), {})
            second = pilot.get((qid, paper), {})
            metrics = _metrics(output)
            record = {
                "query_id": qid, "paper_baseline": paper, "code_baseline": code,
                "category": query.get("category", ""), "answerable": bool(query.get("is_answerable")),
                "query": str(query.get("query", "")),
                "generated_answer": answer, "generated_answer_available": bool(answer),
                "generated_answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest() if answer else "",
                "reference_answer": reference, "reference_answer_available": bool(reference),
                "reference_answer_sha256": hashlib.sha256(reference.encode("utf-8")).hexdigest() if reference else "",
                "evidence_classification": classification,
                "evidence_completeness": "complete archived generator-visible evidence payload; prompt bytes not archived" if classification == "EXACT" else "incomplete or unavailable",
                "evidence_components": components, "generator_visible_evidence": evidence,
                "generator_visible_evidence_sha256": hashlib.sha256(evidence.encode("utf-8")).hexdigest() if evidence else "",
                "citation_metadata_available": bool(citations), "visible_citation_metadata": citations,
                "citation_mapping_status": citation_mapping_status,
                "citation_references_in_answer": answer_citations,
                "unresolvable_citation_references": unresolvable,
                "historical_judge_scores": metrics,
                "historical_judge_score_available": any(v is not None for v in metrics.values()),
                "historical_eval_trace_available": bool(hist),
                "historical_eval_trace_metrics": {k: hist.get(k) for k in ("semantic_accuracy", "faithfulness", "citation_fidelity")},
                "second_judge_score_available": bool(second),
                "second_judge_score_source": second.get("evaluation_context_source", "") if second else "",
                "source_artifacts": {
                    "run_manifest": _source_record(run_dir / "run_manifest.yaml"),
                    "run_config": _source_record(run_dir / "config_snapshot.yaml"),
                    "retrieved_contexts": {"path": str(contexts_path), "sha256": raw_context_sha},
                    "baseline_answer_file": _source_record(results_paths[code]),
                },
            }
            if not answer: record["exclusion_reasons"] = ["generated answer missing"]
            elif not reference: record["exclusion_reasons"] = ["reference answer missing"]
            elif classification != "EXACT": record["exclusion_reasons"] = ["complete final generator-visible evidence not proven"]
            else: record["exclusion_reasons"] = []
            query_records[paper] = record
        records.extend(query_records[p] for p in PAPER_TO_CODE)

    source_contract["run_manifest_sha256"] = manifest_sha
    source_contract["run_config_sha256"] = config_sha
    source_contract["retrieved_contexts_sha256"] = raw_context_sha
    return records, source_contract


def choose_stratified_cohort(records: Iterable[dict], *, seed: int = 20261106,
                             answerable_n: int = 80, per_negative_category: int = 4) -> list[str]:
    by_id: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        by_id[r["query_id"]][r["paper_baseline"]] = r
    eligible = {qid: pair for qid, pair in by_id.items()
                if set(pair) == {"B3", "B5"} and all(x["evidence_classification"] == "EXACT" and x["generated_answer_available"] for x in pair.values())}
    strata: dict[str, list[str]] = defaultdict(list)
    for qid, pair in eligible.items():
        left = pair["B3"]
        strata["answerable" if left["answerable"] else left["category"]].append(qid)
    rng = random.Random(seed)
    chosen = []
    for stratum in sorted(strata):
        want = answerable_n if stratum == "answerable" else per_negative_category
        qids = sorted(strata[stratum])
        if len(qids) < want:
            raise ValueError(f"stratum {stratum} has {len(qids)} eligible IDs, need {want}")
        chosen.extend(rng.sample(qids, want))
    return sorted(chosen)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def build_outputs(run_dir: Path = DEFAULT_RUN, output_dir: Path = DEFAULT_OUTPUT,
                  repo: Path | None = None, *, seed: int = 20261106,
                  answerable_n: int = 80, per_negative_category: int = 4) -> dict[str, Any]:
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    repo = Path(repo or DEFAULT_RUN.parents[1]).resolve()
    records, source_contract = load_historical_records(run_dir, repo)
    if len(records) != 750: raise ValueError(f"expected 750 baseline records, found {len(records)}")
    cohort = choose_stratified_cohort(records, seed=seed, answerable_n=answerable_n,
                                      per_negative_category=per_negative_category)
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = list(records[0])
    with (output_dir / "corrected_evidence_eligibility.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in record.items()})

    by_id: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in records: by_id[record["query_id"]][record["paper_baseline"]] = record
    paired_rows = []
    for qid in cohort:
        pair = by_id[qid]
        b3, b5 = pair["B3"], pair["B5"]
        paired_rows.append({
            "query_id": qid, "category": b3["category"], "answerable": b3["answerable"],
            "query": b3["query"],
            "reference_answer": b3["reference_answer"] if b3["reference_answer_available"] else None,
            "metrics_to_score": (["semantic_accuracy"] if b3["answerable"] and b3["reference_answer_available"] else []) + ["faithfulness", "citation_fidelity"],
            "semantic_accuracy_status": "eligible_with_reference" if b3["answerable"] and b3["reference_answer_available"] else "excluded_designated_negative_not_independently_validated",
            "evidence_status": {"B3": "exact_archived", "B5": "exact_archived"},
            "judge_inputs": {
                "B3": {"paper_baseline": "B3", "code_baseline": "B4", "candidate_answer": b3["generated_answer"],
                       "generator_visible_evidence": b3["generator_visible_evidence"], "evidence_sha256": b3["generator_visible_evidence_sha256"],
                       "evidence_components": b3["evidence_components"], "citation_metadata": b3["visible_citation_metadata"],
                       "metrics_to_score": (["semantic_accuracy"] if b3["answerable"] and b3["reference_answer_available"] else []) + ["faithfulness", "citation_fidelity"]},
                "B5": {"paper_baseline": "B5", "code_baseline": "B6", "candidate_answer": b5["generated_answer"],
                       "generator_visible_evidence": b5["generator_visible_evidence"], "evidence_sha256": b5["generator_visible_evidence_sha256"],
                       "evidence_components": b5["evidence_components"], "citation_metadata": b5["visible_citation_metadata"],
                       "metrics_to_score": (["semantic_accuracy"] if b5["answerable"] and b5["reference_answer_available"] else []) + ["faithfulness", "citation_fidelity"]},
            },
            "historical_scores_reusable_for_corrected_protocol": False,
            "historical_scores_reuse_reason": "historical rubric/input evidence equivalence is not demonstrated; preserve as historical only",
            "source_run": str(run_dir),
        })
    _write_jsonl(output_dir / "paired_judge_inputs.jsonl", paired_rows)
    (output_dir / "paired_judge_query_ids.json").write_text(json.dumps({
        "schema_version": "1.0", "source_run": str(run_dir), "seed": seed,
        "stratum_design": {"answerable": answerable_n, "each_unanswerable_category": per_negative_category},
        "query_ids": cohort,
        "population_counts": {f"{category}|answerable={answerable}": count for (category, answerable), count in sorted(Counter((r["category"], r["answerable"]) for r in records if r["paper_baseline"] == "B3").items(), key=lambda x: str(x[0]))},
        "pair_count": len(cohort),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "r2_shared_rubric.md").write_text(RUBRIC, encoding="utf-8")
    budget = build_budget(paired_rows)
    (output_dir / "r2_judge_budget.json").write_text(json.dumps(budget, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = summarize(records, cohort, source_contract, run_dir, output_dir)
    packet_files = ["corrected_evidence_eligibility.csv", "paired_judge_inputs.jsonl",
                    "paired_judge_query_ids.json", "r2_shared_rubric.md", "r2_judge_budget.json"]
    packet_manifest = {"schema_version": "1.0", "source_run": str(run_dir),
                       "source_commit": source_contract["source_commit"],
                       "paper_to_code": PAPER_TO_CODE, "files": {name: {
                           "sha256": sha256_file(output_dir / name),
                           "size_bytes": (output_dir / name).stat().st_size}
                           for name in packet_files}}
    (output_dir / "r2_packet_manifest.json").write_text(json.dumps(packet_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["artifacts"]["packet_manifest"] = str(output_dir / "r2_packet_manifest.json")
    (output_dir / "r2_evidence_recovery_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_budget(pairs: list[dict], *, judges: int = 2, output_tokens_per_call: int = 256) -> dict[str, Any]:
    chars = 0
    per_arm = Counter()
    for pair in pairs:
        for arm in ("B3", "B5"):
            inp = pair["judge_inputs"][arm]
            char_count = sum(len(str(pair[k])) for k in ("query", "reference_answer")) + len(inp["candidate_answer"]) + len(inp["generator_visible_evidence"]) + len(RUBRIC)
            per_arm[arm] += char_count
            chars += char_count
    calls = len(pairs) * 2 * judges
    return {
        "paired_query_ids": len(pairs), "judges": judges, "evaluations": calls,
        "input_characters": chars * judges,
        "input_token_estimate_characters_divided_by_4": (chars * judges + 3) // 4,
        "input_token_estimate_is_unverified": True,
        "output_token_budget_per_call": output_tokens_per_call,
        "maximum_output_tokens": calls * output_tokens_per_call,
        "calls_per_judge": len(pairs) * 2,
        "metrics_per_evaluation": ["semantic_accuracy where eligible", "faithfulness", "citation_fidelity"],
        "note": "One structured response per judge/baseline/query scores the three metrics; invalid/error results are missing, never zero-imputed.",
    }


def summarize(records: list[dict], cohort: list[str], source_contract: dict, run_dir: Path, output_dir: Path) -> dict:
    by_id: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in records: by_id[record["query_id"]][record["paper_baseline"]] = record
    paired = [by_id[q] for q in sorted(by_id) if set(by_id[q]) == {"B3", "B5"}]
    both_answer = sum(all(p[b]["generated_answer_available"] for b in ("B3", "B5")) for p in paired)
    both_ref = sum(all(p[b]["reference_answer_available"] for b in ("B3", "B5")) for p in paired)
    both_evidence = sum(all(p[b]["evidence_classification"] in {"EXACT", "VERIFIED_RECONSTRUCTION"} for b in ("B3", "B5")) for p in paired)
    citation = sum(all(p[b]["citation_metadata_available"] for b in ("B3", "B5")) for p in paired)
    semantic = sum(all(p[b]["generated_answer_available"] and p[b]["reference_answer_available"] and p[b]["answerable"] for b in ("B3", "B5")) for p in paired)
    faith = sum(all(p[b]["generated_answer_available"] and p[b]["evidence_classification"] in {"EXACT", "VERIFIED_RECONSTRUCTION"} for b in ("B3", "B5")) for p in paired)
    citation_eligible = sum(all(p[b]["generated_answer_available"] and p[b]["evidence_classification"] in {"EXACT", "VERIFIED_RECONSTRUCTION"} and p[b]["citation_metadata_available"] for b in ("B3", "B5")) for p in paired)
    cohort_counts = Counter("answerable" if by_id[q]["B3"]["answerable"] else by_id[q]["B3"]["category"] for q in cohort)
    return {
        "historical_run": str(run_dir), "source_commit": source_contract["source_commit"], "paper_to_code": PAPER_TO_CODE,
        "classification_definition": {
            "EXACT": "The evidence payload inserted into the historical generator prompt is archived verbatim in the July run's final post-trim context fields, and source routing at the run commit selects those fields.",
            "VERIFIED_RECONSTRUCTION": "Not assigned; no non-archived evidence was synthesized.",
            "PARTIAL": "Evidence exists but required final context component or source routing cannot be verified.",
            "UNAVAILABLE": "No usable historical evidence record.",
            "Important": "EXACT refers to evidence payload, not byte-identical full system/user prompt.",
        },
        "source_contract": source_contract,
        "counts": {
            "query_pairs": len(paired), "both_generated_answers": both_answer,
            "both_reference_answers": both_ref, "complete_evidence_pairs": both_evidence,
            "complete_citation_metadata_pairs": citation,
            "explicit_numbered_citation_mapping_pairs": sum(all(p[b]["citation_mapping_status"] == "explicit_numbered_source_ids" for b in ("B3", "B5")) for p in paired),
            "citation_target_unresolvable_records": sum(bool(r["unresolvable_citation_references"]) for r in records),
            "semantic_accuracy_eligible_pairs": semantic,
            "faithfulness_eligible_pairs": faith,
            "citation_fidelity_eligible_pairs": citation_eligible,
            "all_metric_eligible_pairs": min(semantic, faith, citation_eligible),
            "generated_answers_by_paper_baseline": {paper: sum(r["paper_baseline"] == paper and r["generated_answer_available"] for r in records) for paper in ("B3", "B5")},
            "references_by_paper_baseline": {paper: sum(r["paper_baseline"] == paper and r["reference_answer_available"] for r in records) for paper in ("B3", "B5")},
            "semantic_accuracy_historical_scores_by_pair": {metric: sum(all(p[b]["historical_judge_scores"].get(metric) is not None for b in ("B3", "B5")) for p in paired) for metric in ("semantic_accuracy", "faithfulness", "citation_fidelity")},
            "faithfulness_historical_scores_by_pair": {metric: sum(all(p[b]["historical_judge_scores"].get(metric) is not None for b in ("B3", "B5")) for p in paired) for metric in ("semantic_accuracy", "faithfulness", "citation_fidelity")},
            "citation_fidelity_historical_scores_by_pair": {metric: sum(all(p[b]["historical_judge_scores"].get(metric) is not None for b in ("B3", "B5")) for p in paired) for metric in ("semantic_accuracy", "faithfulness", "citation_fidelity")},
            "second_judge_scored_records": sum(r["second_judge_score_available"] for r in records),
            "second_judge_scored_pairs": sum(all(p[b]["second_judge_score_available"] for b in ("B3", "B5")) for p in paired),
            "historical_score_available_pairs": sum(all(p[b]["historical_judge_score_available"] for b in ("B3", "B5")) for p in paired),
        },
        "cohort": {"selected_pairs": len(cohort), "by_stratum": dict(sorted(cohort_counts.items()))},
        "artifacts": {
            "eligibility_csv": str(output_dir / "corrected_evidence_eligibility.csv"),
            "paired_judge_inputs_jsonl": str(output_dir / "paired_judge_inputs.jsonl"),
            "paired_query_ids": str(output_dir / "paired_judge_query_ids.json"),
            "rubric": str(output_dir / "r2_shared_rubric.md"),
            "budget": str(output_dir / "r2_judge_budget.json"),
        },
        "historical_scores": "Retained in eligibility registry for audit, not reused in corrected evaluation absent rubric/input equivalence proof.",
        "previous_zero_eligible_audit_diagnosis": "The old builder already pointed at the July run, so this was not a CUDA-vs-July source-run error. It required exact archived full prompts or reconstruction of every prompt field and treated missing serialized prompt bytes as incomplete; the July archive preserves final evidence payloads and the exact historical branch/template routing, enough to verify generator-visible evidence without claiming byte-identical prompts.",
        "recommendation": "Proceed with the 100-pair answerability/category-stratified evaluation only after reviewer approval; both judges receive each baseline's own archived generator-visible evidence on exactly the same IDs.",
    }


RUBRIC = """# R2 Shared Frozen Judge Rubric (v1)\n\nScore each candidate answer on the same 0–1 scale. Return one JSON object with numeric `semantic_accuracy`, `faithfulness`, and `citation_fidelity`, each in [0,1], plus a short rationale and evidence references. Do not infer missing context. If an item cannot be evaluated because input evidence or reference is absent, return `status: invalid_input`; do not return 0 as a placeholder. Raw model output must be retained verbatim. Parsing failure, refusal, timeout, or malformed score is an error/missing result, never a zero.\n\n- **Semantic Accuracy:** For answerable questions with a supplied reference, compare the answer's factual conclusion with that reference and question. 1 = materially correct and complete; 0.5 = partly correct or materially incomplete; 0 = materially contradicted or wrong. For designated synthetic negatives without independent validation, mark this metric not evaluated in the confirmatory analysis. This metric does not require citation support.\n- **Faithfulness:** Judge each substantive answer claim only against the complete generator-visible evidence for that configuration. 1 = all material claims supported; 0.5 = mixed support; 0 = central claims unsupported/contradicted. Do not treat reference-answer agreement as evidence support. A justified abstention is faithful.\n- **Citation Fidelity:** Assess citation presence, correctness, and target support against only the visible citation/source metadata in that configuration's evidence. Missing citations on claims requiring source support reduce the score; incorrect/unresolvable citation targets and citations that do not support the associated claim reduce it. A correct source target does not excuse an unsupported claim. A justified abstention with no factual claim is not penalized for having no citation.\n\nJudges must score all three metrics independently, may not see the other judge's output, and must receive byte-identical frozen query/reference/rubric and configuration evidence/answer inputs across judges. The B3 and B5 contexts are intentionally configuration-specific and need not be identical to each other.\n"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo", type=Path, default=DEFAULT_RUN.parents[1])
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--answerable-n", type=int, default=80)
    parser.add_argument("--per-negative-category", type=int, default=4)
    args = parser.parse_args(argv)
    report = build_outputs(args.run_dir, args.output_dir, args.repo, seed=args.seed,
                           answerable_n=args.answerable_n, per_negative_category=args.per_negative_category)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
