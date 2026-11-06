"""Evidence-based eligibility inventory for historical B3/B4 and B5/B6 prompts.

This module deliberately treats archived judge prompts and verified reconstruction
as separate evidence classes. Retrieved chunks are inputs to reconstruction, not
proof that a complete historical prompt was sent.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

PAPER_TO_CODE = {"B3": "B4", "B5": "B6"}
STATUSES = ("exact_archived", "verified_reconstructed", "partial", "unavailable")
RECONSTRUCTION_FIELDS = ("query", "golden_answer", "candidate_answer", "rubric", "rendered_context")


@dataclass(frozen=True)
class Eligibility:
    query_id: str
    paper_baseline: str
    code_baseline: str
    status: str
    reason: str
    evidence: tuple[str, ...]


def classify_record(record: Mapping[str, Any]) -> Eligibility:
    """Classify one baseline/query from explicit artifact evidence.

    ``exact_archived_prompt`` proves archival. A verified reconstruction must
    include every prompt field and an explicit verification flag. Context chunks
    by themselves can only yield ``partial``.
    """
    query_id = str(record.get("query_id", "")).strip()
    paper = str(record.get("paper_baseline", "")).strip().upper()
    if not query_id:
        raise ValueError("query_id is required")
    if paper not in PAPER_TO_CODE:
        raise ValueError(f"unsupported paper baseline {paper!r}; expected B3 or B5")
    evidence = tuple(str(x) for x in record.get("evidence", []) if x)
    if record.get("exact_archived_prompt"):
        return Eligibility(query_id, paper, PAPER_TO_CODE[paper], "exact_archived",
                           "exact archived prompt artifact is present", evidence)
    if record.get("reconstruction_verified") is True and all(
        isinstance(record.get(field), str) and record[field].strip()
        for field in RECONSTRUCTION_FIELDS
    ):
        return Eligibility(query_id, paper, PAPER_TO_CODE[paper], "verified_reconstructed",
                           "all prompt fields are present and reconstruction was verified", evidence)
    has_any = bool(evidence) or any(record.get(k) for k in (
        "retrieved_chunks", "query", "golden_answer", "candidate_answer", "rubric",
        "rendered_context", "historical_score", "historical_eval_detail",
    ))
    status = "partial" if has_any else "unavailable"
    reason = ("saved artifacts exist but do not establish the complete prompt" if has_any
              else "no relevant saved prompt/input artifacts found")
    return Eligibility(query_id, paper, PAPER_TO_CODE[paper], status, reason, evidence)


def build_registry(records: Iterable[Mapping[str, Any]]) -> list[Eligibility]:
    """Return one row per ID/baseline, sorted and duplicate-safe."""
    rows: dict[tuple[str, str], Eligibility] = {}
    for record in records:
        row = classify_record(record)
        key = (row.query_id, row.paper_baseline)
        if key in rows:
            raise ValueError(f"duplicate eligibility record for {key}")
        rows[key] = row
    return [rows[k] for k in sorted(rows)]


def paired_eligibility(rows: Iterable[Eligibility]) -> dict[str, dict[str, str]]:
    """Summarize paired eligibility; a pair is eligible only if both sides qualify."""
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        by_id.setdefault(row.query_id, {})[row.paper_baseline] = row.status
    return {
        qid: {"B3": side.get("B3", "unavailable"), "B5": side.get("B5", "unavailable"),
              "paired_eligible": str(side.get("B3") in STATUSES[:2] and
                                     side.get("B5") in STATUSES[:2]).lower()}
        for qid, side in sorted(by_id.items())
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(rows: Iterable[Eligibility], path: Path) -> None:
    fields = ("query_id", "paper_baseline", "code_baseline", "status", "reason", "evidence")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            values["evidence"] = ";".join(row.evidence)
            writer.writerow(values)



def registry_from_historical_availability(path: Path) -> list[Eligibility]:
    """Conservative July artifact inventory from the preserved detail-availability table.

    The table records score/detail presence, not full rendered prompts. Consequently
    any such row is partial evidence; it can never promote a record to complete.
    Missing metric rows are treated as unavailable only when no other evidence exists.
    """
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            paper = row.get("paper_baseline", "").upper()
            if paper not in PAPER_TO_CODE:
                continue
            key = (row.get("query_id", "").strip(), paper)
            if not key[0]:
                continue
            grouped.setdefault(key, []).append(row)
    output: list[Eligibility] = []
    for (query_id, paper), items in sorted(grouped.items()):
        detail = any(x.get("historical_eval_detail_present", "").lower() == "true" for x in items)
        score = any(x.get("historical_score_value_present", "").lower() == "true" for x in items)
        evidence = [str(path), "saved_metric_detail" if detail else "",
                    "saved_numeric_score" if score else ""]
        output.append(Eligibility(query_id, paper, PAPER_TO_CODE[paper], "partial",
                                  "availability row is evidence of a historical record, but it does not preserve a complete rendered prompt"
                                  if detail or score else
                                  "availability row records the ID, but neither score detail nor a complete rendered prompt is present",
                                  tuple(x for x in evidence if x)))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records_jsonl", type=Path, nargs="?", help="saved per-ID artifact evidence records (JSONL)")
    parser.add_argument("--historical-availability", type=Path, help="inventory existing historical_detail_availability.csv")
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    rows = (registry_from_historical_availability(args.historical_availability)
            if args.historical_availability else build_registry(read_jsonl(args.records_jsonl))
            if args.records_jsonl else None)
    if rows is None:
        parser.error("provide records_jsonl or --historical-availability")
    write_csv(rows, args.output_csv)
    for status in STATUSES:
        print(f"{status}: {sum(row.status == status for row in rows)}")
    print(f"IDs/baseline rows: {len(rows)}; output: {args.output_csv}")


if __name__ == "__main__":
    main()


def build_detailed_historical_registry(
    release_dir: Path,
    historical_availability: Path,
    second_judge_scores: Path,
) -> list[dict[str, Any]]:
    """Join historical answers/references, saved score details, and pilot presence.

    This records recovered components per ID, while preserving prompt/evidence
    status as partial because no complete archived generator prompts were found.
    """
    import hashlib
    import re

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required only to read the saved historical score YAML") from exc

    historical = {}
    with historical_availability.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["query_id"], row["paper_baseline"])
            state = historical.setdefault(key, {"metrics": set(), "details": set()})
            state["metrics"].add(row.get("metric", ""))
            if row.get("historical_eval_detail_present", "").lower() == "true":
                state["details"].add(row.get("metric", ""))

    pilot = set()
    with second_judge_scores.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("status") == "scored":
                pilot.add((item.get("query_id"), item.get("paper_baseline")))

    records = []
    for paper, code in PAPER_TO_CODE.items():
        path = release_dir / "baselines" / f"result_metrics_judge_{code.lower()}.yaml"
        with path.open(encoding="utf-8") as stream:
            payload = yaml.load(stream, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
        results = payload.get("results", [])
        if len(results) != 375:
            raise ValueError(f"expected 375 historical rows in {path}, found {len(results)}")
        for item in results:
            qid = str(item.get("id", ""))
            baseline = item.get("baselines", {}).get(code, {})
            answer = baseline.get("generated_answer")
            reference = item.get("golden_answer")
            metrics = baseline.get("eval_metrics", {})
            state = historical.get((qid, paper), {"metrics": set(), "details": set()})
            citations = sorted(set(re.findall(r"\[(?:Block\s*\d+|[^\]]*PubMed[^\]]*)\]", answer or "", flags=re.I)))
            records.append({
                "query_id": qid, "paper_baseline": paper, "code_baseline": code,
                "status": "partial", "evidence_class": "partial",
                "generated_answer_available": bool(answer),
                "generated_answer_sha256": hashlib.sha256(answer.encode()).hexdigest() if answer else "",
                "reference_answer_available": bool(reference),
                "reference_answer_sha256": hashlib.sha256(reference.encode()).hexdigest() if reference else "",
                "historical_semantic_accuracy": metrics.get("semantic_accuracy"),
                "historical_faithfulness": metrics.get("faithfulness"),
                "historical_citation_fidelity": metrics.get("citation_fidelity"),
                "historical_score_metrics_present": ";".join(sorted(state["metrics"])),
                "historical_eval_details_present": ";".join(sorted(state["details"])),
                "second_judge_pilot_scored": (qid, paper) in pilot,
                "second_judge_pilot_context_source": "reconstructed retrieved_chunks; not full generator prompt" if (qid, paper) in pilot else "",
                "citation_references_in_answer": ";".join(citations),
                "citation_metadata_in_saved_evidence": bool(citations),
                "complete_generator_visible_evidence": False,
                "exact_archived_prompt": False,
                "verified_reconstruction": False,
                "prompt_provenance_status": "partial; prompt hash/context components do not prove exact final serialized prompt",
                "corrected_judge_eligible": False,
                "ineligibility_reason": "complete generator-visible evidence and exact prompt provenance are not verified",
            })
    records.sort(key=lambda x: (x["query_id"], x["paper_baseline"]))
    if len(records) != 750 or len({r["query_id"] for r in records}) != 375:
        raise ValueError("historical registry must contain 375 paired IDs / 750 system records")
    return records


def write_detailed_registry(records: Iterable[Mapping[str, Any]], output: Path) -> None:
    records = list(records)
    if not records:
        raise ValueError("cannot write an empty eligibility registry")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
