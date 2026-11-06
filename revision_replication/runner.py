"""Small deterministic paired-run runner with checkpoints and opt-in real generation."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Protocol

from revision_replication.gemma_mlx import AUTHORIZATION
from revision_replication.provenance import ReplicationProvenance


class Generator(Protocol):
    requires_real_execution_authorization: bool
    def generate(self, prompt: str, *, max_tokens: int = 768, authorization: str | None = None) -> dict: ...


def select_cohort(query_ids: list[str], *, size: int, seed: str) -> list[str]:
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query IDs must be unique")
    if size < 1 or size > len(query_ids):
        raise ValueError("cohort size must be between 1 and available queries")
    return sorted(query_ids, key=lambda q: hashlib.sha256(f"{seed}:{q}".encode()).hexdigest())[:size]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _atomic_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(path, flags, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
        return
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class PairedRunner:
    def __init__(self, *, generator: Generator, provenance: ReplicationProvenance, output_dir: Path, context_limit: int = 12000):
        provenance.validate()
        if context_limit < 1:
            raise ValueError("context_limit must be positive")
        self.generator = generator
        self.provenance = provenance
        self.output_dir = Path(output_dir)
        self.context_limit = context_limit

    def plan(self, rows: list[dict], *, cohort: list[str], run_id: str, max_tokens: int = 768) -> dict:
        if len(set(cohort)) != len(cohort):
            raise ValueError("frozen cohort contains duplicate query IDs")
        selected = set(cohort)
        pairs: dict[str, set[str]] = {}
        observed: set[tuple[str, str]] = set()
        for row in rows:
            key = (row["query_id"], row["code_baseline"])
            if key in observed:
                raise ValueError(f"duplicate query/baseline input row: {key}")
            observed.add(key)
            pairs.setdefault(row["query_id"], set()).add(row["code_baseline"])
        invalid = [qid for qid in cohort if pairs.get(qid) != {"B4", "B6"}]
        if invalid:
            raise ValueError(f"cohort must have exactly code B4 and B6 rows; invalid IDs: {invalid[:5]}")
        if any(row["query_id"] not in selected for row in rows):
            raise ValueError("rows include IDs outside frozen cohort")
        if observed != {(qid, code) for qid in cohort for code in ("B4", "B6")}:
            raise ValueError("input rows must contain exactly one record for each frozen query/baseline pair")
        plan = {
            "schema_version": "1.0", "run_id": run_id, "mode": "plan_only",
            "paper_to_code": {"B3": "B4", "B5": "B6"}, "cohort_query_ids": sorted(cohort),
            "planned_operations": 2 * len(cohort), "max_new_tokens": max_tokens,
            "context_limit": self.context_limit, "provenance": self.provenance.as_dict(),
            "execution_authorized": False,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.output_dir / "run_manifest.json"
        if manifest.exists():
            old = json.loads(manifest.read_text())
            if old != plan:
                raise FileExistsError("run manifest exists and differs; use a new run directory")
        else:
            _atomic_json(manifest, plan, exclusive=True)
        return plan

    def run(self, rows: list[dict], *, cohort: list[str], run_id: str, max_tokens: int = 768,
            dry_run: bool = True, authorization: str | None = None) -> dict:
        plan = self.plan(rows, cohort=cohort, run_id=run_id, max_tokens=max_tokens)
        if dry_run:
            return plan
        if getattr(self.generator, "requires_real_execution_authorization", False) and authorization != AUTHORIZATION:
            raise PermissionError("real generator execution is disabled without explicit authorization")
        if not rows:
            return plan
        checkpoint = self.output_dir / "operations.jsonl"
        old = {}
        if checkpoint.exists():
            for line in checkpoint.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    old[(rec["query_id"], rec["code_baseline"])] = rec
        expected = {(qid, code) for qid in cohort for code in ("B4", "B6")}
        with checkpoint.open("a", encoding="utf-8") as stream:
            for row in sorted(rows, key=lambda item: (item["query_id"], item["code_baseline"])):
                key = (row["query_id"], row["code_baseline"])
                if key in old:
                    if old[key].get("input_sha256") != _sha(row["prompt"]):
                        raise ValueError(f"checkpoint input changed for {key}")
                    continue
                record = {"run_id": run_id, "query_id": key[0], "code_baseline": key[1], "paper_baseline": {"B4": "B3", "B6": "B5"}[key[1]],
                          "input_sha256": _sha(row["prompt"]), "evidence_ids": row.get("evidence_ids", []),
                          "evidence_sha256": _sha(json.dumps(row.get("evidence", []), sort_keys=True)),
                          "status": "pending"}
                try:
                    response = self.generator.generate(row["prompt"], max_tokens=max_tokens, authorization=authorization)
                    record.update(response)
                    record.setdefault("status", "model_error")
                except Exception as exc:
                    record.update(status="model_error", error_type=type(exc).__name__, error=str(exc))
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                old[key] = record
        missing = expected - set(old)
        summary = {**plan, "mode": "execute", "completed_operations": len(expected) - len(missing), "missing_operations": sorted(missing)}
        if not missing:
            raw = checkpoint.read_bytes()
            final = {
                "schema_version": "1.0", "run_id": run_id,
                "plan_sha256": _sha(json.dumps(plan, sort_keys=True, separators=(",", ":"))),
                "operations_sha256": hashlib.sha256(raw).hexdigest(),
                "operation_count": len(expected),
                "status_counts": {status: sum(1 for key in expected if old[key].get("status") == status)
                                  for status in sorted({old[key].get("status", "unknown") for key in expected})},
                "paper_to_code": {"B3": "B4", "B5": "B6"},
                "provenance": self.provenance.as_dict(),
            }
            result_path = self.output_dir / "results_manifest.json"
            if result_path.exists():
                if json.loads(result_path.read_text()) != final:
                    raise ValueError("immutable results manifest disagrees with completed operations")
            else:
                _atomic_json(result_path, final, exclusive=True)
        return summary
