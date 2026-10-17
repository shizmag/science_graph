from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ablation_portability.storage import sha256_file


def record_decision(snapshot_root: Path, candidate: str, statement: str) -> dict:
    snapshot_root = snapshot_root.resolve(strict=True)
    statement = statement.strip()
    if not statement:
        raise ValueError("An author decision statement is required")
    candidate_dirs = sorted(
        path for path in snapshot_root.glob("candidates/*/v1")
        if (path / "snapshot_manifest.json").is_file()
    )
    if not candidate_dirs:
        raise FileNotFoundError(f"No versioned candidate snapshots found under {snapshot_root}")
    candidates = {}
    for directory in candidate_dirs:
        manifest_path = directory / "snapshot_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        name = manifest.get("source_candidate")
        if not name or name in candidates:
            raise ValueError(f"Missing or duplicate source candidate identity in {manifest_path}")
        if manifest.get("baseline_selection", {}).get("status") == "author_selected":
            raise ValueError(
                f"An author-selected candidate already exists ({name}); record a revised decision explicitly"
            )
        candidates[name] = (directory, manifest_path, manifest)
    if candidate not in candidates:
        raise ValueError(f"Unknown candidate {candidate!r}; available: {sorted(candidates)}")

    selected = candidates[candidate]
    selected_db_hash = selected[2].get("snapshot_identity", {}).get("sha256")
    record = {
        "schema_version": "1.0",
        "selected_candidate": candidate,
        "selected_snapshot_sha256": selected_db_hash,
        "decision_statement": statement,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidates_considered": sorted(candidates),
    }
    decision_path = snapshot_root / "baseline_decision.json"
    decision_path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    decision_hash = sha256_file(decision_path)

    for name, (_, manifest_path, manifest) in candidates.items():
        if name == candidate:
            manifest["baseline_selection"] = {
                "status": "author_selected",
                "author_selection": candidate,
                "decision_statement": statement,
                "decision_file": "../../../baseline_decision.json",
                "decision_sha256": decision_hash,
                "selected_database_sha256": selected_db_hash,
            }
        else:
            manifest["baseline_selection"] = {
                "status": "candidate_not_selected",
                "author_selection": candidate,
                "decision_file": "../../../baseline_decision.json",
                "decision_sha256": decision_hash,
            }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Record an explicit author choice of the new frozen database baseline")
    parser.add_argument("snapshot_root", type=Path, help="Directory containing candidates/<name>/v1 snapshots")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--statement", required=True, help="Short author-provided rationale for the choice")
    args = parser.parse_args()
    print(json.dumps(record_decision(args.snapshot_root, args.candidate, args.statement), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
