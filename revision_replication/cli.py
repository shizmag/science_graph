"""Metadata-only and explicitly gated commands for local paired replication."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from revision_replication.gemma_mlx import AUTHORIZATION, GemmaMLXAdapter
from revision_replication.provenance import GEMMA_PATH, ReplicationProvenance, inspect_gemma_metadata
from revision_replication.runner import PairedRunner, select_cohort


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _runner(args):
    provenance = ReplicationProvenance(**json.loads(args.provenance.read_text()))
    rows = _read_jsonl(args.rows)
    ids = sorted({r["query_id"] for r in rows})
    if args.cohort_ids:
        cohort = json.loads(args.cohort_ids.read_text())
        if not isinstance(cohort, list) or any(not isinstance(x, str) for x in cohort):
            raise ValueError("--cohort-ids must be a JSON list of query ID strings")
    else:
        cohort = select_cohort(ids, size=args.cohort_size or len(ids), seed=args.seed)
    rows = [row for row in rows if row["query_id"] in set(cohort)]
    generator = GemmaMLXAdapter(model_path=args.model_dir, context_limit=args.context_limit)
    runner = PairedRunner(generator=generator, provenance=provenance, output_dir=args.output_dir, context_limit=args.context_limit)
    authorized = getattr(args, "authorize_real_generation", False)
    result = runner.run(rows, cohort=cohort, run_id=args.run_id, max_tokens=args.max_tokens,
                        dry_run=not authorized, authorization=AUTHORIZATION if authorized else None)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect-gemma")
    inspect.add_argument("--model-dir", type=Path, default=GEMMA_PATH)
    plan = sub.add_parser("plan")
    execute = sub.add_parser("run")
    for command in (plan, execute):
        command.add_argument("--rows", type=Path, required=True, help="JSONL rows with query_id, code_baseline B4/B6, and prompt")
        command.add_argument("--provenance", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--run-id", required=True)
        command.add_argument("--model-dir", type=Path, default=GEMMA_PATH)
        command.add_argument("--context-limit", type=int, default=12000)
        command.add_argument("--max-tokens", type=int, default=768)
        group = command.add_mutually_exclusive_group()
        group.add_argument("--cohort-size", type=int)
        group.add_argument("--cohort-ids", type=Path, help="JSON array of a prespecified, frozen query cohort")
        command.add_argument("--seed", default="gemma-r1-confirmatory-v1")
    execute.add_argument("--authorize-real-generation", action="store_true",
                         help="Explicitly authorizes actual local generation; omit for a metadata-only plan")
    if argv is None:
        import sys
        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    if args.command == "inspect-gemma":
        print(json.dumps(inspect_gemma_metadata(args.model_dir), indent=2, sort_keys=True))
        return 0
    if args.command == "plan":
        # Ensure plan command cannot authorize model execution by construction.
        args.authorize_real_generation = False
    return _runner(args)

if __name__ == "__main__":
    raise SystemExit(main())
