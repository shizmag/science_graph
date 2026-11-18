# Reviewer #1 — historical full-evidence F/CF reassessment

**IMPLEMENTED AND VERIFIED (offline). READY FOR API EXECUTION. PENDING RESULTS.**
No new B3/B5 generations. No paid API requests made. Historical archives and R2 v3
results are not modified. All new outputs live in this directory.

## Reproduce from repository root
Use Python 3.12 and the package versions in `environment_versions.json` (PyYAML,
jsonschema, numpy, scipy, pytest). Existing environment: `.venv/bin/python`.

```sh
# Phase A: verifies original source/run/packet, raw answers, 160 inputs, 100 fixtures.
# Full 313 MB historical YAML parse can take several minutes and substantial RAM.
.venv/bin/python revision_round2/reviewer1_evidence_rejudge/runner/prepare.py
# Refuses any differing overwrite. Verify/create frozen manifest offline:
.venv/bin/python revision_round2/reviewer1_evidence_rejudge/runner/freeze.py
.venv/bin/python -m pytest -q revision_round2/reviewer1_evidence_rejudge/tests
# Explicitly OFFLINE, no credential/network use:
.venv/bin/python revision_round2/reviewer1_evidence_rejudge/runner/run.py
```

Only **after explicit user approval** for paid judging:

```sh
.venv/bin/python revision_round2/reviewer1_evidence_rejudge/runner/run.py --authorize-judge-execution
```

Configure the same credentials as R2 v3: OmniRoute uses `OMNIROUTE_API_KEY` or
`REVISION_JUDGE_API_KEY`, `REVISION_JUDGE_BASE_URL` at local port 20128 `/v1`;
OpenRouter uses `OPENROUTER_API_KEY`. Existing R2 env-loading helper is reused;
never print or include secrets in a public package. No model substitution.
320 first-pass calls expected, maximum 1280 attempts. No current price quotation
has been verified; do not assume earlier provider pricing is still applicable.
Maximum output-token reservation: 3.84 million on first passes, 15.36 million with
all retries, PLUS input tokens. Actual usage and billing can differ.

Run the same command to resume: valid checkpoints are not rejudged; exhausted/fatal
or uncertain attempts remain explicit and require documented reconciliation or a
new version. Do not delete attempt histories to reset retries. Process locking
prevents concurrent runners. `attempts/`, `started/`, `accepted/` retain immutable
per-attempt/checkpoint files; `accepted_scores.jsonl` is a derived export.
A crash may leave exports stale; analysis reads authoritative per-task checkpoints.

After judging:

```sh
.venv/bin/python revision_round2/reviewer1_evidence_rejudge/statistical_analysis.py
```

Analysis performs no LLM calls. It verifies the freeze, task/request/response identity
and recomputes scores, writes `analysis_results/summary.json`. Partial runs are labelled
PENDING RESULTS. All metric-specific common IDs and judge-specific exclusions are
retained. Do not interpret currently empty results as zero scores.

## Package map
- `protocol.md`, `judge_protocol/`: frozen scientific rubric, schema and policy.
- `cohort_manifest.json`: original fixed80 IDs and conditional selection provenance.
- `evidence_manifest.jsonl`: internal answer/reference/raw/evidence/prompt identity.
- `reconstructed_prompt_manifest.jsonl`: historical-template prompt reconstruction.
- `judge_tasks.jsonl`: 320 internal mappings, each with explicit blind `visible` payload.
- `provenance_checks.json`, `provenance_validation.md`: source/hash/fixture validation.
- `freeze.json`: checksummed input, code and rubric manifest; protocol fingerprint is
  printed by dry-run and recorded in `local_dry_run.json`.
- `tests/`, `local_test_results.txt`: local mocks and archive identity assertions.
- `statistical_analysis.py`, `analysis_results/`: deterministic paired analysis.
- `experiment_report.md`, `reviewer_response_draft.md`, `manuscript_changes.md`:
  scientifically bounded reporting, with PENDING RESULTS where appropriate.
- `security_screening.json`: new-manifest screening, not public-release approval.

Public release requires a separate manual review of raw answers, scientific text,
provider metadata, licensing and identifiers. Do not copy `.env`, original config
snapshots, or historical archives wholesale into a public supplement.
