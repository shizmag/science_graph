# File-based blinded Judge B

From the repository root, the packets are generated with:

```sh
.venv/bin/python revision_round1/judge/antigravity_bridge/bridge.py prepare
.venv/bin/python revision_round1/judge/antigravity_bridge/bridge.py validate
.venv/bin/python revision_round1/judge/antigravity_bridge/bridge.py status
```

Run `agy` interactively from the repository root, choose the independent model in the UI, and paste the prompt in `AGY_RUNBOOK.md` with a new session ID and the actual model version (or `unknown`). Stop at the batch boundary. Start another `agy` session with a new ID and paste the same prompt to resume. If the tool cannot access a Git-ignored results directory, change the IDE's ignored-file access settings for `agy_r2_workspace/results`, `sessions`, and `corrections`; do not disable source-control exclusions. The author must confirm the application's actual filesystem access. The bridge's workspace isolation is an instruction, not an operating-system security sandbox: configure workspace write restrictions in Antigravity if available.

Each task has an opaque ID, embedded rubric and complete evidence. Public packets do not contain the hidden key mapping; the mapping is `private/mapping.json`, excluded by the repository's bridge-private ignore rule. Do not open it or export results during judging. Judge-facing text may contain natural citation identifiers from the research; no system labels or previous scores are added by the bridge. Task hashes and the mapping are deterministically regenerated. Preparation refuses to overwrite changed artifacts. Result and session paths are Git-ignored by the workspace's `.gitignore`; do not force-add them. The scratch `draft.json` should be deleted after use and is also ignored.

Results are externally processed by the interactive model. Explicitly authorize research-data processing under your institutional, provider and copyright policies before judging. The preparation checks common credential formats and environment secrets but cannot prove the absence of private health information or other restricted content. Verify data permissions independently. No model/API calls occur in the bridge. No results have been generated during preparation.

When judging is complete (or for a clearly labeled partial analysis), the author—not the blinded judge—can unblind and export:

```sh
.venv/bin/python revision_round1/judge/antigravity_bridge/bridge.py export --output agy_r2_workspace/exports --luna-dir revision_round1/experiments/r2_v2_pilot
```

The export validates hashes, uses matching v2 Luna records only, creates per-response and query-paired CSVs plus summary JSON, and reports paired bootstrap intervals and agreement where sufficient. The partial Luna pilot is not a full-cohort result. Self-reported model metadata do not prove independent model families. Store the private mapping and raw results safely; neither should be given to the judge. For an explicit correction, retain the original and run `submit DRAFT --correction-reason 'reason'`; results are versioned with a hash of the prior version. Do not edit result files directly.
