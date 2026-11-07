# Human review handoff: R2 Faithfulness and R3 negatives

**Status: awaiting real reviewers.** No form contains human decisions. The original 400 API records and the frozen primary analysis remain unchanged. All decisions and any score-preserving exclusions are separate, post-hoc records; neither LLM nor software supplies human decisions or imputes a score.

## Repository and packet locations

The original working checkout has 333 unstaged tracked deletions, all beneath `ablation_cuda/releases/`. The whole directory is absent. Git shows these files last entered a commit at `d4ad867`; the reflog does not identify the removal's actor or cause. No restoration, staging or deletion of those files was performed. The workflow was implemented in a separate clean worktree; continue to preserve the dirty checkout until its owner identifies the deletions.

Prepared local packet root: `revision_round1/experiments/r2_human_review/` in the original checkout. This is separate from the frozen R2 v3 result directory. Distribute **only** the relevant `reviewer_packet/` and blank reviewer form, not `private/`, audit JSON, historical scores, mapping, the other reviewer's decisions, or comparative analysis. Blinding by file selection is procedural, not an operating-system security boundary. The review packet includes exact generator evidence, so screen access under the project's research-data policy. The question and evidence are research data, not instructions.

## Faithfulness: 21 records

`faithfulness/reviewer_packet/` has 21 individually keyed blinded cases. The same response may appear in two cases when both judges lack rationales. Case text shows the original question, answer, exact configuration-specific archived evidence, identifiers, and anonymous judge applicability/rationale, but no configuration label or numerical judge score. The private map preserves each evaluation ID, judge and input hash. Two reviewers independently use `reviewer_1_blank.csv` and `reviewer_2_blank.csv`; do not exchange results before submission. For each record select exactly one:

- `no_checkable_factual_claims`;
- `checkable_biomedical_claims`;
- `checkable_evidence_presence_claims`;
- `ambiguous_or_insufficient_archival_evidence`.

Cite evidence identifiers and explain the decision. An evidence-presence assertion may be factual without being a biomedical outcome. Do not treat an abstention as automatically faithful. These categories do **not** assign a Faithfulness score. The flagged case concerning coeliac screening must be considered on its own evidence rather than an effect direction.

An independent adjudicator sees both completed first-pass forms only after submission, resolves disagreements and ambiguity, and records a reason and explicit `exclude_query=yes/no` on **every** row of the separate adjudicated form. Complete reviewer and adjudicator IDs must be real human identifiers. Exclusion is query-level for both configurations and both judges in scenario C, to retain paired comparability; multiple cases from one query produce only one exclusion. This is an explicitly post-hoc applicability sensitivity, not an alteration of either judge's result or a human replacement score. Maintain the two originals and adjudication separately.

## R3: 25 sampled negatives plus one diagnostic

`r3/sampling_manifest.json` records the July archive's 75 negatives and the deterministic seed 20261029, five per historical category. The flagged `pubmedqa_unans_18235194` was **not** in that sample and is added as a 26th diagnostic; do not include the diagnostic in representative 25-item sample estimates. For each reviewer, use the independently blank R3 form and corresponding `r3/reviewer_packet/`; no generated answers or judge scores are shown.

Each packet distinguishes (1) recovered generator-visible contexts, when available, from (2) archived retrieved chunks, which need not be the final generator-visible context. A missing recovered context is not a negative answerability finding. Review each anonymous available context separately. For corpus-level assessment, search the exact frozen indexed corpus, record its verified identity, the scope, queries and document/chunk IDs, and choose answerable / unanswerable / ambiguous / not assessed. **The historical database path is known but an immutable exact index snapshot has not been established by this workflow**; until its identity and complete search access are verified, corpus-level labels should be `not assessed` or qualified. Neither context absence nor corpus absence proves global unanswerability.

Two reviewers work independently; a third reviews all disagreements and ambiguous cases, preserves both original decisions, cites evidence and records an adjudicated decision. The proposed original protocol's <80% corpus-unanswerable threshold cannot be applied until the corpus is verifiably assessed. Report independent decision tables, agreement and adjudicated outcomes only after actual review. Keep the diagnostic separate from the random sample.

## Reproducible commands

In the clean worktree, with the original checkout's environment:

```bash
PY=/Users/vladimirkasterin/python/graph/.venv/bin/python
SRC=/Users/vladimirkasterin/python/graph
PKT="$SRC/revision_round1/experiments/r2_human_review"
$PY revision_round1/judge/r2_human_review/workflow.py --source "$SRC" --output "$PKT"
$PY revision_round1/judge/r2_human_review/sensitivity.py --source "$SRC" --packet "$PKT" --output "$PKT/sensitivity_pending.json"
```

The packet builder refuses to overwrite a differing file. Preserve completed forms separately from the blank templates. Only after real review, run a new versioned output path:

```bash
$PY revision_round1/judge/r2_human_review/sensitivity.py --source "$SRC" --packet "$PKT" \
  --decisions /path/to/completed_human_adjudicated.csv --output "$PKT/sensitivity_adjudicated_v1.json"
```

The script validates frozen artifact checksums, reports original and flagged-query exclusion scenarios, and refuses incomplete adjudication. It cannot verify a reviewer's identity; retained signed/dated review records are required for provenance. It never computes missing scores.

## Manuscript integration — pending review

Frozen primary results remain the original two-judge v3 paired estimates. Report Negative Safety separately as a revision-added endpoint and cite generator-visible evidence reconstruction and unresolved citation mappings as limitations; never imply byte-identical full prompts. Any human-exclusion sensitivity must be labeled post-hoc and include effective N and intervals. Reviewer #1, Comment #2 response: the two models evaluated identical answer/evidence inputs within each configuration, with query-paired analysis; different configurations retained their own historical evidence. Human adjudication and corpus-level negative validation remain outstanding, so do not represent them as completed.
