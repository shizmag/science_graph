# Source and checksum manifest

## Primary experiment sources

- Run ID: `20261020T065029857103Z`
- Run root: `graphs/run_20261020T065029857103Z_OCC-RAG-1.7B_CUDA_ABLATION/`
- Judge root: `reports/ablation_luna6_low/run_20261020T065029857103Z/`
- Run package status: `PARTIAL` (manifest records 5,250 expected operation records; it is not an all-success completion flag).
- Judge status: `COMPLETE`; 4,265 eligible results scored; 985 unscored operation-ineligible rows.
- Frozen questions: 375 (300 answerable, 75 designated-negative). Dataset SHA-256: `d2820ced7c2994218245887ad753700923262fb91ec983eeacc01b217ec43b6b`.
- `external_db` restored snapshot SHA-256: `c7fdd7c6fed69b788563896efaa8cbfe204216d6ff82043ff3a7414267aba8f0`; July-source identity equivalence: not proven.
- Generator revision: `e7d9f8b912dc1ac9a66ab42f0d07a7663d6ed58c`; embedder `d128750597153bb5987e10b1c3493a34e5a4502a`; cross-encoder `1427fd652930e4ba29e8149678df786c240d8825`.
- Internal B6 maps to paper-facing B5. This run is not historical B3.
- Run `SHA256SUMS`: 14,891 listed files independently verified present and digest-matching in the audit environment.

## Judge provenance

- Model route: OmniRoute `codex/gpt-6-luna-low`; canonical `gpt-6-luna-low`; temperature 0; reasoning effort low; seed 20261020.
- Judge system/rubric text SHA-256: `920702b80a4bbfa2680e97b277ee279b81a2ac528a3b73cbda0de192b9f8641a`.
- Judge manifest source judge-input hash: `7ff43a02ba0e43aed6df36a12cf45fb2fc54a173b60394f2b2d0e9d4b4f6c615`; score export hash: `ec1e4595174aa013a291d2e55ef6429c5b63c23e9812053757c9d8477e00ee28`; mapping hash: `3e12f710fb6fa63e0dd88040c8743c18ea8ace98e74356c969b6f9acb7101d28`.
- 4,265 eligible slots; 2,297 distinct exact case payloads; 1,968 reused slots; max exact-payload multiplicity 9. Per-case digest includes question, answerability label, reference answer, actual evaluator context, cleaned output. Global system rubric/model/decode fixed across single run. No cross-run cache. Independent per-batch request replay not done.

## Reproducible outputs

- Script: `revision_ablation_defense/analyze_existing.py`
- Regenerate using `.venv/bin/python revision_ablation_defense/analyze_existing.py` from repository root. Script reads preserved local artifacts/Parquet only. No model generation or judge API calls.
- Canonical 5,250-row operation ledger: `tables/canonical_5250.csv`.
- All summary tables and their definitions are described in the corresponding numbered analysis notes and `supplementary_ablation_addendum.md`.
- Integrity assertion result: `integrity_audit.json`.

## Reviewer source provenance

The exact Knowledge-Based Systems decision letter was not found in the local research artifact inventory. Provisional reviewer wording and section numbering come from `revision_manuscript/response_to_reviewers_draft.md`; compare all quotations against the official editorial system before submission. No manuscript source/PDF page-line references were added or modified.
