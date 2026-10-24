# Integrity and accounting audit

## Scope and artifacts

This audit joins the frozen dataset, `metrics_details.csv`, `generation_results.jsonl`, per-arm result JSON, prompt/context traces, judge input export, separate response mapping, final score JSONL, judge script and judge manifest. Run: `20261020T065029857103Z`; internal reference B6 maps to paper baseline B5. Dataset SHA-256 `d2820ced7c2994218245887ad753700923262fb91ec983eeacc01b217ec43b6b`; selected `external_db` snapshot `c7fdd7c6fed69b788563896efaa8cbfe204216d6ff82043ff3a7414267aba8f0`; the manifest explicitly says July snapshot identity is **not proven**. GPU/runtime/model revisions and component-arm hashes are retained in the run manifest. This is an additional within-run CUDA sensitivity analysis, not a reanalysis of the July MLX B3–B5 contrast and not proof that the historical difference has been causally decomposed.

## Reconstructed operation ledger

`tables/canonical_5250.csv` is a query × arm ledger, not a response-only table. It has exactly 5,250 unique `(arm_id, query_id)` rows, the same 375 frozen query IDs in each of the 14 arms, 300 source-labeled answerable questions and 75 synthetic designated-negative questions. It joins operation status, prompt/failure token count, context-character count, output token count, retrieval/context measurements, latency, parser status, entropy and retained mass, response/context/config hashes, judge IDs/status and nullable judge scores. Missing measurements are null/empty and failed answers are not scored as zero.

| Planned | Successful generation | Prompt-overflow failures | Judge score rows | Ineligible/unscored |
|---:|---:|---:|---:|---:|
| 5,250 | 4,265 (81.24%) | 985 (18.76%) | 4,265/4,265 eligible | 985 |

Checks independently pass: per-arm population, no duplicate operation keys, one result record per planned operation, generation JSON/hash agreement, no eligible score missing, no failed operation scored, and common dataset/snapshot/model-manifest identities. Failures are all non-transient `RuntimeError: Serialized prompt has N tokens; the configured context limit leaves no room for generation (12000)` errors. The run remains **PARTIAL**; the summary CSV's whole-population retrieval average must not be described as a successful-only retrieval estimate. Failure elapsed time is not “generator latency.”

## Judge score audit

Judge: OmniRoute `codex/gpt-6-luna-low` / canonical `gpt-6-luna-low`; temperature 0, reasoning effort low, single fixed system rubric. It scored answer relevance, Semantic Accuracy, Faithfulness and Citation Fidelity on [0,1]. The prompt asks for each case independently; it defines Semantic Accuracy against the reference, Faithfulness against provided context, and Citation Fidelity as valid citations divided by cited sources (no citation where context exists scores 0). Inputs include question, answerability label, reference, actual evaluator-visible context and cleaned output. Arm mapping was kept separate from judge inputs.

The 4,265 eligible answer slots collapse to 2,297 unique exact payloads; 1,968 slots reused scores; largest multiplicity 9. Each case hash is recomputed from all five case-specific score-affecting fields. Its digest equals the `case_<first-24-hex>` case ID; duplicate groups have no score-tuple conflict. Raw output SHA-256 agrees across generation result, export, response mapping and score row. No missing eligible score or scored failure was found. The global judge rubric/script SHA-256 is `920702b80a4bbfa2680e97b277ee279b81a2ac528a3b73cbda0de192b9f8641a`; global model/decode details are held in script and manifest, not repeated inside the per-case digest. No cross-run cache was used. **Caveat:** no independent per-batch remote request replay was performed; judge reliability/validity is not established by cache identity, and a single LLM judge is not human ground truth.

## Reproducibility

Run `.venv/bin/python revision_ablation_defense/analyze_existing.py` to regenerate tables and `integrity_audit.json` from the preserved artifacts. The run package `SHA256SUMS` lists 14,891 files; all listed files were independently present and checksum-matched. Reproducible bootstrap analyses use 10,000 query-level resamples and deterministic seeds recorded in the CSVs. No answer generation or judge API call is part of this script.
