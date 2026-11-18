# Phase A — provenance validation

**PASS: 160/160 historical answer–evidence records; zero exclusions.**
80 fixed answerable query IDs, 80 B3/code-B4 records and 80 B5/code-B6 records.
All retained generations have archived status `success`; zero empty cleaned outputs.

## Source and archive
`historical_evidence_recovery.load_historical_records` verifies the archived manifest's
recorded Git commit is `b4f3322d8cae3395bcbf0b5a1f2e47a904d1591a`, context-file checksum
matches the archived dataset hash, expected run size is 375, code configurations B4/B6
exist, and B6 inherits enabled graph expansion from the snapshot. Historical source
blobs are hash-recorded by its source contract. This establishes the archive's recorded
source identity, not independent proof of the exact runtime environment.

Raw outputs were read directly from `evaluation_results.yaml` (not regenerated and
not replaced by cleaned answers). Per-query configuration keys and questions were
checked. Cleaned outputs are the persisted baseline evaluation answers used by the
hash-verified R2 packet. Evidence and cleaned answers match that packet exactly.

## Source-to-prompt mapping
- B3/code B4: **80 ask_no_expander**, archived `trimmed_text` + `trimmed_graph` only.
- B5/code B6: **80 ask_expander**, archived `enrichment_block` only.
- No fixed80 B5 fallback occurs; empty/sentinel fallback is covered with local fixtures.
- Historical `core/pipelined.py` passes the **raw query**, without the query wrapper
  found in a separate helper. Templates are read from Git at the recorded commit.
- Reconstruction matches **100/100 corrected replay prompt fixtures**, including
  stored prompt text and SHA-256. Each input is rendered three times identically.
- Every original selected evidence field is checked as present in its chosen prompt;
  no intermediate retrieved chunks are appended. No gold is an argument to prompt
  rendering, nor included in the judge-visible whitelist.
- UTF-8 string hashes and canonical object hashes are recomputed in the tests;
  manifests are deterministic newline-terminated canonical JSONL.
- Both judges receive identical visible payloads for each historical response. Internal
  paper/code labels and reference answers stay outside actual judge messages.

The historical evaluator's `build_context_string(retrieved_chunks)` formats only
retrieved chunk text with newly enumerated Block labels. It does not insert the
archived final enrichment block or full post-trim graph payload. Consequently that
judge context was not guaranteed to equal the generator-visible evidence, and its
numbering cannot be imported into B5 enrichment citation resolution.

## Boundaries
Status on each record:
`archived_evidence_verified / historical_model_input_unverified`.
The reconstructed prompt is a reproducible historical-template reconstruction.
The original complete serialized tokenizer input, tokenizer runtime and exact
historical corpus/index fingerprint are NOT established. Corrected replay output
matches, even when present, do not establish unknown historical input bytes.
This protocol performs no current retrieval and no new B3/B5 generation.

## Selection and security
The fixed cohort is reproduced from the original eligible-pair strata with seed
20261106, preserving selection conditional on paired answer/evidence availability.
It is not advertised as a fresh independent sample of all answerable queries.
Source references in new manifests are repository-relative; original config values
are not copied (their source hashes are retained). New-manifest credential/private-path
screening is recorded separately. Internal archives are untouched; public distribution
still requires manual screening and licensing review.
