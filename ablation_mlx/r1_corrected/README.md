# Corrected R1 structured-context ablation (local MLX)

This directory contains the separately fingerprinted MLX implementation for
the corrected R1 static-graph structured-context ablation.  It is not a
continuation of the October CUDA operation ledger.

## Scientific and provenance boundary

The October CUDA archive remains immutable historical evidence.  Its
`B6_minus_static_graph_context` arm changed configuration but did not change
the generator-visible prompt: the ask/stream expansion templates omitted the
retrieved text and static graph sections.  The production templates now render
retrieved text, static graph relations, and dynamic enrichment.  The new MLX
experiment tests **static graph relations only**.  Retrieved chunks, question,
dynamic graph enrichment, ontology enrichment, formatting, retrieval limits,
budget policy, model, tokenizer, and greedy decoding are held constant.

The corrected MLX results must not be merged into the October CUDA ledger or
described as a retroactive repair of that study.  The July MLX experiment is a
different experiment and is not used as the source model or database here.

Frozen inputs:

- Model: `occ-ai/OCC-RAG-1.7B`, archived revision
  `e7d9f8b912dc1ac9a66ab42f0d07a7663d6ed58c`.
- Local model weight SHA-256:
  `913448d005d2af695a738da177120c9e741dc8a5b2328f717d7ed108c1fd2e4f0`.
  The local weight matches the archived lock.
- The local config SHA-256 is
  `b8c3a6571d676cc6cfc075c9e4ab21b772c42de95ea2eb71fe401c523454e750`,
  while the archived config lock is
  `cbfc589e89ed20d820ab698ca3b56f8d5914710f5b4019cb240eeb882376c9f0`.
  The local directory has no revision file.  Therefore this handoff does not
  claim byte-identical full model provenance.
- Dataset: 375 records (300 answerable and 75 designated negatives), SHA-256
  `d2820ced7c2994218245887ad753700923262fb91ec983eeacc01b217ec43b6b`.
- Database snapshot: `external_db-20261016T193200Z`, SHA-256
  `c7fdd7c6fed69b788563896efaa8cbfe204216d6ff82043ff3a7414267aba8f0`.
  The payload is read-only SQLite opened with `mode=ro&immutable=1`.
- The static relation policy excludes `HAS_TAG`.  CPU census: 347/375
  activation-positive and 28/375 activation-negative records.

The immutable preparation namespace is:

```text
/Volumes/happy-disk/codex-worktrees/r1-mlx-structured-context/revision_round1/experiments/r1_mlx_corrected
```

Its manifest and cohort hashes are respectively
`2da88f58b85926a483cb45dbbac1216e066bedc7b1a273641284fd49f32333a8` and
`84288bcb864fc8f961c251b22c1a955114b27690678e326003060e230284fa5a8`.
Its MLX package fields are null by design; it is a pre-runtime preparation
namespace and must not be used for inference.

## Runtime identity and durable namespace

The pinned runtime is:

- arm64 CPython `3.12.13`;
- `mlx==0.31.2` and `mlx-lm==0.31.3`;
- Darwin arm64 hardware, with the complete runtime identity recorded in the
  execution manifest.

The environment is installed in the external venv:

```text
/Volumes/happy-disk/dev/uv/venvs/r1-mlx-structured-context
```

The frozen execution namespace is:

```text
/Volumes/happy-disk/codex-worktrees/r1-mlx-structured-context/revision_round1/experiments/r1_mlx_corrected_runtime_2026-11-10
```

Its immutable execution manifest hash is:

```text
019a4ef450ac08e3e0902db968eeb5d4c44db6e8a20c0f60d4986c5c00249458
```

The runtime namespace is separate from the preparation namespace.  The
`freeze-runtime` command refuses a non-empty target and copies the already
prepared cohort rather than selecting examples again.  The runtime manifest
binds the model, tokenizer, dataset, database, protocol, prompt version,
runtime, decoding configuration, source manifest/cohort hashes, and smoke
budget.  No historical context/prompt/generation cache entries are imported.

The existing CUDA `CheckpointStore` atomic result-file primitive is reused
under this fresh namespace.  `DurableResultStore` adds a manifest-bound
WAL/FULL SQLite attempts ledger, immutable checksum-verified accepted rows,
append-only attempt history, terminal outcomes, a process lock, and restart
reconciliation.  A generation is conservatively counted immediately before
the adapter call (`generation_started=1`), so a crash in that narrow interval
cannot allow a restart to exceed the smoke budget.  This does not claim
exactly-once hardware computation: an interrupted call can remain an
identifiable retryable attempt, while a committed accepted result is never
silently overwritten.

## Actual command interface

Use the venv interpreter and put global options before the subcommand.  The
database path below is the verified external immutable payload; it is not
copied into the repository.

```bash
PY=/Volumes/happy-disk/dev/uv/venvs/r1-mlx-structured-context/bin/python
MODEL=/Volumes/happy-disk/models/llm/OCC-RAG-1.7B
DATASET=/Volumes/happy-disk/codex-worktrees/r1-mlx-structured-context/ablation_portability/dataset
DB=/Users/vladimirkasterin/python/graph/ablation_portability/snapshots/candidates/external_db/v1/database.db
SOURCE=/Volumes/happy-disk/codex-worktrees/r1-mlx-structured-context/revision_round1/experiments/r1_mlx_corrected
RUNTIME=/Volumes/happy-disk/codex-worktrees/r1-mlx-structured-context/revision_round1/experiments/r1_mlx_corrected_runtime_2026-11-10
MANIFEST=019a4ef450ac08e3e0902db968eeb5d4c44db6e8a20c0f60d4986c5c00249458
```

CPU-only diagnostics and frozen-input inspection:

```bash
$PY -m ablation_mlx.r1_corrected --model "$MODEL" --dataset "$DATASET" --database "$DB" doctor
$PY -m ablation_mlx.r1_corrected --model "$MODEL" --dataset "$DATASET" --database "$DB" preflight
$PY -m ablation_mlx.r1_corrected --model "$MODEL" --dataset "$DATASET" --database "$DB" --artifact-root "$SOURCE" activation
```

The metadata-only runtime freeze command (already executed for the namespace
above; do not rerun it against that non-empty target) is:

```bash
$PY -m ablation_mlx.r1_corrected \
  --model "$MODEL" --dataset "$DATASET" --database "$DB" \
  --artifact-root "$SOURCE" --runtime-artifact-root "$RUNTIME" \
  freeze-runtime --source-manifest-hash 2da88f58b85926a483cb45dbbac1216e066bedc7b1a273641284fd49f32333a8
```

Guarded runtime inspection and export:

```bash
$PY -m ablation_mlx.r1_corrected \
  --model "$MODEL" --dataset "$DATASET" --database "$DB" \
  --artifact-root "$RUNTIME" \
  --manifest-hash "$MANIFEST" status

$PY -m ablation_mlx.r1_corrected \
  --model "$MODEL" --dataset "$DATASET" --database "$DB" \
  --artifact-root "$SOURCE" --runtime-artifact-root "$RUNTIME" \
  --manifest-hash "$MANIFEST" validate

$PY -m ablation_mlx.r1_corrected \
  --model "$MODEL" --dataset "$DATASET" --database "$DB" \
  --artifact-root "$RUNTIME" export
```

`validate` is the guarded offline validator.  It rechecks current runtime and
frozen-input identity, immutable evidence, cohort hashes, accepted-result
integrity, and entropy telemetry labels.  `export` is accepted-ledger-only:
missing generations remain `MISSING`, and no quality score is fabricated.

The smoke command used for the authorized bounded test was:

```bash
$PY -m ablation_mlx.r1_corrected \
  --model "$MODEL" --dataset "$DATASET" --database "$DB" \
  --artifact-root "$SOURCE" --runtime-artifact-root "$RUNTIME" \
  smoke --manifest-hash "$MANIFEST" --max-new-generations 12 --stop-after 2
```

The same command form was used for restart/resume segments.  Do not run it
again: the cumulative budget is exhausted at 12 marked calls.

The future full-run command is documented here but was **not executed**:

```bash
$PY -m ablation_mlx.r1_corrected \
  --model "$MODEL" --dataset "$DATASET" --database "$DB" \
  --artifact-root "$SOURCE" --runtime-artifact-root "$RUNTIME" \
  run --manifest-hash "$MANIFEST" --max-new-generations 0
```

`run` with `--max-new-generations 0` means no local per-invocation cap; it is
not authorization to launch the full cohort.  The full cohort size and
confirmatory analysis remain to be scientifically finalized.

## Smoke result and recovery proof

The deterministic smoke IDs were frozen before generation:

```text
pubmedqa_15703931
pubmedqa_18041059
pubmedqa_18926458
pubmedqa_27040842
pubmedqa_9199905
pubmedqa_unans_16962519
```

The run consumed exactly 12 marked model-call attempts, never more:

- 10 accepted `SUCCESS` records: five ON/OFF pairs for the first five IDs
  above;
- 2 `FAILED_RETRYABLE` records from early MLX API compatibility failures;
- zero accepted records for `pubmedqa_unans_16962519` because the conservative
  budget was exhausted;
- no attempt was deleted or silently retried after budget exhaustion.

The first two actual retryable failures were retained in the append-only
ledger.  The first failed command, before an attempt was started, exposed the
MLX tokenizer incompatibility and consumed zero calls.  The runner was then
updated to support `TokenizerWrapper.encode(...)`; the MLX adapter was updated
to use `make_sampler(temp=0.0)` rather than the unsupported downstream
`temp=` keyword.  These compatibility fixes are part of the implementation,
not post-hoc changes to accepted outputs.

Recovery behavior was exercised explicitly:

1. A stop-after segment committed two paired successes.
2. A restarted process skipped those accepted operations and committed the
   next paired successes.
3. A final invocation consumed the remaining calls and stopped at 12.
4. A third invocation reported `adapter_loaded:false`,
   `generation_calls_total:12`, and made no new attempt or model call.

The final ledger status was `accepted=10`, `attempts=12`,
`generation_calls=12`, `success=10`, `failed_retryable=2`.

Generator-visible intervention proof held offline: the four activation-positive
smoke queries had different ON/OFF prompt hashes, while the activation-negative
`pubmedqa_27040842` had identical prompts legitimately. All accepted records
preserve prompt, evidence, input-token, model, tokenizer, decoding, and
runtime identities.

## Measured smoke telemetry and workload estimates

These are measurements of the five paired queries that produced accepted
outputs, not a scientific quality result:

- accepted input tokens: 7,389 total, mean 738.9, minimum 564, maximum 1,034;
- accepted output tokens: 2,527 total, mean 252.7;
- generation duration: 106.357 seconds total, mean 10.636 seconds, minimum
  6.311 seconds, maximum 14.085 seconds;
- measured output-token / generation-duration aggregate: approximately 23.8
  tokens/second.  This is a generate-call measure, not a pure decode-only
  benchmark because the call includes prompt processing;
- observed process maximum RSS across successful segments:
  1,198,047,232 bytes; observed MLX peak footprint was 4,488,447,336 bytes;
- three restart segments took 169.51, 176.87, and 205.03 seconds wall time.
  Those wall times include repeated process/model-load overhead and must not
  be treated as steady-state one-resident-model throughput;
- prefill-only performance was not separately instrumented and is not
  invented here;
- the selected smoke prompts were far below the 12,000-token input limit
  (maximum accepted input was 1,034 tokens), and no historical overflow case
  was present in the frozen preparation metadata.

Using only the measured 10.636-second generate-call mean, a rough decode-call
extrapolation is:

| Planned paired queries | Approx. calls | Generate-call time only |
| ---: | ---: | ---: |
| 100 | 200 | 2,127 s (35.5 min) |
| 150 | 300 | 3,191 s (53.2 min) |
| 300 | 600 | 6,381 s (106.4 min) |

These are not launch estimates: they exclude model load, prompt-length
variation, input overflows, failures, filesystem overhead, and thermal/runtime
effects. A rough raw-result-size extrapolation from the observed 14,281-byte
mean accepted result file is about 2.9 MB for 100 fully successful pairs and
8.6 MB for 300 pairs, before SQLite, attempt, outcome, and export overhead.
Use `du` and the status/validate commands during any future run rather than
relying on these small-sample extrapolations. The smoke was not a benchmark
and no full cohort was launched.

Entropy telemetry is **unavailable and invalid** for this adapter. Direct
`mlx_lm.generate` does not expose the full predictive distribution required for
H20/H50/H100, so selected-token information is never substituted. One initial
accepted ON/OFF pair was written before the telemetry-label correction and has
`entropy_valid=false` with the stale source label
`full_predictive_distribution`; the validator reports both records as
warnings. Accepted artifacts are intentionally not rewritten. All H20/H50/H100
values must be treated as missing in any future analysis.

The external future-judge export contains 750 query/arm records with
`SUCCESS=10` and `MISSING=740`; judging was not run. Missing generations must
remain missing and must not be converted to lexical or zero-quality scores.

## Future analysis and disclosure

The intention-to-treat denominator is all selected query pairs. Activation-
positive pairs are a predefined mechanistic subgroup, and successfully
generated pairs are a separate conditional-quality subgroup. Future paired
analysis should report activation, prompt-length change, availability,
overflow, and quality denominators separately. No lexical quality heuristic
was used here, and no external LLM judge was called.

This corrected MLX study changes runtime properties from the historical CUDA
study: MLX rather than CUDA/PyTorch, a separately frozen local environment,
and a local model directory whose config hash differs from the archived config
lock despite matching weights. Those changes are why its results require a
new manifest and cannot be silently pooled with October CUDA outputs.
