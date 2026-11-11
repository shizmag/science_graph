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

## Prepared-input execution

Run the CPU-only readiness check first, using a new runtime namespace (the
namespace must already have been created by `freeze-runtime`):

```bash
python3 -m ablation_mlx.r1_corrected.cli \
  --runtime-artifact-root RUNTIME_NAMESPACE preflight --require-artifacts
```

Materialize the immutable prompts and token IDs without loading MLX weights:

```bash
python3 -m ablation_mlx.r1_corrected.cli \
  --runtime-artifact-root RUNTIME_NAMESPACE prepare-contexts \
  --manifest-hash FROZEN_RUNTIME_MANIFEST_HASH
```

Start sequential generation from those artifacts.  The command validates all
artifacts before constructing the adapter, then keeps one adapter/model
resident and checkpoints each accepted slot:

```bash
python3 -m ablation_mlx.r1_corrected.cli \
  --runtime-artifact-root RUNTIME_NAMESPACE generate-prepared \
  --manifest-hash FROZEN_RUNTIME_MANIFEST_HASH
```

Interrupt with Ctrl-C at any time.  Accepted rows remain durable; rerun the
same command to resume.  A completed rerun performs no inference and does not
load the model.  To inspect progress without loading MLX:

```bash
python3 -m ablation_mlx.r1_corrected.cli \
  --runtime-artifact-root RUNTIME_NAMESPACE status \
  --manifest-hash FROZEN_RUNTIME_MANIFEST_HASH
```

Validate or export accepted results with `validate` and `export` after the
run.  The checkpoint directory is the runtime namespace itself and contains
the SQLite ledger plus atomic accepted-result records.  Every row is bound to
the frozen runtime manifest hash, cohort hash, prepared-input hash, prompt
hash, and token-ID hash; do not copy artifacts between namespaces.

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
- Static evidence means the frozen graph relation types listed in
  `evidence.py`; `HAS_TAG` is excluded because it is a metadata/tagging
  relation, not a directed graph fact.  Retrieved chunks, dynamic expansion,
  and all other non-target evidence remain byte-identical across the two arms.

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
NEW_RUNTIME=/Volumes/happy-disk/codex-worktrees/r1-mlx-structured-context/revision_round1/experiments/r1_mlx_corrected_confirmatory_v1
MANIFEST=019a4ef450ac08e3e0902db968eeb5d4c44db6e8a20c0f60d4986c5c00249458
CENSUS=/Volumes/happy-disk/codex-worktrees/r1-mlx-structured-context/revision_round1/derived/r1_mlx_prompt_census_2026-11-10.json
```

CPU-only diagnostics and frozen-input inspection:

```bash
$PY -m ablation_mlx.r1_corrected --model "$MODEL" --dataset "$DATASET" --database "$DB" doctor
$PY -m ablation_mlx.r1_corrected --model "$MODEL" --dataset "$DATASET" --database "$DB" preflight
$PY -m ablation_mlx.r1_corrected --model "$MODEL" --dataset "$DATASET" --database "$DB" --artifact-root "$SOURCE" activation
$PY -m ablation_mlx.r1_corrected --model "$MODEL" --dataset "$DATASET" --database "$DB" census --tokenizer "$MODEL/tokenizer.json" --output "$CENSUS"
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
  --artifact-root "$RUNTIME" --manifest-hash "$MANIFEST" export \
  --output "$RUNTIME/future_judging_export.jsonl"
```

`validate` is the guarded offline validator.  It rechecks current runtime and
frozen-input identity, immutable evidence, cohort hashes, accepted-result
integrity, legacy entropy labels, and predictive telemetry protocol labels.
`export` is accepted-ledger-only:
missing generations remain `MISSING`, and no quality score is fabricated.

The smoke command used for the authorized bounded test was:

```bash
$PY -m ablation_mlx.r1_corrected \
  --model "$MODEL" --dataset "$DATASET" --database "$DB" \
  --artifact-root "$SOURCE" --runtime-artifact-root "$RUNTIME" \
  smoke --manifest-hash "$MANIFEST" --max-new-generations 12 --stop-after 2

# Read-only status, validation and compatible retry/resume form (do not run
# against the exhausted smoke without a separately authorized budget change).
$PY -m ablation_mlx.r1_corrected \
  --model "$MODEL" --dataset "$DATASET" --database "$DB" \
  --artifact-root "$RUNTIME" --manifest-hash "$MANIFEST" status
$PY -m ablation_mlx.r1_corrected \
  --model "$MODEL" --dataset "$DATASET" --database "$DB" \
  --artifact-root "$SOURCE" --runtime-artifact-root "$RUNTIME" \
  --manifest-hash "$MANIFEST" validate
$PY -m ablation_mlx.r1_corrected \
  --model "$MODEL" --dataset "$DATASET" --database "$DB" \
  --artifact-root "$SOURCE" --runtime-artifact-root "$RUNTIME" \
  smoke --manifest-hash "$MANIFEST" --max-new-generations 2 --stop-after 2
```

The same command form was used for restart/resume segments.  Do not run it
again: the cumulative budget is exhausted at 12 marked calls.

The future full-run command, for a separately authorized new compatible runtime
namespace, is documented here but was **not executed**:

```bash
$PY -m ablation_mlx.r1_corrected \
  --model "$MODEL" --dataset "$DATASET" --database "$DB" \
  --artifact-root "$SOURCE" --runtime-artifact-root "$NEW_RUNTIME" \
  run --manifest-hash "$NEW_MANIFEST" --max-new-generations 0
```

`NEW_MANIFEST` must be created by a separate freeze in the empty
`$NEW_RUNTIME` namespace using the source manifest. `run` with
`--max-new-generations 0` means no local per-invocation cap; it is not
authorization to launch the full cohort. The N recommendation and uncertainty
limits appear below.

## CPU-only full-cohort prompt census

The exact frozen tokenizer and read-only evidence database were applied to all
375 frozen query IDs without loading the model or running generation. The
report is a derived preflight artifact, not an execution ledger:

```text
revision_round1/derived/r1_mlx_prompt_census_2026-11-10.json
File SHA-256: `9d785cf75539b311bfd09a31a654199d9d67740c9c3a478a8d5e8df24a6f641d`.
Embedded canonical content hash (`report_sha256`):
`b38a9d228b741dcaf297bff88431605a9c894fbdc057d23ccfac180c0b5163d0`.
These differ by design: the file hash covers serialized JSON, while
`report_sha256` hashes the canonical report payload.
```

Counts are 375 total (300 answerable, 75 designated negatives), with 347
activation-positive and 28 activation-negative. Among answerable queries,
278/300 (92.67%) activate the targeted static relation component; therefore
the effective activated answerable sample is about 92.67% of the paired N
before generation failures or quality exclusions. No prompt exceeds the
unchanged 12,000-token cap. All 375 historical October overflow values are
**unknown**, because that status is absent from the frozen query metadata; they
are not treated as zero or false.

Token counts (empirical percentiles use nearest-rank):

| Subset / measure | n | min | max | mean | p50 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All ON prompts | 375 | 367 | 1,802 | 822.90 | 809 | 1,077 | 1,157 | 1,446 |
| All OFF prompts | 375 | 367 | 838 | 604.11 | 604 | 693 | 721 | 781 |
| Static-context token contribution (ON−OFF) | 375 | 0 | 1,161 | 218.79 | 215 | 421 | 481 | 734 |

The maximum smoke prompt length was 1,034 tokens; that is below the census
ON p95 (1,157) and maximum (1,802), so the smoke was not representative of the
longest/most expensive ON contexts. The longest cases were omitted from the
smoke selection. The ON/OFF overflow difference is retained as an operational
endpoint for future runs even though both arms have zero expected overflow in
this census.

Reproduce the CPU-only census to a **new** output path (the command refuses
existing files and both protected runtime roots):

```bash
$PY -m ablation_mlx.r1_corrected \
  --model "$MODEL" --dataset "$DATASET" --database "$DB" \
  census --tokenizer "$MODEL/tokenizer.json" \
  --output "$ROOT/revision_round1/derived/r1_mlx_prompt_census_NEW.json"
```

## Predictive-distribution telemetry

New generations use telemetry protocol
`r1-mlx-predictive-distribution-v1`, implementation
`mlx_lm.generate_step@0.31.3`, source
`mlx_lm.generate_step.full_vocabulary_logprobs`. MLX-LM v0.31.3
`generate_step()` scores the prompt's next-token distribution from
`logits[:, -1, :]`, normalizes it over the full vocabulary, then yields the
greedy token and full-vocabulary log probabilities before advancing. The
adapter passes no logits processors or temperature transform and reduces each
distribution immediately; it does not retain response-wide dense vectors.
The first emitted token is scored against the prompt-conditioned next-token
distribution (no one-token offset). EOS/stop distributions are retained as a
compact telemetry row and included in position means, while stop IDs are
omitted from answer text/output token IDs, matching the high-level stream's
response behavior.

H20/H50/H100 preserve the historical CUDA definitions: for each generated
position, select the largest K probabilities, renormalize within that top-K,
and compute Shannon entropy in bits; report the arithmetic mean of that local
entropy across compact generated-token rows. `retained_mass_kK` instead sums
those top-K probabilities under the original full-vocabulary denominator.
Selected-token logprob/NLL, top-1 probability, logit margin, local top-1 and
retained mass use stable max-shift/log-sum-exp reductions and `math.fsum`.
No dense distribution is persisted. The exact implementation identifier,
protocol, source, K, units, aggregation and EOS policy are bound into new
runtime manifests; validation rejects incompatible or mixed populated
telemetry protocols.

The original exhausted smoke is **not upgraded** by this implementation. Its
10 accepted outputs were generated before predictive telemetry was exposed;
no entropy values are backfilled. The two accepted records with stale
`full_predictive_distribution` source labels remain identifiable as invalid,
and all old accepted rows remain byte-identical. The new telemetry protocol
applies only to future explicitly authorized generations.

## Cohort-size recommendation and uncertainty

The confirmatory population is the 300 answerable queries, not all 375 census
rows. At the observed 278/300 activation prevalence, N=100, 150, and 300
answerable pairs imply approximately 93, 139, and 278 activated pairs before
generation failures. For a paired binary treatment-difference endpoint, each
pairwise difference lies in `[-1, 1]`; a conservative worst-case 95% CI
half-width is therefore `1.96/sqrt(n_eff)`, approximately ±20.3, ±16.6, and
±11.8 percentage points for those effective activated Ns. Pair covariance can
improve precision, but it cannot be estimated credibly from the five completed
smoke pairs. These are precision illustrations, not a power calculation;
missingness and query dependence can reduce effective sample size. Recommend
the full 300 answerable pairs if compute permits; 100/150 should be explicitly
limited milestones, not substituted silently for the frozen cohort. October
CUDA results remain a separate execution and must not be pooled with MLX.

The existing smoke budget is fixed at 12 cumulative marked calls and is
exhausted (12 started, 10 accepted, 2 retryable failures, 5 complete pairs,
one query with neither arm accepted). Resumption skips accepted outputs and
retries missing/retryable operations, but there is no budget-amendment command.
Do not raise or reset this smoke cap. A future extension requires an explicit,
auditable, manifest-bound amendment preserving the append-only attempt counter;
that mechanism is not implemented. Until authorized and implemented, use a new
compatible runtime namespace for additional inference.

## Post-authorization parity/validation plan (not run)

No real inference was run during this readiness work. Once separately
authorized and with a fresh explicitly amended/new manifest, use one same
frozen prompt per arm to compare legacy high-level greedy and telemetry-enabled
generation: exact output token IDs, decoded text, stop/EOS behavior, finish
reason, and per-token telemetry alignment. Minimal parity budget is two calls
(same prompt, one per path). Ledger inspection showed the two retryable
failures were earlier ON/OFF attempts for `pubmedqa_15703931`, later
successfully retried; they do not overlap the missing-arm query. Allow two calls
to complete `pubmedqa_unans_16962519` and two calls for one long-context
ON/OFF pair: six generation calls total beyond the exhausted smoke counter.
Real predictive entropy validation is included in the telemetry-enabled parity
calls. Do not execute these commands before authorization.

Example commands after authorization (paths/manifests must be newly frozen and
compatible; the existing exhausted `$RUNTIME` manifest is not amended here):

```bash
# Freeze a separately authorized compatible namespace first, then:
$PY -m ablation_mlx.r1_corrected --model "$MODEL" --dataset "$DATASET" \
  --database "$DB" --artifact-root "$SOURCE" \
  --runtime-artifact-root "$NEW_RUNTIME" \
  run --manifest-hash "$NEW_MANIFEST" --max-new-generations 2
$PY -m ablation_mlx.r1_corrected --model "$MODEL" --dataset "$DATASET" \
  --database "$DB" --artifact-root "$SOURCE" \
  --runtime-artifact-root "$NEW_RUNTIME" \
  validate --manifest-hash "$NEW_MANIFEST"
```

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
- 2 earlier `FAILED_RETRYABLE` attempts for `pubmedqa_15703931` ON/OFF; both
  were later retried successfully and do not represent missing results;
- zero accepted records for `pubmedqa_unans_16962519` because the conservative
  budget was exhausted;
- no attempt was deleted or silently retried after budget exhaustion.

The two retryable attempts retained in the append-only ledger were for
`pubmedqa_15703931` (ON and OFF). Both were later successfully retried as
separate attempts; the failures were caused by passing unsupported `temp=` to
the installed `generate_step()` interface. A separate tokenizer compatibility
failure occurred before attempt start and consumed zero calls.

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

## Measured smoke workload estimates

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
- the selected smoke prompts were shorter than the full-cohort ON p95 (1,157)
  and maximum (1,802) prompt lengths; therefore the smoke omitted some longer
  cases even though the full CPU census found no current 12,000-token overflow;
- historical October overflow status is unknown for all 375 rows because the
  frozen query metadata contains no recognized historical-overflow field.

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
