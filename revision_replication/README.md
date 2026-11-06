# Local Gemma paired replication (execution gated)

This package prepares a **new** independent-generator comparison using the supplied
Gemma 3 text 4B 4-bit MLX model. It is not a replay of historical MLX or CUDA
outputs. Paper baselines map explicitly to code configurations **B3 → B4** and
**B5 → B6**; the CUDA `B6_full` arm is not presumed equivalent to historical B5.

## Safety and provenance

- Metadata inspection reads config/tokenizer/index metadata and hashes files; it
  does not deserialize model tensors or execute inference.
- `plan` never constructs a model backend and never runs a forward pass.
- `run` defaults to plan-only behavior. Real local generation additionally
  requires `--authorize-real-generation`; do not use that flag before scientific
  approval and the local MLX runtime compatibility check.
- There is deliberately no real judge implementation. The paired judge protocol
  validates same-query, same-reference, same-evidence inputs and uses an injected
  fake in tests; its real-call boundary raises unconditionally.
- Context is never silently truncated. Overflow is a per-operation failure.
- Cohort selection is deterministic and should be frozen before confirmatory
  execution. A 5-pair pilot tests engineering compatibility, not generalizability.
- Historical MLX, archived CUDA, and new Gemma results must remain separate.

## Isolated optional MLX environment

The active application environment need not be changed. After approval, create a
dedicated virtual environment and install the pinned packages from
`back/benchmarks/rag/env/requirements-mlx-replication.txt`. This task did not
install them because the current environment lacks `mlx` and `mlx-lm`.

```sh
python3 -m venv /tmp/gemma-replication-venv
/tmp/gemma-replication-venv/bin/pip install -r back/benchmarks/rag/env/requirements-mlx-replication.txt
```

Model compatibility must be explicitly checked before running: the metadata
manifest at `revision_replication/gemma_model_compatibility.json` verifies the
local checkpoint/tokenizer files, but states that the pinned runtime has not yet
loaded the checkpoint. No model weights were loaded as part of infrastructure
work.

## Reproducible commands

From the repository root:

### 1. Metadata-only model check (no weights loaded)

```sh
.venv/bin/python -m revision_replication.cli inspect-gemma \
  --model-dir /Volumes/happy-disk/models/llm/gemma-3-text-4b-it-4bit
```

### 2. Saved CUDA result analysis (read-only archive inputs)

```sh
.venv/bin/python -m ablation_cuda.recovery_analysis \
  --archive graphs/run_20261020T065029857103Z_OCC-RAG-1.7B_CUDA_ABLATION \
  --output /tmp/cuda_availability_analysis.json \
  --export-registry /tmp/cuda_operational_input_registry.jsonl \
  --bootstrap-resamples 20000 --seed 20261106
.venv/bin/python -m ablation_cuda.export_recovery \
  /tmp/cuda_availability_analysis.json /tmp/cuda_publication_tables
```

The analysis uses all 300 answerable IDs for availability contrasts. Missing
judge outputs remain missing, not Semantic Accuracy zeroes. An operational
composite is specified in the JSON but remains pending judge scores.

### 3. Plan-only deterministic cohort (no model loading or calls)

`rows.jsonl` must contain one row per query for each of `code_baseline` B4 and B6
with the complete, already-frozen `prompt`, evidence IDs, and evidence content.
The provenance JSON must satisfy `ReplicationProvenance` and identify dataset,
corpus, retrieval, context origin, generator/tokenizer revision, quantization,
prompt template, generation and evaluation settings.

```sh
.venv/bin/python -m revision_replication.cli plan \
  --rows /path/to/frozen_b3_b5_rows.jsonl \
  --provenance /path/to/frozen_provenance.json \
  --output-dir /tmp/gemma-plan-5 \
  --run-id gemma-independent-pilot-v1 \
  --cohort-ids /path/to/frozen_5_query_ids.json \
  --context-limit 12000 --max-tokens 128
```

### 4. Future real generation (NOT run in infrastructure work)

Only after the model/runtime smoke test, cohort prespecification, and explicit
scientific authorization, an approved five-pair local pilot would use the same
command with `run` and the opt-in flag:

```sh
.venv/bin/python -m revision_replication.cli run \
  --rows /path/to/frozen_b3_b5_rows.jsonl \
  --provenance /path/to/frozen_provenance.json \
  --output-dir /path/to/new_immutable_run_dir \
  --run-id gemma-independent-pilot-v1 \
  --cohort-ids /path/to/frozen_5_query_ids.json \
  --context-limit 12000 --max-tokens 128 \
  --authorize-real-generation
```

Do not execute this example without separate approval. A confirmatory cohort
must be justified and frozen before generation; it is not automatically “all
remaining rows” or an arbitrary convenience sample.

### 5. Corrected judge evaluation

No command for actual judge evaluation is provided: external/network judge calls
are intentionally hard-disabled. Before a separate approved judging
implementation, build and review the per-query eligibility registry and paired
call/token budget. Only `exact_archived` or `verified_reconstructed`
generator-visible evidence can be eligible; top retrieved chunks alone do not
prove the generator's complete input. The current saved historical registry has
375 paired IDs, but zero meet that evidence criterion.

## Result artifacts

`operations.jsonl` is append-only per-operation checkpoint state, keyed by run,
query, and code baseline, with input/evidence hashes and structured statuses.
`results_manifest.json` is immutable once the planned pairs have records. Do not
edit or overwrite prior runs. Reuse statistics and publication code only after
checking the run kind and provenance; never merge historical MLX, CUDA, and Gemma
results as one experiment.
