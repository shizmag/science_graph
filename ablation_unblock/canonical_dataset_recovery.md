# Canonical July dataset recovery

**Outcome: `EXACT_SOURCE_RECOVERED`.** The canonical source was present; it was
missed because the portability workflow searched the wrong checkout-relative
directory.

## Why the workflows disagreed

The July run is at
`graphs/run_20260729_211324_OCC-RAG-1.7B/` at the project root. Its
`retrieved_contexts.yaml` is present there. The portability freezer had looked
under `back/benchmarks/rag/graphs/run_20260729_211324_OCC-RAG-1.7B/`, which is a
different, absent path. The revision workflow used the project-root `graphs/`
directory. Both generated-output locations are git-ignored, which obscured the
top-level artifact during the earlier scoped search. No archive restore, new
sampling, or unrelated October data was needed.

The canonical source SHA-256 is
`738b9d45c5049f49c6944807ac829846c42b4955f052c8af42439312ce7612cc`.
Historical `metrics_details.csv` contains 1,875 records (375 query IDs × five
baselines) and independently confirms July-run membership. Its SHA-256 is
`a2010645d14f50826bc34491fa0efb9d451096640327c548bdc896f764db4e5e`.

## Field provenance

| Frozen field | Source and validation |
|---|---|
| `row_index` | Original array position in the July `retrieved_contexts.yaml`; preserves its order. |
| `query_id`, `question` | July source fields `id` and `query`; exact ID and text crosschecked against the original golden dataset and historical membership. |
| `answerability_label` | July `is_answerable`; exact label crosschecked against the golden dataset and historical metrics. |
| `gold_reference_answer` | July `golden_answer`; crosschecked against the golden dataset. For positive PubMedQA records, the archived rendering also matches the original sampled record's `Final_decision` plus `long_answer`. Negative values retain the archived string; archived null is represented as an empty value, not a generated answer. |
| `source_pmid`, `original_paper_ids` | July `expected_papers`; exact paper ID crosschecked against the golden file and numeric PMID against `sampled_300.json`. |
| `synthetic_negative_category` | July `category`, crosschecked for designated negatives against `revision_release/dataset_provenance/synthetic_negative_trace.csv`. Answerable rows have no negative category. |

The original golden dataset SHA-256 is
`7091a67d0a8edd2a971ccb65e26f0bf33f26a06c5453005ec893f6791809b85f`;
`sampled_300.json` is
`ee1dd17b220169497b35413a194aa42198df2c171939c703a50fe127068db146`; and the
negative trace is
`22149ae449e20ff8576019ebe6e79ac3587c019c40b4b81e600bc259ccc59697`.
Per-row and per-field source path, field name, value hash, and validation status
are recorded in [`questions_provenance.csv`](questions_provenance.csv).

## Frozen result

The canonical dataset is in
[`ablation_portability/dataset/`](../ablation_portability/dataset/):

- 375 rows and 375 unique IDs, in original order.
- 300 answerable and 75 designated-negative records.
- No duplicate question text and no missing required fields.
- No runtime resampling; `select_frozen_query()` takes an explicit ID and fails
  when it is absent.
- Deterministic compact UTF-8 JSONL serialization and byte-identical repeat
  extraction.

`questions.jsonl` SHA-256:
`d2820ced7c2994218245887ad753700923262fb91ec983eeacc01b217ec43b6b`.
`questions_manifest.json` SHA-256:
`e99ab9055dbe883da72f5466afa09629b13f1b9139f516a45f8aa38e21f4cee9`.
`questions_provenance.csv` SHA-256:
`66399b54b2ba81950caecd8a2368334191aa3282038b025caaf3a18df6bcb0be`.
All three hashes are in the dataset directory's `SHA256SUMS`. The manifest records
`EXACT_SOURCE_RECOVERED`; it does not establish which database was used in July.
