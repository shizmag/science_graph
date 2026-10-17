# Query population audit

Sources:
- back/benchmarks/rag/dataset/pubmedqa/pubmedqf.parquet: 1,075,513 bytes; upstream revision/license absent.
- sampled_300.json: 585,335 bytes, 300 records.
- pubmedqa_golden.yaml: 271,123 bytes, 400 candidate rows: 300 answerable, 100 negative candidates, five negative categories with 20 each.
- notes/: 300 Markdown files, about 3.1MB.
- Historical retrieved_contexts.yaml: 375 rows, strongest evidence of completed-run membership.

[VERIFIED_FROM_ARTIFACT] Archived population has exactly 375 unique IDs: 300 answerable and 75 designated-negative. IDs/text cross-match source; selected 75 negatives match seed42 sampling from 100 candidates. Positive golds match. Negative golds are empty string in frozen run artifact versus null-like source representation.

[VERIFIED_FROM_CODE] load_benchmark_dataset in back/benchmarks/rag/core/config.py:418-499 samples with random.Random(seed=42) when limits are provided and sorts IDs. Startup can resample; future harness must load a standalone frozen manifest and never sample.

back/benchmarks/rag/index_pubmedqa_sample.py:26-95 samples 300 with pandas random_state=42 and writes Markdown notes; :97-110 and :126-159 show the default index call uses use_llm=False. Negative-generation procedure and original source revision are unknown. No thin standalone manifest exists. Historical dataset hash is not a hash of a clean question-only file. Do not modify questions.
