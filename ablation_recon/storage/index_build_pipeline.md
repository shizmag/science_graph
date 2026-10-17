# Index build pipeline

back/benchmarks/rag/index_pubmedqa_sample.py:26-95 samples 300 Parquet rows using pandas random_state=42 and writes Markdown notes; :97-110 and :126-159 show the index call and default use_llm=False. Shared indexing path chunks documents, embeds and writes SQLite/USEARCH. Historical config identifies E5, but DB vectors have no revision fingerprint.

Recovered stages: local parquet acquisition (revision/license unknown); seed-42 sample; 100 negative candidates (creation unknown); Markdown note writing; chunking/ID assignment (exact parameters/version not frozen); E5 passage embedding and float32 insertion; SQLite FTS5; graph/ontology extraction; USEARCH cosine sidecar. Query time uses E5 query prefix and optional spaCy concept processing.

Exact rebuild requires source/note hashes, source order, chunker settings/version, stable IDs, E5 revision/prefix/pooling/truncation/dtype/normalization, FTS tokenizer, graph extraction, ontology data, and usearch version. CUDA embedding may change floating-point details/ties; graph and lexical paths are mostly CPU. Rebuild was not performed.
