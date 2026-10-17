# Embedding/index compatibility

Historical config identifies intfloat/multilingual-e5-base, revision d128750597153bb5987e10b1c3493a34e5a4502a. back/src/vector_search.py:14-123 applies query: and passage: prefixes for E5 names, SentenceTransformers encode to NumPy, no explicit normalization. Module config uses mean pooling. Model card documents 768D/512 tokens. Persisted vectors are float32; embedding inference dtype and historical batch size are not explicitly pinned in the inspected path and remain UNKNOWN.

SQLite stores 768-element float32 BLOBs (3072 bytes) at back/src/repository/sqlite_impl.py:1871-1900; USEARCH cosine sidecar is configured at :1671-1673. Database does not store exact E5 revision, tokenizer, prefix, pooling or normalization metadata. Equal dimension does not prove same vectors. Default all-MiniLM-L6-v2 is 384D and is incompatible.

Reuse only after source DB is accepted and CUDA E5 revision, dependency, prefix, pooling, truncation, dtype and normalization are fixed. Compare representative vector dimensions/norms/values, stored IDs, cosine scores and top-k ordering. Use tolerances for GPU float changes and disclose ties. If lineage cannot be verified, rebuild and label as new index baseline.
