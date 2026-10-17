# Local one-query probe

## Result: not executed

Host is macOS ARM64 with no nvidia-smi/GPU. Selected environment lacks mlx_lm, usearch, rank_bm25 and spaCy; root environment uses Python 3.14.5 and lacks PyYAML. Assets are not all available to the active runtime.

SQLite constructors initialize schema and WAL in back/src/repository/sqlite_impl.py:49-80,1648-1663. USEARCH may rebuild and save the sidecar at :1671-1697. Running against originals could mutate persistent state. A verified disposable clone and complete matching runtime were not available. Installing/downloading was prohibited.

Archived traces and answers are historical artifacts, not a live E2E run. No exit status or timing is claimed. Next phase should restore a consistent snapshot into isolated writable temp location, protect original, pass explicit identical DB path to every stage, write output elsewhere, skip evaluator and preserve process status and all intermediates.
