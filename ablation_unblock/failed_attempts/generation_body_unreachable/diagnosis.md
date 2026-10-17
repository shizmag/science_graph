# Preserved B6 retrieval-only attempt

This one-query run completed B6 retrieval on the frozen `external_db` restore. It returned five chunks including the expected source paper; the saved retrieval summary reports Recall 1.0 and Context Precision 1.0. The source snapshot stayed unchanged.

The process exited 0 but no `evaluation_results.yaml` was written. Inspection of `core/generation.py` showed `run_benchmarking` ended after writing its run manifest. The generation loop was indented inside `_load_existing_checkpoint` after unconditional returns, so generation was unreachable. The smoke correctly remains PARTIAL. The captured `execution.log`, retrieval YAML, graph trace, and manifest are preserved.
