# Preserved B6 worker launch failure

The worker exited with code 1 before `run_pipeline.py` loaded. Its traceback is in `execution.log`: `ModuleNotFoundError: No module named 'ablation_portability'`. Retrieval and generation did not run. The launcher now uses `python -m ablation_portability.scripts.smoke_one_query` from the project root.
