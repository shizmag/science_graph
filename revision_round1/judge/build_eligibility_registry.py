"""Build a per-ID full-evidence eligibility table from saved historical artifacts."""
from pathlib import Path
try:  # support both ``python path/to/script.py`` and ``python -m``
    from .eligibility_registry import build_detailed_historical_registry, write_detailed_registry
except ImportError:  # pragma: no cover - exercised by direct-script invocation
    from eligibility_registry import build_detailed_historical_registry, write_detailed_registry

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "graphs/run_20260729_211324_OCC-RAG-1.7B"
HERE = Path(__file__).resolve().parent


def main():
    records = build_detailed_historical_registry(
        RUN, HERE / "historical_detail_availability.csv", HERE / "second_judge_scores.jsonl"
    )
    output = HERE / "eligibility_registry.csv"
    write_detailed_registry(records, output)
    print(f"paired IDs={len(records)//2}; records={len(records)}; corrected-judge eligible={sum(r['corrected_judge_eligible'] for r in records)//2}; output={output}")

if __name__ == "__main__":
    main()
