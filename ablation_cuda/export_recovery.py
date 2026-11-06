"""Export publication-ready CSVs from a saved CUDA recovery JSON."""
from __future__ import annotations

import argparse
from pathlib import Path

from revision_replication.exports import export_cuda_recovery_tables


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    print(export_cuda_recovery_tables(args.analysis_json, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
