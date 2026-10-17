from __future__ import annotations

import argparse
import os
import sqlite3
import tempfile
from pathlib import Path

from ablation_portability.storage import read_only_connection


def restore_snapshot(snapshot: Path, destination: Path) -> Path:
    source_db = snapshot / "database.db" if snapshot.is_dir() else snapshot
    source_db = source_db.resolve(strict=True)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Restore destination must be new or empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "database.db"
    with tempfile.NamedTemporaryFile(
        prefix=".restore.", suffix=".db.tmp", dir=destination, delete=False
    ) as handle:
        temp_path = Path(handle.name)
    try:
        src = read_only_connection(source_db)
        dst = sqlite3.connect(temp_path, timeout=30.0)
        try:
            dst.execute("PRAGMA journal_mode=DELETE")
            src.backup(dst, pages=2048, sleep=0.05)
            dst.commit()
            dst.execute("PRAGMA journal_mode=DELETE")
            if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Restored SQLite database failed integrity_check")
        finally:
            src.close()
            dst.close()
        os.replace(temp_path, target)
        return target
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore a snapshot to a clean directory")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(restore_snapshot(args.snapshot, args.destination))


if __name__ == "__main__":
    main()
