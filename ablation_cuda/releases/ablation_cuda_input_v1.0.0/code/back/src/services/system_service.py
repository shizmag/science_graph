"""
System Service — manages administration and system-level operations.
"""

import shutil
from pathlib import Path
from typing import List, Dict, Any


def reset_system(db_path_str: str, archive_dir_str: str) -> List[Dict[str, Any]]:
    """
    Completely deletes all database files, vector indexes, and archive contents.
    Returns a status report list for the operations performed.
    """
    report = []
    db_path = Path(db_path_str)

    # 1. Delete SQLite Database
    if db_path.exists():
        try:
            db_path.unlink()
            report.append({
                "operation": "Delete SQLite Database",
                "path": str(db_path),
                "success": True,
                "detail": None
            })
        except Exception as e:
            report.append({
                "operation": "Delete SQLite Database",
                "path": str(db_path),
                "success": False,
                "detail": str(e)
            })

    # 2. Delete WAL & SHM files
    for suffix in ["-wal", "-shm"]:
        side_file = Path(str(db_path) + suffix)
        if side_file.exists():
            try:
                side_file.unlink()
                report.append({
                    "operation": f"Delete SQLite {suffix.upper()}",
                    "path": str(side_file),
                    "success": True,
                    "detail": None
                })
            except Exception as e:
                report.append({
                    "operation": f"Delete SQLite {suffix.upper()}",
                    "path": str(side_file),
                    "success": False,
                    "detail": str(e)
                })

    # 3. Delete USearch vector index
    usearch_path = Path(str(db_path).replace(".db", ".usearch"))
    if usearch_path.exists():
        try:
            usearch_path.unlink()
            report.append({
                "operation": "Delete USearch Index",
                "path": str(usearch_path),
                "success": True,
                "detail": None
            })
        except Exception as e:
            report.append({
                "operation": "Delete USearch Index",
                "path": str(usearch_path),
                "success": False,
                "detail": str(e)
            })

    # 4. Clear Archive Directory
    archive_dir = Path(archive_dir_str)
    if archive_dir.exists():
        try:
            for child in archive_dir.iterdir():
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
            report.append({
                "operation": "Clear Archive Directory",
                "path": str(archive_dir),
                "success": True,
                "detail": None
            })
        except Exception as e:
            report.append({
                "operation": "Clear Archive Directory",
                "path": str(archive_dir),
                "success": False,
                "detail": str(e)
            })

    return report
