"""Transactional per-query checkpoints and dependency-keyed stage cache."""

from __future__ import annotations

import hashlib
import gzip
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_input_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_DIRECTORY)
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, json.dumps(value, ensure_ascii=False, sort_keys=True,
                                        indent=2, allow_nan=False).encode("utf-8") + b"\n")


def hydrate_result(value: dict[str, Any], record_path: Path, experiment_root: Path) -> dict[str, Any]:
    reference = value.get("telemetry_sidecar")
    if not reference:
        return value
    relative = Path(str(reference.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(reference.get("path", "")):
        raise ValueError("Unsafe token-telemetry sidecar path")
    sidecar = experiment_root / relative
    if not sidecar.is_file() or sha256_file(sidecar) != reference.get("sha256"):
        raise ValueError("Token-telemetry sidecar is missing or corrupt")
    try:
        with gzip.open(sidecar, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Token-telemetry sidecar cannot be decoded") from exc
    events = value.get("generation_events", [])
    for row in rows:
        index = int(row["event_index"])
        if index < 0 or index >= len(events):
            raise ValueError("Token-telemetry sidecar references a missing generation event")
        events[index]["token_telemetry"] = row.get("token_telemetry", [])
    final = next((event for event in reversed(events)
                  if event.get("purpose") == "final_answer_generation"), None)
    value["token_telemetry"] = final.get("token_telemetry", []) if final else []
    return value


class CheckpointStore:
    """SQLite index points only to atomically written, content-verified result files."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = root / "index.sqlite3"
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("""CREATE TABLE IF NOT EXISTS operations (
                arm_id TEXT NOT NULL,
                query_id TEXT NOT NULL,
                input_identity TEXT NOT NULL,
                result_path TEXT NOT NULL,
                result_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 1,
                updated_epoch REAL NOT NULL,
                PRIMARY KEY (arm_id, query_id)
            )""")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(operations)")}
            if "new_token_count" not in columns:
                connection.execute("ALTER TABLE operations ADD COLUMN new_token_count INTEGER NOT NULL DEFAULT 0")
            if "reused_token_count" not in columns:
                connection.execute("ALTER TABLE operations ADD COLUMN reused_token_count INTEGER NOT NULL DEFAULT 0")
            if "cache_hit_count" not in columns:
                connection.execute("ALTER TABLE operations ADD COLUMN cache_hit_count INTEGER NOT NULL DEFAULT 0")
            if "reused_generation_call_count" not in columns:
                connection.execute("ALTER TABLE operations ADD COLUMN reused_generation_call_count INTEGER NOT NULL DEFAULT 0")
            connection.execute("""CREATE TABLE IF NOT EXISTS stage_cache (
                stage TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                created_epoch REAL NOT NULL,
                PRIMARY KEY (stage, cache_key)
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS retries (
                arm_id TEXT NOT NULL, query_id TEXT NOT NULL, attempt INTEGER NOT NULL,
                error TEXT NOT NULL, transient INTEGER NOT NULL, time_epoch REAL NOT NULL,
                PRIMARY KEY(arm_id,query_id,attempt))""")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def get(self, arm_id: str, query_id: str, input_identity: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT result_path,result_sha256,status FROM operations WHERE arm_id=? AND query_id=? AND input_identity=?",
                (arm_id, query_id, input_identity),
            ).fetchone()
        if not row or row[2] != "SUCCESS":
            return None
        path = self.root.parent / row[0]
        if not path.is_file() or sha256_file(path) != row[1]:
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if value.get("status") != "SUCCESS" or value.get("input_identity") != input_identity:
            return None
        prompt = value.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            return None
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if value.get("prompt_sha256") != prompt_hash:
            return None
        if value.get("stable_result_id") != stable_input_hash({
            "input_identity": input_identity,
            "prompt_sha256": prompt_hash,
        }):
            return None
        try:
            return hydrate_result(value, path, self.root.parent)
        except ValueError:
            return None

    def put(self, arm_id: str, query_id: str, input_identity: str,
            result_path: Path, result: dict[str, Any], attempt_count: int = 1) -> None:
        result_path = result_path.resolve()
        stored_result = dict(result)
        generation_events = result.get("generation_events", [])
        telemetry_rows = []
        stored_events = []
        for event_index, event in enumerate(generation_events):
            event_copy = {key: value for key, value in event.items() if key != "token_telemetry"}
            stored_events.append(event_copy)
            token_rows = event.get("token_telemetry", [])
            if token_rows:
                telemetry_rows.append({"event_index": event_index, "token_telemetry": token_rows})
        stored_result["generation_events"] = stored_events
        stored_result.pop("token_telemetry", None)
        if telemetry_rows:
            sidecar_path = result_path.with_suffix(".telemetry.jsonl.gz")
            sidecar_payload = "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False) + "\n"
                for row in telemetry_rows
            ).encode("utf-8")
            compressed = gzip.compress(sidecar_payload, compresslevel=6, mtime=0)
            atomic_write_bytes(sidecar_path, compressed)
            stored_result["telemetry_sidecar"] = {
                "path": sidecar_path.relative_to(self.root.parent.resolve()).as_posix(),
                "sha256": hashlib.sha256(compressed).hexdigest(),
                "generation_event_count": len(generation_events),
                "token_count": sum(len(row["token_telemetry"]) for row in telemetry_rows),
                "units": {"entropy": "bits", "log_probability": "nats", "probability": "fraction"},
                "schema": "gzip-compressed JSONL; generation event index plus per-token telemetry",
            }
        else:
            stored_result["telemetry_sidecar"] = None
        payload = json.dumps(stored_result, ensure_ascii=False, sort_keys=True,
                             indent=2, allow_nan=False).encode("utf-8") + b"\n"
        atomic_write_bytes(result_path, payload)
        digest = hashlib.sha256(payload).hexdigest()
        relative = result_path.relative_to(self.root.parent.resolve()).as_posix()
        status = str(result.get("status", "PARTIAL"))
        new_token_count = int(result.get("cumulative_new_cuda_tokens", 0))
        reused_token_count = int(result.get("reused_generation_tokens", 0))
        trace = result.get("retrieval_candidates_and_ranking_trace", {})
        cache_hit_count = sum(int(event.get("items", 1)) for event in trace.get("stage_cache_events", [])
                              if event.get("hit"))
        reused_generation_call_count = sum(
            1 for event in result.get("generation_events", [])
            if event.get("event", {}).get("cache_hit")
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""INSERT INTO operations
                (arm_id,query_id,input_identity,result_path,result_sha256,status,attempt_count,updated_epoch,
                 new_token_count,reused_token_count,cache_hit_count,reused_generation_call_count)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(arm_id,query_id) DO UPDATE SET
                input_identity=excluded.input_identity,result_path=excluded.result_path,
                result_sha256=excluded.result_sha256,status=excluded.status,
                attempt_count=excluded.attempt_count,updated_epoch=excluded.updated_epoch,
                new_token_count=excluded.new_token_count,reused_token_count=excluded.reused_token_count,
                cache_hit_count=excluded.cache_hit_count,
                reused_generation_call_count=excluded.reused_generation_call_count""",
                (arm_id, query_id, input_identity, relative, digest, status, attempt_count,
                 time.time(), new_token_count, reused_token_count, cache_hit_count,
                 reused_generation_call_count))
            connection.commit()

    def record_attempt(self, arm_id: str, query_id: str, *, error: str, attempt: int,
                       transient: bool) -> None:
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS retries (
                arm_id TEXT NOT NULL, query_id TEXT NOT NULL, attempt INTEGER NOT NULL,
                error TEXT NOT NULL, transient INTEGER NOT NULL, time_epoch REAL NOT NULL,
                PRIMARY KEY(arm_id,query_id,attempt))""")
            connection.execute("INSERT OR REPLACE INTO retries VALUES(?,?,?,?,?,?)",
                               (arm_id, query_id, attempt, error, int(transient), time.time()))

    def next_attempt(self, arm_id: str, query_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COALESCE(MAX(attempt),0) FROM retries WHERE arm_id=? AND query_id=?",
                                     (arm_id, query_id)).fetchone()
            operation = connection.execute("SELECT attempt_count FROM operations WHERE arm_id=? AND query_id=?",
                                           (arm_id, query_id)).fetchone()
        return max(int(row[0]) + 1, int(operation[0]) + 1 if operation else 1)

    def counters(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute("SELECT status,COUNT(*) FROM operations GROUP BY status").fetchall()
            tokens = connection.execute("""SELECT COALESCE(SUM(new_token_count),0),
                COALESCE(SUM(reused_token_count),0),COALESCE(SUM(cache_hit_count),0),
                COALESCE(SUM(reused_generation_call_count),0) FROM operations WHERE status='SUCCESS'""").fetchone()
            retries = connection.execute("SELECT COUNT(*) FROM retries WHERE attempt>1").fetchone()
        values = {status: count for status, count in rows}
        return {"successful_operations": values.get("SUCCESS", 0),
                "failed_operations": sum(count for status, count in values.items() if status != "SUCCESS"),
                "new_generated_tokens": int(tokens[0]), "reused_generation_tokens": int(tokens[1]),
                "cache_hits": int(tokens[2]), "reused_generations": int(tokens[3]),
                "retried_operations": int(retries[0])}

    def cache_get(self, stage: str, dependencies: dict[str, Any]) -> Any | None:
        cache_key = stable_input_hash(dependencies)
        with self._connect() as connection:
            row = connection.execute("SELECT value_json FROM stage_cache WHERE stage=? AND cache_key=?",
                                     (stage, cache_key)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def cache_put(self, stage: str, dependencies: dict[str, Any], value: Any) -> str:
        cache_key = stable_input_hash(dependencies)
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._connect() as connection:
            connection.execute("INSERT OR REPLACE INTO stage_cache VALUES(?,?,?,?)",
                               (stage, cache_key, payload, time.time()))
        return cache_key

    def cache_stats(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute("SELECT stage,COUNT(*) FROM stage_cache GROUP BY stage").fetchall()
        return {stage: count for stage, count in rows}
