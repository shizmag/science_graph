"""Manifest-guarded durable state for the corrected R1 MLX experiment.

The CUDA :class:`CheckpointStore` is deliberately reused for atomic result
files.  This module adds the MLX experiment namespace and the stronger
accepted-result/attempt invariants that the legacy ``(arm_id, query_id)`` key
does not provide.
"""

from __future__ import annotations

from contextlib import closing, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any
import uuid

from ablation_cuda.runtime.checkpoints import (
    CheckpointStore,
    atomic_write_json,
    sha256_file,
    stable_input_hash,
)

from .identity import load_manifest


class CheckpointError(RuntimeError):
    """Raised when a durable operation violates the frozen run identity."""


_SUCCESS = "SUCCESS"
_RUNNING = "RUNNING"
_RESERVED = "__RESERVED__"
_TERMINAL_STATUSES = frozenset(
    {"OVERFLOW", "FAILED_RETRYABLE", "FAILED_TERMINAL", "INTERRUPTED"}
)
_FINAL_STATUSES = frozenset({_SUCCESS, *_TERMINAL_STATUSES})


class DurableResultStore:
    """Durable state for one manifest-fingerprinted experiment namespace.

    The R1 SQLite database is the source of truth for accepted results.  The
    legacy store is only the atomic file writer/validator underneath it and
    is always created below this fresh experiment root.  Successful result
    paths contain the attempt ID, so a retry can never overwrite an earlier
    candidate file.
    """

    def __init__(self, root: Path, manifest_path: Path, manifest_hash: str):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest_hash = str(manifest_hash)
        self._assert_manifest()

        self.legacy = CheckpointStore(self.root / "legacy")
        self.database = self.root / "r1_attempts.sqlite3"
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    manifest_hash TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    arm_id TEXT NOT NULL,
                    input_identity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_path TEXT,
                    error_class TEXT,
                    error_message TEXT,
                    started_epoch REAL NOT NULL,
                    finished_epoch REAL,
                    generation_started INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS attempts_operation_idx
                    ON attempts(manifest_hash, query_id, arm_id, input_identity);
                CREATE TABLE IF NOT EXISTS accepted (
                    manifest_hash TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    arm_id TEXT NOT NULL,
                    input_identity TEXT NOT NULL,
                    result_path TEXT NOT NULL,
                    result_sha256 TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    accepted_epoch REAL NOT NULL,
                    PRIMARY KEY(manifest_hash, query_id, arm_id)
                );
                """
            )
            # Namespaces created by an earlier development revision may have
            # the table but not the crash-safe inference budget column.
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(attempts)").fetchall()
            }
            if "generation_started" not in columns:
                connection.execute(
                    "ALTER TABLE attempts ADD COLUMN generation_started "
                    "INTEGER NOT NULL DEFAULT 0"
                )
                connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database,
            timeout=30,
            isolation_level=None,
        )
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _assert_manifest(self) -> None:
        try:
            current = load_manifest(self.manifest_path)
        except Exception as exc:  # turn parser/IO errors into a protocol error
            raise CheckpointError(
                f"Unable to validate frozen experiment manifest: {self.manifest_path}"
            ) from exc
        if current.get("manifest_hash") != self.manifest_hash:
            raise CheckpointError(
                "The frozen experiment manifest changed after the checkpoint store was opened"
            )

    @contextmanager
    def execution_session(self):
        """Exclusively own this experiment while reconciling/resuming it."""
        self._assert_manifest()
        lock_path = self.root / ".r1-run.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CheckpointError(
                    f"Another process is already running this experiment: {self.root}"
                ) from exc
            self._assert_manifest()
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _relative_path(self, path: Path) -> str:
        resolved = Path(path).expanduser().resolve()
        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise CheckpointError(
                f"Checkpoint path escapes experiment root: {path}"
            ) from exc

    def _safe_path(self, relative: str) -> Path:
        if not relative or relative == _RESERVED:
            raise CheckpointError("Accepted result has no committed result path")
        candidate = Path(relative)
        if candidate.is_absolute():
            raise CheckpointError(f"Accepted result path is not relative: {relative}")
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise CheckpointError(
                f"Accepted result path escapes experiment root: {relative}"
            ) from exc
        return resolved

    def _attempt(self, connection: sqlite3.Connection, attempt_id: str) -> tuple[Any, ...]:
        row = connection.execute(
            """SELECT query_id, arm_id, input_identity, status, result_path,
                      error_class, error_message
                 FROM attempts
                WHERE manifest_hash=? AND attempt_id=?""",
            (self.manifest_hash, attempt_id),
        ).fetchone()
        if row is None:
            raise CheckpointError(f"Unknown checkpoint attempt: {attempt_id}")
        return row

    @staticmethod
    def _verify_success_payload(
        path: Path,
        result: dict[str, Any],
        *,
        input_identity: str,
    ) -> None:
        if result.get("status") != _SUCCESS:
            raise CheckpointError(f"Accepted result is not SUCCESS: {path}")
        if result.get("input_identity") != input_identity:
            raise CheckpointError(f"Accepted result identity mismatch: {path}")
        prompt = result.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise CheckpointError(f"Accepted result has no prompt: {path}")
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if result.get("prompt_sha256") != prompt_sha256:
            raise CheckpointError(f"Accepted result prompt hash mismatch: {path}")
        expected_stable_id = stable_input_hash(
            {"input_identity": input_identity, "prompt_sha256": prompt_sha256}
        )
        if result.get("stable_result_id") != expected_stable_id:
            raise CheckpointError(f"Accepted result stable ID mismatch: {path}")

    def accepted(
        self,
        query_id: str,
        arm_id: str,
        input_identity: str,
    ) -> dict[str, Any] | None:
        """Return a verified accepted result, or ``None`` if not accepted."""
        self._assert_manifest()
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT input_identity, result_path, result_sha256,
                          attempt_id, accepted_epoch
                     FROM accepted
                    WHERE manifest_hash=? AND query_id=? AND arm_id=?""",
                (self.manifest_hash, query_id, arm_id),
            ).fetchone()
        if row is None:
            return None
        if row[0] != input_identity:
            raise CheckpointError(
                f"Accepted operation identity mismatch: {query_id}/{arm_id}"
            )
        path = self._safe_path(str(row[1]))
        if not path.is_file() or sha256_file(path) != row[2]:
            raise CheckpointError(f"Accepted result file failed integrity check: {path}")
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"Accepted result is not valid JSON: {path}") from exc
        if not isinstance(result, dict):
            raise CheckpointError(f"Accepted result is not an object: {path}")
        self._verify_success_payload(path, result, input_identity=input_identity)
        return {
            "query_id": query_id,
            "arm_id": arm_id,
            "attempt_id": row[3],
            "accepted_epoch": row[4],
            "result_path": path,
            "result": result,
        }

    def accepted_rows(self) -> list[dict[str, Any]]:
        """Return every accepted result after validating its file and identity.

        The accepted table, not a directory scan, defines the scientific
        export set. Each row is re-read through :meth:`accepted` so a stale,
        modified, or orphaned file can never be exported as a completed run.
        """
        self._assert_manifest()
        with closing(self._connect()) as connection:
            operations = connection.execute(
                """SELECT query_id, arm_id, input_identity
                     FROM accepted
                    WHERE manifest_hash=?
                    ORDER BY query_id, arm_id""",
                (self.manifest_hash,),
            ).fetchall()
        verified: list[dict[str, Any]] = []
        for query_id, arm_id, input_identity in operations:
            result = self.accepted(query_id, arm_id, input_identity)
            if result is None:
                raise CheckpointError(
                    f"Accepted operation disappeared during export: {query_id}/{arm_id}"
                )
            verified.append(result)
        return verified

    def operation_states(
        self,
        query_ids: list[str] | tuple[str, ...],
        arm_ids: list[str] | tuple[str, ...],
    ) -> dict[tuple[str, str], str | None]:
        """Return durable completion state without requiring tokenization.

        This is deliberately a preflight-only view.  Accepted rows are read
        through :meth:`accepted_rows`, so their result files and ledger
        identities are still verified.  The latest non-accepted attempt is
        reported for each operation.  ``FAILED_RETRYABLE`` and
        ``INTERRUPTED`` remain pending; ``SUCCESS``, ``OVERFLOW`` and
        ``FAILED_TERMINAL`` are complete for the purpose of deciding whether
        a resumed command needs to load the model.

        The method does not replace the runner's prompt/tokenizer identity
        check.  It only permits a safe no-op or over-budget return before a
        model/tokenizer is constructed when the frozen namespace has no work
        that could be attempted by this invocation.
        """
        self._assert_manifest()
        requested = {
            (str(query_id), str(arm_id))
            for query_id in query_ids
            for arm_id in arm_ids
        }
        if not requested:
            return {}

        accepted = {
            (str(item["query_id"]), str(item["arm_id"]))
            for item in self.accepted_rows()
        }
        states: dict[tuple[str, str], str | None] = {
            key: ("SUCCESS" if key in accepted else None)
            for key in requested
        }
        query_values = sorted({key[0] for key in requested})
        arm_values = sorted({key[1] for key in requested})
        query_placeholders = ",".join("?" for _ in query_values)
        arm_placeholders = ",".join("?" for _ in arm_values)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""SELECT query_id, arm_id, status
                       FROM attempts
                      WHERE manifest_hash=?
                        AND query_id IN ({query_placeholders})
                        AND arm_id IN ({arm_placeholders})
                      ORDER BY rowid DESC""",
                (self.manifest_hash, *query_values, *arm_values),
            ).fetchall()
        for query_id, arm_id, status in rows:
            key = (str(query_id), str(arm_id))
            if key in states and states[key] is None:
                states[key] = str(status)
        return states

    @staticmethod
    def is_complete_operation_status(status: str | None) -> bool:
        """Whether a stored state needs no new model work on resume."""
        return status in {_SUCCESS, "OVERFLOW", "FAILED_TERMINAL"}

    def recover_inflight(self) -> int:
        """Mark RUNNING attempts from a prior process as INTERRUPTED.

        The runner is intentionally sequential.  Calling this at startup is
        the explicit reconciliation step for an arbitrary process exit; it
        never marks an accepted result as failed.
        """
        self._assert_manifest()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """UPDATE attempts
                      SET status=?, error_class=?, error_message=?, finished_epoch=?
                    WHERE manifest_hash=? AND status=?""",
                (
                    "INTERRUPTED",
                    "ProcessRestart",
                    "Recovered a RUNNING attempt before resuming the frozen run",
                    time.time(),
                    self.manifest_hash,
                    _RUNNING,
                ),
            )
            connection.commit()
        return int(cursor.rowcount)

    def begin_attempt(self, query_id: str, arm_id: str, input_identity: str) -> str:
        """Reserve one append-only RUNNING attempt."""
        self._assert_manifest()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            accepted = connection.execute(
                """SELECT input_identity FROM accepted
                    WHERE manifest_hash=? AND query_id=? AND arm_id=?""",
                (self.manifest_hash, query_id, arm_id),
            ).fetchone()
            if accepted is not None:
                if accepted[0] != input_identity:
                    raise CheckpointError(
                        f"Accepted operation identity mismatch: {query_id}/{arm_id}"
                    )
                raise CheckpointError(f"Operation is already accepted: {query_id}/{arm_id}")
            running = connection.execute(
                """SELECT attempt_id FROM attempts
                    WHERE manifest_hash=? AND query_id=? AND arm_id=?
                      AND input_identity=? AND status=?""",
                (self.manifest_hash, query_id, arm_id, input_identity, _RUNNING),
            ).fetchone()
            if running is not None:
                raise CheckpointError(f"Operation already has a RUNNING attempt: {running[0]}")
            attempt_id = uuid.uuid4().hex
            connection.execute(
                """INSERT INTO attempts(
                    attempt_id, manifest_hash, query_id, arm_id,
                    input_identity, status, started_epoch
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    attempt_id,
                    self.manifest_hash,
                    query_id,
                    arm_id,
                    input_identity,
                    _RUNNING,
                    time.time(),
                ),
            )
            connection.commit()
        return attempt_id

    def mark_generation_started(self, attempt_id: str) -> None:
        """Durably consume one inference-call budget unit before adapter call.

        The marker is conservative across a process crash in the tiny interval
        before `generate` is entered: a possibly-started call is never refunded.
        This makes the six-pair smoke ceiling cumulative across restarts.
        """
        self._assert_manifest()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            attempt = self._attempt(connection, attempt_id)
            if attempt[3] != _RUNNING:
                raise CheckpointError(f"Attempt {attempt_id} is already {attempt[3]}")
            cursor = connection.execute(
                """UPDATE attempts SET generation_started=1
                     WHERE attempt_id=? AND manifest_hash=? AND status=?
                       AND generation_started=0""",
                (attempt_id, self.manifest_hash, _RUNNING),
            )
            if cursor.rowcount != 1:
                raise CheckpointError(f"Attempt {attempt_id} already consumed inference budget")
            connection.commit()

    def generation_calls(self) -> int:
        """Return the cumulative, crash-safe adapter-call budget consumed."""
        self._assert_manifest()
        with closing(self._connect()) as connection:
            result = connection.execute(
                """SELECT COUNT(*) FROM attempts
                     WHERE manifest_hash=? AND generation_started=1""",
                (self.manifest_hash,),
            ).fetchone()
        return int(result[0])

    def finish_attempt(
        self,
        attempt_id: str,
        *,
        status: str,
        result_path: Path | None = None,
        error_class: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Finalize exactly one RUNNING attempt."""
        self._assert_manifest()
        if status not in _FINAL_STATUSES:
            raise CheckpointError(f"Unsupported attempt status: {status}")
        relative = self._relative_path(result_path) if result_path is not None else None
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            attempt = self._attempt(connection, attempt_id)
            if attempt[3] != _RUNNING:
                raise CheckpointError(f"Attempt {attempt_id} is already {attempt[3]}")
            cursor = connection.execute(
                """UPDATE attempts SET status=?, result_path=?, error_class=?,
                          error_message=?, finished_epoch=?
                    WHERE attempt_id=? AND manifest_hash=? AND status=?""",
                (
                    status,
                    relative,
                    error_class,
                    error_message,
                    time.time(),
                    attempt_id,
                    self.manifest_hash,
                    _RUNNING,
                ),
            )
            if cursor.rowcount != 1:
                raise CheckpointError(f"Attempt {attempt_id} was not finalized")
            connection.commit()

    def commit_success(
        self,
        *,
        attempt_id: str,
        query_id: str,
        arm_id: str,
        input_identity: str,
        result: dict[str, Any],
    ) -> Path:
        """Write and durably accept one SUCCESS without replacement."""
        self._assert_manifest()
        if result.get("status") != _SUCCESS or result.get("input_identity") != input_identity:
            raise CheckpointError("Only a matching SUCCESS result may be accepted")
        payload = dict(result)
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise CheckpointError("A SUCCESS result must contain a non-empty prompt")
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if payload.get("prompt_sha256") != prompt_sha256:
            raise CheckpointError("Result prompt hash does not match the persisted prompt")
        payload["stable_result_id"] = stable_input_hash(
            {"input_identity": input_identity, "prompt_sha256": prompt_sha256}
        )
        result_path = self.root / "results" / arm_id / f"{query_id}.{attempt_id}.json"
        relative = self._relative_path(result_path)

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                attempt = self._attempt(connection, attempt_id)
                if attempt[:3] != (query_id, arm_id, input_identity):
                    raise CheckpointError("Success does not match checkpoint attempt")
                if attempt[3] != _RUNNING:
                    raise CheckpointError(f"Attempt {attempt_id} is already {attempt[3]}")
                existing = connection.execute(
                    """SELECT input_identity FROM accepted
                        WHERE manifest_hash=? AND query_id=? AND arm_id=?""",
                    (self.manifest_hash, query_id, arm_id),
                ).fetchone()
                if existing is not None:
                    raise CheckpointError(f"Operation is already accepted: {query_id}/{arm_id}")
                connection.execute(
                    """INSERT INTO accepted(
                        manifest_hash, query_id, arm_id, input_identity,
                        result_path, result_sha256, attempt_id, accepted_epoch
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        self.manifest_hash,
                        query_id,
                        arm_id,
                        input_identity,
                        _RESERVED,
                        _RESERVED,
                        attempt_id,
                        time.time(),
                    ),
                )
                self.legacy.put(
                    arm_id,
                    query_id,
                    input_identity,
                    result_path,
                    payload,
                    attempt_count=1,
                )
                result_sha256 = sha256_file(result_path)
                accepted_update = connection.execute(
                    """UPDATE accepted SET result_sha256=?, result_path=?
                        WHERE manifest_hash=? AND query_id=? AND arm_id=?
                          AND attempt_id=? AND result_path=?""",
                    (
                        result_sha256,
                        relative,
                        self.manifest_hash,
                        query_id,
                        arm_id,
                        attempt_id,
                        _RESERVED,
                    ),
                )
                if accepted_update.rowcount != 1:
                    raise CheckpointError("Accepted result reservation was lost")
                attempt_update = connection.execute(
                    """UPDATE attempts SET status=?, result_path=?, finished_epoch=?
                        WHERE attempt_id=? AND manifest_hash=? AND status=?""",
                    (
                        _SUCCESS,
                        relative,
                        time.time(),
                        attempt_id,
                        self.manifest_hash,
                        _RUNNING,
                    ),
                )
                if attempt_update.rowcount != 1:
                    raise CheckpointError("Success attempt was not finalized")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return result_path

    def record_terminal(
        self,
        *,
        attempt_id: str,
        query_id: str,
        arm_id: str,
        input_identity: str,
        status: str,
        result: dict[str, Any],
        error_class: str | None = None,
        error_message: str | None = None,
    ) -> Path:
        """Persist a non-success outcome and finalize its attempt."""
        self._assert_manifest()
        if status not in _TERMINAL_STATUSES:
            raise CheckpointError(f"Unsupported terminal status: {status}")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            attempt = self._attempt(connection, attempt_id)
            if attempt[:3] != (query_id, arm_id, input_identity):
                raise CheckpointError("Terminal result does not match checkpoint attempt")
            if attempt[3] != _RUNNING:
                raise CheckpointError(f"Attempt {attempt_id} is already {attempt[3]}")
            connection.rollback()

        payload = dict(result)
        payload.update(
            {
                "status": status,
                "query_id": query_id,
                "arm_id": arm_id,
                "input_identity": input_identity,
                "manifest_hash": self.manifest_hash,
                "attempt_id": attempt_id,
            }
        )
        result_path = self.root / "outcomes" / arm_id / f"{query_id}.{attempt_id}.json"
        self._relative_path(result_path)
        atomic_write_json(result_path, payload)
        self.finish_attempt(
            attempt_id,
            status=status,
            result_path=result_path,
            error_class=error_class,
            error_message=error_message,
        )
        return result_path

    def record_failure(
        self,
        *,
        attempt_id: str,
        query_id: str,
        arm_id: str,
        input_identity: str,
        error_class: str,
        error_message: str,
        retryable: bool = True,
    ) -> Path:
        return self.record_terminal(
            attempt_id=attempt_id,
            query_id=query_id,
            arm_id=arm_id,
            input_identity=input_identity,
            status="FAILED_RETRYABLE" if retryable else "FAILED_TERMINAL",
            result={},
            error_class=error_class,
            error_message=error_message,
        )

    def progress(self) -> dict[str, int]:
        self._assert_manifest()
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT status, COUNT(*) FROM attempts
                    WHERE manifest_hash=? GROUP BY status""",
                (self.manifest_hash,),
            ).fetchall()
            accepted = connection.execute(
                "SELECT COUNT(*) FROM accepted WHERE manifest_hash=?",
                (self.manifest_hash,),
            ).fetchone()[0]
            attempts = connection.execute(
                "SELECT COUNT(*) FROM attempts WHERE manifest_hash=?",
                (self.manifest_hash,),
            ).fetchone()[0]
            generation_calls = connection.execute(
                "SELECT COUNT(*) FROM attempts WHERE manifest_hash=? AND generation_started=1",
                (self.manifest_hash,),
            ).fetchone()[0]
        result = {str(status).lower(): int(count) for status, count in rows}
        result["accepted"] = int(accepted)
        result["attempts"] = int(attempts)
        result["generation_calls"] = int(generation_calls)
        return result
