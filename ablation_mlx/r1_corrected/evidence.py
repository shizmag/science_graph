"""Read-only evidence assembly for the corrected R1 protocol."""

from __future__ import annotations

from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping

from .protocol import EvidenceBundle, EvidenceChunk, GraphRelation


class EvidenceError(ValueError):
    """Raised when the frozen evidence source cannot be used safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _source_candidates(record: dict[str, Any]) -> tuple[str, ...]:
    """Derive only explicit frozen source identifiers; never perform retrieval."""

    values: list[str] = []
    for key in ("source_pmid", "source_id"):
        value = record.get(key)
        if value:
            values.append(str(value))
    for key in ("original_pmids", "original_paper_ids"):
        value = record.get(key, ())
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, (list, tuple)):
            values.extend(str(item) for item in value if item)
    normalized = []
    for value in values:
        clean = value.strip()
        if clean and clean not in normalized:
            normalized.append(clean)
    return tuple(normalized)


def _like_patterns(source_id: str) -> tuple[str, ...]:
    clean = source_id.strip()
    patterns = [clean]
    if clean.isdigit():
        patterns.extend((f"pubmed_{clean}_%", f"%_{clean}_%"))
    return tuple(dict.fromkeys(patterns))


class ReadOnlyEvidenceStore:
    """Query the frozen SQLite snapshot through an immutable read-only URI."""

    # The R1 intervention is static graph *relations*, not document tags.  Tags
    # are metadata and are intentionally excluded.  These are the relation
    # families that the application's graph-neighbor path can expose as
    # structured evidence; keeping the policy explicit makes activation rate
    # reproducible and prevents a metadata-only edge from activating the arm.
    STATIC_RELATION_TYPES = frozenset({
        "CITES",
        "AUTHORED",
        "MENTIONS_CONCEPT",
        "USED_DATASET",
        "PUBLISHED_IN",
        "INTRODUCED_DATASET",
        "AFFILIATED_WITH",
        "PREREQUISITE_FOR",
        "SUBCLASS_OF",
        "SPONSORED_BY",
        "HAS_CODE",
        "IS_A",
    })

    def __init__(self, database_path: Path, expected_sha256: str, snapshot_id: str):
        self.database_path = Path(database_path).expanduser().resolve()
        self.expected_sha256 = expected_sha256
        self.snapshot_id = snapshot_id
        if not self.database_path.is_file():
            raise EvidenceError(f"Evidence database does not exist: {self.database_path}")
        actual = _sha256_file(self.database_path)
        if actual != expected_sha256:
            raise EvidenceError(
                f"Evidence database SHA-256 mismatch: expected {expected_sha256}, got {actual}"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.database_path}?mode=ro&immutable=1", uri=True
        )
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _matching_paper_ids(self, connection: sqlite3.Connection, sources: Iterable[str]) -> tuple[str, ...]:
        found: list[str] = []
        for source in sources:
            for pattern in _like_patterns(source):
                rows = connection.execute(
                    "SELECT DISTINCT paper_id FROM chunks WHERE paper_id=? OR paper_id LIKE ? ORDER BY paper_id",
                    (source, pattern),
                ).fetchall()
                for (paper_id,) in rows:
                    if paper_id and paper_id not in found:
                        found.append(str(paper_id))
        return tuple(found)

    def _chunks(self, connection: sqlite3.Connection, paper_ids: Iterable[str], limit: int) -> tuple[EvidenceChunk, ...]:
        ids = tuple(paper_ids)
        if not ids:
            return ()
        placeholders = ",".join("?" for _ in ids)
        rows = connection.execute(
            f"SELECT id,paper_id,text_content,page_number FROM chunks WHERE paper_id IN ({placeholders}) ORDER BY paper_id,page_number,id LIMIT ?",
            (*ids, limit),
        ).fetchall()
        return tuple(
            EvidenceChunk(str(row[0]), str(row[1]), str(row[2]), int(row[3] or 0))
            for row in rows
        )

    def _node_ids(self, connection: sqlite3.Connection, paper_ids: Iterable[str]) -> tuple[str, ...]:
        found: list[str] = []
        for paper_id in paper_ids:
            rows = connection.execute(
                "SELECT id FROM nodes WHERE id=? OR id LIKE ? OR properties LIKE ? ORDER BY id",
                (paper_id, f"{paper_id}%", f"%{paper_id}%"),
            ).fetchall()
            for (node_id,) in rows:
                if node_id and node_id not in found:
                    found.append(str(node_id))
        return tuple(found)

    def _relations(
        self,
        connection: sqlite3.Connection,
        node_ids: Iterable[str],
        limit: int,
        relation_types: frozenset[str],
    ) -> tuple[GraphRelation, ...]:
        ids = tuple(node_ids)
        if not ids:
            return ()
        placeholders = ",".join("?" for _ in ids)
        rows = connection.execute(
            f"SELECT source_id,target_id,type,properties FROM edges WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders})) AND type IN ({','.join('?' for _ in relation_types)}) ORDER BY source_id,target_id,type LIMIT ?",
            (*ids, *ids, *sorted(relation_types), limit),
        ).fetchall()
        labels: dict[str, str] = {}
        all_ids = {str(row[0]) for row in rows} | {str(row[1]) for row in rows}
        if all_ids:
            node_placeholders = ",".join("?" for _ in all_ids)
            for node_id, label in connection.execute(
                f"SELECT id,label FROM nodes WHERE id IN ({node_placeholders})",
                tuple(sorted(all_ids)),
            ):
                labels[str(node_id)] = str(label)
        relations = []
        for source_id, target_id, relation_type, properties in rows:
            source_id = str(source_id)
            target_id = str(target_id)
            relation_type = str(relation_type)
            relation_id = hashlib.sha256(
                f"{source_id}\0{target_id}\0{relation_type}\0{properties or ''}".encode()
            ).hexdigest()[:24]
            relation_props = tuple(sorted((str(k), str(v)) for k, v in _json_object(properties).items()))
            relations.append(GraphRelation(
                relation_id=relation_id,
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                source_label=labels.get(source_id, ""),
                target_label=labels.get(target_id, ""),
                properties=relation_props,
            ))
        return tuple(relations)

    def evidence_for_record(
        self,
        record: Mapping[str, Any],
        *,
        chunk_limit: int = 5,
        relation_limit: int = 32,
        static_relation_types: Iterable[str] | None = None,
    ) -> EvidenceBundle:
        query_id = str(record.get("query_id", ""))
        question = str(record.get("question", ""))
        if not query_id or not question:
            raise EvidenceError("Frozen record must contain non-empty query_id and question")
        sources = _source_candidates(record)
        with closing(self._connect()) as connection:
            paper_ids = self._matching_paper_ids(connection, sources)
            chunks = self._chunks(connection, paper_ids, chunk_limit)
            node_ids = self._node_ids(connection, paper_ids)
            relation_types = (
                self.STATIC_RELATION_TYPES
                if static_relation_types is None
                else frozenset(str(value) for value in static_relation_types)
            )
            relations = self._relations(connection, node_ids, relation_limit, relation_types)
        retrieval_identity = hashlib.sha256(
            json.dumps({"snapshot_id": self.snapshot_id, "sources": sources, "paper_ids": paper_ids, "chunk_limit": chunk_limit, "relation_limit": relation_limit}, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        note = (
            "Activation-positive: frozen snapshot returned at least one graph relation."
            if relations
            else "Activation-negative: frozen snapshot returned no graph relation for the selected source evidence."
        )
        return EvidenceBundle(
            query_id=query_id,
            question=question,
            source_ids=paper_ids,
            chunks=chunks,
            relations=relations,
            dynamic_enrichment="",
            retrieval_identity=retrieval_identity,
            evidence_note=note,
        )

    def population_evidence(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        chunk_limit: int = 5,
        relation_limit: int = 32,
        static_relation_types: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        rows = []
        for record in records:
            bundle = self.evidence_for_record(
                record,
                chunk_limit=chunk_limit,
                relation_limit=relation_limit,
                static_relation_types=static_relation_types,
            )
            rows.append({
                "query_id": bundle.query_id,
                "question": bundle.question,
                "answerability_label": record.get("answerability_label"),
                "historical_category": record.get("historical_category"),
                "source_pmid": record.get("source_pmid"),
                "original_pmids": record.get("original_pmids", []),
                "original_paper_ids": record.get("original_paper_ids", []),
                "synthetic_negative_category": record.get("synthetic_negative_category"),
                "source_ids": list(bundle.source_ids),
                "chunk_count": len(bundle.chunks),
                "relation_count": len(bundle.relations),
                "activation_positive": bundle.activation_positive,
                "context_chars": sum(len(chunk.text) for chunk in bundle.chunks),
                "evidence_hash": bundle.evidence_hash,
            })
        return rows
