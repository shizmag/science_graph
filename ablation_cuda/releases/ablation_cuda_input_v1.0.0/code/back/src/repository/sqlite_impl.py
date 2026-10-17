import sqlite3
import json
import os
import hashlib
import numpy as np
import threading
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple
from src.repository.base import GraphRepository, VectorRepository, ResolvedPaperNode
from src.models import Paper, Author, Concept, Chunk

def stable_hash(text: str) -> int:
    """Returns a stable 60-bit integer hash of a string ID."""
    return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:15], 16)

class ConnectionProxy:
    def __init__(self, conn, is_transaction=False):
        self._conn = conn
        self._is_transaction = is_transaction

    def __enter__(self):
        if not self._is_transaction:
            self._conn.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._is_transaction:
            return self._conn.__exit__(exc_type, exc_val, exc_tb)
        return False

    def commit(self):
        if not self._is_transaction:
            self._conn.commit()

    def rollback(self):
        if not self._is_transaction:
            self._conn.rollback()

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __del__(self):
        if not self._is_transaction and hasattr(self, "_conn") and self._conn:
            try:
                self._conn.close()
            except Exception:
                pass

class SQLiteGraphRepository(GraphRepository):
    def __init__(self, db_path: str, read_only: bool = False):
        self.db_path = db_path
        self.read_only = read_only
        self._local = threading.local()
        if not self.read_only:
            self._init_db()

    def _get_connection(self):
        active_conn = getattr(self._local, "conn", None)
        if active_conn is not None:
            return ConnectionProxy(active_conn, is_transaction=True)

        if self.read_only:
            uri = f"file:{os.path.abspath(self.db_path)}?mode=ro&immutable=1"
            conn = sqlite3.connect(uri, uri=True, timeout=30.0)
        else:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA foreign_keys = ON;")
        if not self.read_only:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA mmap_size = 268435456;")
        conn.execute("PRAGMA cache_size = -64000;")
        conn.row_factory = sqlite3.Row
        return ConnectionProxy(conn, is_transaction=False)

    @contextmanager
    def transaction(self):
        if self.read_only:
            raise RuntimeError("Read-only SQLiteGraphRepository cannot open a write transaction")
        if getattr(self._local, "conn", None) is not None:
            yield
            return

        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA mmap_size = 268435456;")
        conn.execute("PRAGMA cache_size = -64000;")
        conn.row_factory = sqlite3.Row
        self._local.conn = conn
        try:
            conn.execute("BEGIN TRANSACTION;")
            yield
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            self._local.conn = None
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            # Create nodes table with virtual generated title & is_placeholder columns
            conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                properties TEXT NOT NULL,
                title TEXT GENERATED ALWAYS AS (trim(json_extract(properties, '$.title'))) VIRTUAL COLLATE NOCASE,
                doi TEXT GENERATED ALWAYS AS (json_extract(properties, '$.doi')) VIRTUAL,
                content_hash TEXT GENERATED ALWAYS AS (json_extract(properties, '$.content_hash')) VIRTUAL,
                source_type TEXT GENERATED ALWAYS AS (json_extract(properties, '$.source_type')) VIRTUAL,
                is_placeholder INTEGER GENERATED ALWAYS AS (
                    CASE 
                        WHEN json_extract(properties, '$.is_placeholder') = 1 THEN 1
                        WHEN json_extract(properties, '$.placeholder') = 1 THEN 1
                        ELSE 0
                    END
                ) VIRTUAL
            );
            """)
            
            # Schema migration: check if columns exist in nodes table for existing setups
            cursor = conn.execute("PRAGMA table_xinfo(nodes);")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Drop old title column if it's not case-insensitive/trimmed
            if "title" in columns:
                try:
                    conn.execute("ALTER TABLE nodes DROP COLUMN title;")
                    columns.remove("title")
                except sqlite3.OperationalError:
                    pass
                    
            if "title" not in columns:
                try:
                    conn.execute("ALTER TABLE nodes ADD COLUMN title TEXT GENERATED ALWAYS AS (trim(json_extract(properties, '$.title'))) VIRTUAL COLLATE NOCASE;")
                except sqlite3.OperationalError:
                    pass
            if "doi" not in columns:
                try:
                    conn.execute("ALTER TABLE nodes ADD COLUMN doi TEXT GENERATED ALWAYS AS (json_extract(properties, '$.doi')) VIRTUAL;")
                except sqlite3.OperationalError:
                    pass
            if "content_hash" not in columns:
                try:
                    conn.execute("ALTER TABLE nodes ADD COLUMN content_hash TEXT GENERATED ALWAYS AS (json_extract(properties, '$.content_hash')) VIRTUAL;")
                except sqlite3.OperationalError:
                    pass
            if "source_type" not in columns:
                try:
                    conn.execute("ALTER TABLE nodes ADD COLUMN source_type TEXT GENERATED ALWAYS AS (json_extract(properties, '$.source_type')) VIRTUAL;")
                except sqlite3.OperationalError:
                    pass
            if "is_placeholder" not in columns:
                try:
                    conn.execute("""
                    ALTER TABLE nodes ADD COLUMN is_placeholder INTEGER GENERATED ALWAYS AS (
                        CASE 
                            WHEN json_extract(properties, '$.is_placeholder') = 1 THEN 1
                            WHEN json_extract(properties, '$.placeholder') = 1 THEN 1
                            ELSE 0
                        END
                    ) VIRTUAL;
                    """)
                except sqlite3.OperationalError:
                    pass

            # Create edges table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                properties TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id, type),
                FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
            );
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label);")
            conn.execute("DROP INDEX IF EXISTS idx_nodes_title;")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_title ON nodes(title);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_is_placeholder ON nodes(is_placeholder);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_doi ON nodes(doi);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_content_hash ON nodes(content_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_source_type ON nodes(source_type);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);")
            
            # Create bibliographic tables
            conn.execute("""
            CREATE TABLE IF NOT EXISTS reference_corpus_stats (
                work_id TEXT PRIMARY KEY,
                df INTEGER NOT NULL,
                idf REAL NOT NULL,
                n_local_papers INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_reference_vector (
                paper_id TEXT NOT NULL,
                work_id TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (paper_id, work_id),
                FOREIGN KEY (paper_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (work_id) REFERENCES nodes(id) ON DELETE CASCADE
            );
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_reference_mentions (
                chunk_id TEXT NOT NULL,
                paper_id TEXT NOT NULL,
                work_id TEXT NOT NULL,
                citation_marker TEXT,
                context TEXT,
                page_number INTEGER,
                section TEXT,
                raw_reference TEXT,
                PRIMARY KEY (chunk_id, work_id),
                FOREIGN KEY (chunk_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (paper_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (work_id) REFERENCES nodes(id) ON DELETE CASCADE
            );
            """)

            # Create new indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_type_source ON edges(type, source_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_type_target ON edges(type, target_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source_type_target ON edges(source_id, type, target_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_reference_vector_work ON paper_reference_vector(work_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_reference_mentions_work ON chunk_reference_mentions(work_id);")
            
            # Create views for local-aware graph retrieval
            conn.execute("""
            CREATE VIEW IF NOT EXISTS paper_usability_metadata AS
            SELECT 
                n.id AS canonical_paper_id,
                COALESCE(c.cnt, 0) AS chunks_count,
                CASE WHEN COALESCE(c.cnt, 0) > 0 THEN 1 ELSE 0 END AS has_chunks,
                CASE WHEN COALESCE(c.cnt, 0) > 0 THEN 1 ELSE 0 END AS is_local_indexed,
                n.is_placeholder AS is_placeholder,
                CASE WHEN COALESCE(c.cnt, 0) = 0 THEN 1 ELSE 0 END AS is_external,
                CASE WHEN COALESCE(c.cnt, 0) = 0 THEN 1 ELSE 0 END AS is_bridge_only
            FROM nodes n
            LEFT JOIN (
                SELECT paper_id, COUNT(*) AS cnt 
                FROM chunks 
                GROUP BY paper_id
            ) c ON n.id = c.paper_id
            WHERE n.label IN ('Paper', 'UserNote');
            """)
            
            conn.execute("""
            CREATE VIEW IF NOT EXISTS local_paper_projection AS
            SELECT * 
            FROM paper_usability_metadata
            WHERE chunks_count > 0;
            """)
            
            conn.execute("""
            CREATE VIEW IF NOT EXISTS bridge_only_paper_projection AS
            SELECT * 
            FROM paper_usability_metadata
            WHERE chunks_count = 0;
            """)
            
            conn.execute("""
            CREATE VIEW IF NOT EXISTS chunk_availability_mapping AS
            SELECT 
                paper_id AS canonical_paper_id,
                id AS chunk_id,
                NULL AS section,
                NULL AS table_name,
                NULL AS figure,
                NULL AS caption,
                NULL AS equation
            FROM chunks;
            """)
            
            conn.execute("""
            CREATE VIEW IF NOT EXISTS derived_local_to_local_bridge_edges AS
            SELECT 
                e.source_id AS source_local_paper_id,
                e.target_id AS target_local_paper_id,
                NULL AS bridge_node_id,
                'DIRECT_LOCAL_CITATION' AS bridge_relation_type,
                e.type AS source_relation_type,
                NULL AS target_relation_type
            FROM edges e
            JOIN local_paper_projection l1 ON e.source_id = l1.canonical_paper_id
            JOIN local_paper_projection l2 ON e.target_id = l2.canonical_paper_id
            WHERE e.type IN ('CITES', 'CITED_BY')
            
            UNION ALL
            
            SELECT 
                e1.source_id AS source_local_paper_id,
                e2.source_id AS target_local_paper_id,
                e1.target_id AS bridge_node_id,
                'SHARED_REFERENCE' AS bridge_relation_type,
                e1.type AS source_relation_type,
                e2.type AS target_relation_type
            FROM edges e1
            JOIN edges e2 ON e1.target_id = e2.target_id
            JOIN local_paper_projection l1 ON e1.source_id = l1.canonical_paper_id
            JOIN local_paper_projection l2 ON e2.source_id = l2.canonical_paper_id
            JOIN bridge_only_paper_projection b ON e1.target_id = b.canonical_paper_id
            WHERE e1.type IN ('CITES', 'CITES_IN_CONTEXT') 
              AND e2.type IN ('CITES', 'CITES_IN_CONTEXT')
              AND e1.source_id != e2.source_id
              
            UNION ALL
            
            SELECT 
                e1.source_id AS source_local_paper_id,
                e2.source_id AS target_local_paper_id,
                e1.target_id AS bridge_node_id,
                'SHARED_CITATION_BRIDGE' AS bridge_relation_type,
                e1.type AS source_relation_type,
                e2.type AS target_relation_type
            FROM edges e1
            JOIN edges e2 ON e1.target_id = e2.target_id
            JOIN local_paper_projection l1 ON e1.source_id = l1.canonical_paper_id
            JOIN local_paper_projection l2 ON e2.source_id = l2.canonical_paper_id
            JOIN bridge_only_paper_projection b ON e1.target_id = b.canonical_paper_id
            WHERE e1.type IN ('CITES', 'CITED_BY', 'CITES_IN_CONTEXT') 
              AND e2.type IN ('CITES', 'CITED_BY', 'CITES_IN_CONTEXT')
              AND e1.source_id != e2.source_id
              
            UNION ALL
            
            SELECT 
                e1.source_id AS source_local_paper_id,
                e2.source_id AS target_local_paper_id,
                e1.target_id AS bridge_node_id,
                'SHARED_CONCEPT' AS bridge_relation_type,
                e1.type AS source_relation_type,
                e2.type AS target_relation_type
            FROM edges e1
            JOIN edges e2 ON e1.target_id = e2.target_id
            JOIN local_paper_projection l1 ON e1.source_id = l1.canonical_paper_id
            JOIN local_paper_projection l2 ON e2.source_id = l2.canonical_paper_id
            JOIN nodes n ON e1.target_id = n.id
            WHERE e1.type = 'MENTIONS_CONCEPT' 
              AND e2.type = 'MENTIONS_CONCEPT'
              AND n.label = 'Concept'
              AND e1.source_id != e2.source_id
              
            UNION ALL
            SELECT NULL AS source_local_paper_id, NULL AS target_local_paper_id, NULL AS bridge_node_id, 'INTRA_PAPER_CHUNK_ADJACENCY' AS bridge_relation_type, NULL AS source_relation_type, NULL AS target_relation_type WHERE 1=0
            UNION ALL
            SELECT NULL, NULL, NULL, 'INTRA_PAPER_SECTION', NULL, NULL WHERE 1=0
            UNION ALL
            SELECT NULL, NULL, NULL, 'INTRA_PAPER_TABLE', NULL, NULL WHERE 1=0
            UNION ALL
            SELECT NULL, NULL, NULL, 'INTRA_PAPER_FIGURE', NULL, NULL WHERE 1=0
            UNION ALL
            SELECT NULL, NULL, NULL, 'INTRA_PAPER_CAPTION', NULL, NULL WHERE 1=0
            UNION ALL
            SELECT NULL, NULL, NULL, 'INTRA_PAPER_EQUATION', NULL, NULL WHERE 1=0;
            """)
            
            conn.commit()

    def save_paper(self, paper: Paper) -> None:
        if not paper.created_at:
            import datetime
            existing = self.get_paper(paper.id)
            if existing and existing.created_at:
                paper.created_at = existing.created_at
            else:
                paper.created_at = datetime.datetime.now().isoformat()

        props = {**paper.properties, "title": paper.title, "authors": paper.authors, "year": paper.year, "doi": paper.doi, "abstract": paper.abstract, "file_path": paper.file_path, "created_at": paper.created_at}
        label = "UserNote" if paper.properties.get("source_type") == "note" else "Paper"
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO nodes (id, label, properties) VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    label = CASE
                        WHEN label = 'UserNote' THEN 'UserNote'
                        WHEN label = 'Paper' AND is_placeholder = 0 THEN
                            CASE 
                                WHEN excluded.label = 'UserNote' THEN 'UserNote'
                                WHEN excluded.label = 'Paper' AND coalesce(json_extract(excluded.properties, '$.is_placeholder'), 0) != 1 AND coalesce(json_extract(excluded.properties, '$.placeholder'), 0) != 1 THEN 'Paper'
                                ELSE 'Paper'
                            END
                        ELSE excluded.label
                    END,
                    properties = CASE
                        WHEN label = 'UserNote' THEN
                            CASE WHEN excluded.label = 'UserNote' THEN excluded.properties ELSE properties END
                        WHEN label = 'Paper' AND is_placeholder = 0 THEN
                            CASE
                                WHEN excluded.label = 'UserNote' THEN excluded.properties
                                WHEN excluded.label = 'Paper' AND coalesce(json_extract(excluded.properties, '$.is_placeholder'), 0) != 1 AND coalesce(json_extract(excluded.properties, '$.placeholder'), 0) != 1 THEN excluded.properties
                                ELSE properties
                            END
                        ELSE excluded.properties
                    END
                """,
                (paper.id, label, json.dumps(props, ensure_ascii=False))
            )
            conn.commit()

    def save_nodes_bulk(self, nodes: List[Tuple[str, str, Dict[str, Any]]]) -> None:
        if not nodes:
            return
        params = []
        for node_id, label, properties in nodes:
            params.append((node_id, label, json.dumps(properties, ensure_ascii=False)))
            
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO nodes (id, label, properties) VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    label = CASE
                        WHEN label = 'UserNote' THEN 'UserNote'
                        WHEN label = 'Paper' AND is_placeholder = 0 THEN
                            CASE 
                                WHEN excluded.label = 'UserNote' THEN 'UserNote'
                                WHEN excluded.label = 'Paper' AND coalesce(json_extract(excluded.properties, '$.is_placeholder'), 0) != 1 AND coalesce(json_extract(excluded.properties, '$.placeholder'), 0) != 1 THEN 'Paper'
                                ELSE 'Paper'
                            END
                        ELSE excluded.label
                    END,
                    properties = CASE
                        WHEN label = 'UserNote' THEN
                            CASE WHEN excluded.label = 'UserNote' THEN excluded.properties ELSE properties END
                        WHEN label = 'Paper' AND is_placeholder = 0 THEN
                            CASE
                                WHEN excluded.label = 'UserNote' THEN excluded.properties
                                WHEN excluded.label = 'Paper' AND coalesce(json_extract(excluded.properties, '$.is_placeholder'), 0) != 1 AND coalesce(json_extract(excluded.properties, '$.placeholder'), 0) != 1 THEN excluded.properties
                                ELSE properties
                            END
                        ELSE excluded.properties
                    END
                """,
                params
            )
            conn.commit()

    def save_edges_bulk(self, edges: List[Tuple[str, str, str, Dict[str, Any]]]) -> None:
        if not edges:
            return
        node_placeholders = {}
        for source_id, target_id, _, _ in edges:
            for node_id in (source_id, target_id):
                if node_id not in node_placeholders:
                    if node_id.startswith("work:"):
                        label = "ExternalWork"
                    else:
                        label = "Paper" if ":" in node_id or "/" in node_id else "Concept"
                    node_placeholders[node_id] = (node_id, label, json.dumps({"title": node_id, "placeholder": True}, ensure_ascii=False))
                    
        with self._get_connection() as conn:
            if node_placeholders:
                conn.executemany(
                    "INSERT OR IGNORE INTO nodes (id, label, properties) VALUES (?, ?, ?)",
                    list(node_placeholders.values())
                )
                
            params = []
            for source_id, target_id, edge_type, properties in edges:
                params.append((source_id, target_id, edge_type, json.dumps(properties or {}, ensure_ascii=False)))
                
            conn.executemany(
                """
                INSERT INTO edges (source_id, target_id, type, properties) VALUES (?, ?, ?, ?)
                ON CONFLICT(source_id, target_id, type) DO UPDATE SET
                    properties = excluded.properties
                """,
                params
            )
            conn.commit()

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT id, properties FROM nodes WHERE id = ? AND label IN ('Paper', 'UserNote')", (paper_id,)).fetchone()
            if not row:
                return None
            props = json.loads(row["properties"])
            return Paper(
                id=row["id"],
                title=props.get("title", ""),
                authors=props.get("authors", []),
                year=props.get("year"),
                doi=props.get("doi"),
                abstract=props.get("abstract"),
                file_path=props.get("file_path"),
                created_at=props.get("created_at"),
                properties=props
            )

    def get_papers_batch(self, paper_ids: List[str]) -> Dict[str, Paper]:
        if not paper_ids:
            return {}
        unique_ids = list(set(paper_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        query = f"SELECT id, properties FROM nodes WHERE id IN ({placeholders}) AND label IN ('Paper', 'UserNote')"
        with self._get_connection() as conn:
            rows = conn.execute(query, unique_ids).fetchall()
            
        papers = {}
        for row in rows:
            props = json.loads(row["properties"])
            papers[row["id"]] = Paper(
                id=row["id"],
                title=props.get("title", ""),
                authors=props.get("authors", []),
                year=props.get("year"),
                doi=props.get("doi"),
                abstract=props.get("abstract"),
                file_path=props.get("file_path"),
                created_at=props.get("created_at"),
                properties=props
            )
        return papers

    def find_paper_by_title(self, title: str) -> Optional[Paper]:
        with self._get_connection() as conn:
            # Check if matching exact ID first
            row = conn.execute("SELECT id, properties FROM nodes WHERE id = ? AND label IN ('Paper', 'UserNote')", (title,)).fetchone()
            if row:
                props = json.loads(row["properties"])
                return Paper(
                    id=row["id"],
                    title=props.get("title", ""),
                    authors=props.get("authors", []),
                    year=props.get("year"),
                    doi=props.get("doi"),
                    abstract=props.get("abstract"),
                    file_path=props.get("file_path"),
                    created_at=props.get("created_at"),
                    properties=props
                )
            
            # Case-insensitive title match using indexed title column (which now has COLLATE NOCASE)
            clean_title = title.strip()
            row = conn.execute(
                "SELECT id, properties FROM nodes WHERE label IN ('Paper', 'UserNote') AND title = ?",
                (clean_title,)
            ).fetchone()

            if row:
                props = json.loads(row["properties"])
                return Paper(
                    id=row["id"],
                    title=props.get("title", ""),
                    authors=props.get("authors", []),
                    year=props.get("year"),
                    doi=props.get("doi"),
                    abstract=props.get("abstract"),
                    file_path=props.get("file_path"),
                    created_at=props.get("created_at"),
                    properties=props
                )
        return None

    def find_paper_by_doi(self, doi: str) -> Optional[Paper]:
        if not doi:
            return None
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id, properties FROM nodes WHERE label IN ('Paper', 'UserNote') AND doi = ?",
                (doi.strip(),)
            ).fetchone()
            if row:
                props = json.loads(row["properties"])
                return Paper(
                    id=row["id"],
                    title=props.get("title", ""),
                    authors=props.get("authors", []),
                    year=props.get("year"),
                    doi=props.get("doi"),
                    abstract=props.get("abstract"),
                    file_path=props.get("file_path"),
                    created_at=props.get("created_at"),
                    properties=props
                )
        return None

    def find_paper_by_content_hash(self, content_hash: str) -> Optional[Paper]:
        if not content_hash:
            return None
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id, properties FROM nodes WHERE label IN ('Paper', 'UserNote') AND content_hash = ?",
                (content_hash,)
            ).fetchone()
            if row:
                props = json.loads(row["properties"])
                return Paper(
                    id=row["id"],
                    title=props.get("title", ""),
                    authors=props.get("authors", []),
                    year=props.get("year"),
                    doi=props.get("doi"),
                    abstract=props.get("abstract"),
                    file_path=props.get("file_path"),
                    created_at=props.get("created_at"),
                    properties=props
                )
        return None

    def save_author(self, author: Author) -> None:
        props = {**author.properties, "name": author.name}
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO nodes (id, label, properties) VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    label = excluded.label,
                    properties = excluded.properties
                """,
                (author.id, "Author", json.dumps(props, ensure_ascii=False))
            )
            conn.commit()

    def get_author(self, author_id: str) -> Optional[Author]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT id, properties FROM nodes WHERE id = ? AND label = 'Author'", (author_id,)).fetchone()
            if not row:
                return None
            props = json.loads(row["properties"])
            return Author(
                id=row["id"],
                name=props.get("name", row["id"]),
                properties=props
            )

    def save_concept(self, concept: Concept) -> None:
        props = {**concept.properties, "name": concept.name}
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO nodes (id, label, properties) VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    label = excluded.label,
                    properties = excluded.properties
                """,
                (concept.id, "Concept", json.dumps(props, ensure_ascii=False))
            )
            conn.commit()


    def get_concept(self, concept_id: str) -> Optional[Concept]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT id, properties FROM nodes WHERE id = ? AND label = 'Concept'", (concept_id,)).fetchone()
            if not row:
                return None
            props = json.loads(row["properties"])
            return Concept(
                id=row["id"],
                name=props.get("name", row["id"]),
                properties=props
            )

    def add_edge(self, source_id: str, target_id: str, edge_type: str, properties: Dict[str, Any] = None) -> None:
        props = properties or {}
        with self._get_connection() as conn:
            # Insert placeholder nodes directly. If they already exist, SQLite IGNOREs it based on the PRIMARY KEY.
            for node_id in (source_id, target_id):
                label = "Paper" if ":" in node_id or "/" in node_id else "Concept"
                conn.execute(
                    "INSERT OR IGNORE INTO nodes (id, label, properties) VALUES (?, ?, ?)",
                    (node_id, label, json.dumps({"title": node_id, "placeholder": True}, ensure_ascii=False))
                )
            
            conn.execute(
                """
                INSERT INTO edges (source_id, target_id, type, properties) VALUES (?, ?, ?, ?)
                ON CONFLICT(source_id, target_id, type) DO UPDATE SET
                    properties = excluded.properties
                """,
                (source_id, target_id, edge_type, json.dumps(props, ensure_ascii=False))
            )
            conn.commit()

    def get_neighbors(self, node_id: str, max_depth: int = 1, allowed_edge_types: List[str] = None) -> List[tuple[str, str, str, str, str, str]]:
        if max_depth < 1:
            return []
        
        type_filter = ""
        params = {"node_id": node_id, "max_depth": max_depth}
        if allowed_edge_types is not None:
            placeholders = []
            for i, etype in enumerate(allowed_edge_types):
                key = f"etype_{i}"
                placeholders.append(f":{key}")
                params[key] = etype
            type_filter = f"AND e.type IN ({','.join(placeholders)})"
        
        # Optimized recursive CTE traversal with UNION (implicit distinct) and path-based cycle prevention
        query = f"""
        WITH RECURSIVE traverse(current_node, src_id, src_label, edge_type, tgt_id, tgt_label, edge_props, depth, path) AS (
            -- Anchor query: edges directly connected to the starting node
            SELECT 
                CASE WHEN e.source_id = :node_id THEN e.target_id ELSE e.source_id END,
                e.source_id, n1.label, e.type, e.target_id, n2.label, e.properties,
                1,
                ',' || :node_id || ',' || CASE WHEN e.source_id = :node_id THEN e.target_id ELSE e.source_id END || ','
            FROM edges e
            JOIN nodes n1 ON e.source_id = n1.id
            JOIN nodes n2 ON e.target_id = n2.id
            WHERE (e.source_id = :node_id OR e.target_id = :node_id) {type_filter}
            
            UNION
            
            -- Recursive step: transition to adjacent nodes
            SELECT 
                CASE WHEN e.source_id = t.current_node THEN e.target_id ELSE e.source_id END,
                e.source_id, n1.label, e.type, e.target_id, n2.label, e.properties,
                t.depth + 1,
                t.path || CASE WHEN e.source_id = t.current_node THEN e.target_id ELSE e.source_id END || ','
            FROM edges e
            JOIN nodes n1 ON e.source_id = n1.id
            JOIN nodes n2 ON e.target_id = n2.id
            JOIN traverse t ON (e.source_id = t.current_node OR e.target_id = t.current_node)
            WHERE t.depth < :max_depth
              AND t.path NOT LIKE '%,' || CASE WHEN e.source_id = t.current_node THEN e.target_id ELSE e.source_id END || ',%'
              {type_filter}
        )
        SELECT DISTINCT src_id, src_label, edge_type, tgt_id, tgt_label, edge_props FROM traverse;
        """
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [(
                r["src_id"], r["src_label"],
                r["edge_type"],
                r["tgt_id"], r["tgt_label"],
                r["edge_props"]
            ) for r in rows]

    def get_neighbors_batch(self, node_ids: List[str]) -> List[tuple[str, str, str, str, str, str]]:
        if not node_ids:
            return []
        placeholders = ",".join("?" for _ in node_ids)
        query = f"""
        SELECT
            e.source_id as src_id, n1.label as src_label, 
            e.type as edge_type, 
            e.target_id as tgt_id, n2.label as tgt_label, 
            e.properties as edge_props
        FROM edges e
        JOIN nodes n1 ON e.source_id = n1.id
        JOIN nodes n2 ON e.target_id = n2.id
        WHERE e.source_id IN ({placeholders}) OR e.target_id IN ({placeholders})
        """
        params = node_ids + node_ids
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [(
                r["src_id"], r["src_label"],
                r["edge_type"],
                r["tgt_id"], r["tgt_label"],
                r["edge_props"]
            ) for r in rows]

    def get_stats(self) -> Dict[str, int]:
        with self._get_connection() as conn:
            paper_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE label IN ('Paper', 'UserNote')").fetchone()[0]
            author_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE label = 'Author'").fetchone()[0]
            concept_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE label = 'Concept'").fetchone()[0]
            edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            indexed_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE label IN ('Paper', 'UserNote') AND is_placeholder = 0").fetchone()[0]
            mentioned_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE label IN ('Paper', 'UserNote') AND is_placeholder = 1").fetchone()[0]
            return {
                "papers": paper_count,
                "authors": author_count,
                "concepts": concept_count,
                "edges": edge_count,
                "indexed_papers": indexed_count,
                "mentioned_papers": mentioned_count
            }

    def cleanup_orphaned_concepts(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM nodes
                WHERE label = 'Concept'
                AND NOT EXISTS (
                    SELECT 1 FROM edges WHERE source_id = nodes.id OR target_id = nodes.id
                )
                """
            )
            conn.commit()
            return cursor.rowcount

    def _row_to_paper(self, node_id: str, props: Dict[str, Any]) -> Paper:
        return Paper(
            id=node_id,
            title=props.get("title", ""),
            authors=props.get("authors", []),
            year=props.get("year"),
            doi=props.get("doi"),
            abstract=props.get("abstract"),
            file_path=props.get("file_path"),
            created_at=props.get("created_at"),
            properties=props
        )

    def get_all_nodes(self) -> List[tuple[str, str, str]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT id, label, properties FROM nodes").fetchall()
            return [(r["id"], r["label"], r["properties"]) for r in rows]

    def get_all_edges(self) -> List[tuple[str, str, str, str]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT source_id, target_id, type, properties FROM edges").fetchall()
            return [(r["source_id"], r["target_id"], r["type"], r["properties"]) for r in rows]

    def get_node_by_id(self, node_id: str) -> Optional[tuple[str, str]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT label, properties FROM nodes WHERE id = ?", (node_id,)).fetchone()
            if row:
                return (row["label"], row["properties"])
            return None

    def get_papers_by_author(self, author_id: str) -> List[Paper]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT n.id, n.properties FROM nodes n
                JOIN edges e ON n.id = e.target_id
                WHERE e.source_id = ? AND e.type = 'AUTHORED' AND n.label = 'Paper'
                """,
                (author_id,)
            ).fetchall()
            papers = []
            for r in rows:
                props = json.loads(r["properties"])
                papers.append(self._row_to_paper(r["id"], props))
            return papers

    def get_papers_by_entity(self, entity_id: str, edge_type: str) -> List[Paper]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT n.id, n.properties FROM nodes n
                JOIN edges e ON n.id = e.source_id
                WHERE e.target_id = ? AND e.type = ? AND n.label = 'Paper'
                """,
                (entity_id, edge_type)
            ).fetchall()
            papers = []
            for r in rows:
                props = json.loads(r["properties"])
                papers.append(self._row_to_paper(r["id"], props))
            return papers

    def get_distinct_targets(self, source_ids: List[str], edge_type: str) -> List[tuple[str, str]]:
        if not source_ids:
            return []
        placeholders = ",".join("?" for _ in source_ids)
        query = f"""
            SELECT DISTINCT n.id, n.properties
            FROM nodes n
            JOIN edges e ON n.id = e.target_id
            WHERE e.source_id IN ({placeholders}) AND e.type = ?
        """
        params = list(source_ids) + [edge_type]
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [(r["id"], r["properties"]) for r in rows]

    def search_papers_by_title(self, query: str, limit: int = 20) -> List[Paper]:
        with self._get_connection() as conn:
            q_like = f"%{query}%"
            rows = conn.execute(
                "SELECT id, properties FROM nodes WHERE label = 'Paper' AND title LIKE ? LIMIT ?",
                (q_like, limit)
            ).fetchall()
            papers = []
            for r in rows:
                props = json.loads(r["properties"])
                papers.append(self._row_to_paper(r["id"], props))
            return papers

    def get_notes(self) -> List[Paper]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, properties FROM nodes WHERE label = 'UserNote' OR (label = 'Paper' AND source_type = 'note')"
            ).fetchall()
            papers = []
            for r in rows:
                props = json.loads(r["properties"])
                papers.append(self._row_to_paper(r["id"], props))
            # Sort by created_at descending
            papers.sort(key=lambda p: p.created_at or "", reverse=True)
            return papers

    def delete_edges_by_target(self, target_id: str, edge_types: List[str]) -> None:
        """Deletes edges pointing TO target_id with one of the given edge types."""
        if not edge_types:
            return
        placeholders = ", ".join("?" * len(edge_types))
        with self._get_connection() as conn:
            conn.execute(
                f"DELETE FROM edges WHERE target_id = ? AND type IN ({placeholders})",
                [target_id] + list(edge_types),
            )
            conn.commit()

    def delete_edges_by_source(self, source_id: str, edge_types: List[str]) -> None:
        """Deletes edges originating FROM source_id with one of the given edge types."""
        if not edge_types:
            return
        placeholders = ", ".join("?" * len(edge_types))
        with self._get_connection() as conn:
            conn.execute(
                f"DELETE FROM edges WHERE source_id = ? AND type IN ({placeholders})",
                [source_id] + list(edge_types),
            )
            conn.commit()

    def delete_node(self, node_id: str) -> None:
        """Deletes the node and cascades to edges/chunks via foreign keys."""
        with self._get_connection() as conn:
            # Delete associated Chunk nodes first to trigger cascade on edges table
            self.delete_chunk_nodes_for_paper(node_id)
            conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            conn.commit()

    def get_paper_ids(self) -> List[str]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT id FROM nodes WHERE label IN ('Paper', 'UserNote')").fetchall()
            return [r["id"] for r in rows]

    def get_non_placeholder_paper_ids(self) -> List[str]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT id FROM nodes WHERE label IN ('Paper', 'UserNote') AND is_placeholder = 0").fetchall()
            return [r["id"] for r in rows]

    def get_paper_source_types(self) -> Dict[str, str]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT id, properties FROM nodes WHERE label IN ('Paper', 'UserNote')").fetchall()
            res = {}
            for r in rows:
                try:
                    props = json.loads(r["properties"] or "{}")
                    stype = props.get("source_type")
                    if stype:
                        res[r["id"]] = stype
                except Exception:
                    pass
            return res

    def get_browse_rows(self, table: str, page: int, limit: int, search_query: Optional[str] = None) -> List[Dict[str, Any]]:
        off = (page - 1) * limit
        with self._get_connection() as conn:
            if not search_query:
                if table == "documents":
                    rows = conn.execute(
                        "SELECT id, properties FROM nodes WHERE label IN ('Paper', 'UserNote') LIMIT ? OFFSET ?",
                        (limit, off)
                    ).fetchall()
                elif table == "authors":
                    rows = conn.execute(
                        """SELECT n.id, n.properties, COUNT(e.source_id) as papers_count
                           FROM nodes n
                           LEFT JOIN edges e ON n.id = e.source_id AND e.type = 'AUTHORED'
                           WHERE n.label = 'Author'
                           GROUP BY n.id
                           ORDER BY papers_count DESC LIMIT ? OFFSET ?""",
                        (limit, off)
                    ).fetchall()
                else:  # concepts
                    rows = conn.execute(
                        """SELECT n.id, n.properties, COUNT(e.target_id) as degree
                           FROM nodes n
                           LEFT JOIN edges e ON n.id = e.target_id AND e.type = 'MENTIONS_CONCEPT'
                           WHERE n.label = 'Concept'
                           GROUP BY n.id
                           ORDER BY degree DESC LIMIT ? OFFSET ?""",
                        (limit, off)
                    ).fetchall()
            else:
                like_pat = f"%{search_query}%"
                if table == "documents":
                    import re
                    words = re.findall(r'\w+', search_query)
                    fts_query = " OR ".join(words) if words else ""
                    if fts_query:
                        rows = conn.execute(
                            """SELECT id, properties FROM nodes
                               WHERE label IN ('Paper', 'UserNote')
                               AND (
                                   id IN (
                                       SELECT DISTINCT c.paper_id 
                                       FROM chunks_fts f
                                       JOIN chunks c ON c.id = f.id
                                       WHERE chunks_fts MATCH ?
                                   )
                                   OR properties LIKE ?
                               )
                               LIMIT ? OFFSET ?""",
                            (fts_query, like_pat, limit, off)
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            """SELECT id, properties FROM nodes
                               WHERE label IN ('Paper', 'UserNote')
                               AND properties LIKE ?
                               LIMIT ? OFFSET ?""",
                            (like_pat, limit, off)
                        ).fetchall()
                elif table == "authors":
                    rows = conn.execute(
                        """SELECT n.id, n.properties, COUNT(e.source_id) as papers_count
                           FROM nodes n
                           LEFT JOIN edges e ON n.id = e.source_id AND e.type = 'AUTHORED'
                           WHERE n.label = 'Author' AND (n.id LIKE ? OR n.properties LIKE ?)
                           GROUP BY n.id
                           ORDER BY papers_count DESC LIMIT ? OFFSET ?""",
                        (like_pat, like_pat, limit, off)
                    ).fetchall()
                else:  # concepts
                    rows = conn.execute(
                        """SELECT n.id, n.properties, COUNT(e.target_id) as degree
                           FROM nodes n
                           LEFT JOIN edges e ON n.id = e.target_id AND e.type = 'MENTIONS_CONCEPT'
                           WHERE n.label = 'Concept' AND (n.id LIKE ? OR n.properties LIKE ?)
                           GROUP BY n.id
                           ORDER BY degree DESC LIMIT ? OFFSET ?""",
                        (like_pat, like_pat, limit, off)
                    ).fetchall()
            
            return [dict(r) for r in rows]

    def get_browse_count(self, table: str, search_query: Optional[str] = None) -> int:
        with self._get_connection() as conn:
            if not search_query:
                if table == "documents":
                    return conn.execute("SELECT count(*) FROM nodes WHERE label IN ('Paper', 'UserNote')").fetchone()[0]
                label = {"authors": "Author", "concepts": "Concept"}[table]
                return conn.execute("SELECT count(*) FROM nodes WHERE label=?", (label,)).fetchone()[0]
            
            like_pat = f"%{search_query}%"
            if table == "documents":
                import re
                words = re.findall(r'\w+', search_query)
                fts_query = " OR ".join(words) if words else ""
                if fts_query:
                    return conn.execute(
                        """
                        SELECT count(*) FROM nodes
                        WHERE label IN ('Paper', 'UserNote')
                        AND (
                            id IN (
                                SELECT DISTINCT c.paper_id 
                                FROM chunks_fts f
                                JOIN chunks c ON c.id = f.id
                                WHERE chunks_fts MATCH ?
                            )
                            OR properties LIKE ?
                        )
                        """,
                        (fts_query, like_pat)
                    ).fetchone()[0]
                else:
                    return conn.execute(
                        """
                        SELECT count(*) FROM nodes
                        WHERE label IN ('Paper', 'UserNote')
                        AND properties LIKE ?
                        """,
                        (like_pat,)
                    ).fetchone()[0]
            elif table == "authors":
                return conn.execute(
                    "SELECT count(*) FROM nodes WHERE label='Author' AND (id LIKE ? OR properties LIKE ?)",
                    (like_pat, like_pat)
                ).fetchone()[0]
            else: # concepts
                return conn.execute(
                    "SELECT count(*) FROM nodes WHERE label='Concept' AND (id LIKE ? OR properties LIKE ?)",
                    (like_pat, like_pat)
                ).fetchone()[0]

    def update_node_properties(self, node_id: str, properties: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE nodes SET properties=? WHERE id=?",
                (json.dumps(properties, ensure_ascii=False), node_id)
            )
            conn.commit()

    def get_concept_aliases(self) -> Dict[str, str]:
        aliases_map = {}
        with self._get_connection() as conn:
            rows = conn.execute("SELECT id, properties FROM nodes WHERE label = 'Concept'").fetchall()
            for r in rows:
                try:
                    props = json.loads(r["properties"])
                    canonical_name = props.get("name", r["id"])
                    aliases = props.get("aliases") or []
                    for alias in aliases:
                        aliases_map[alias.lower().strip()] = canonical_name
                except Exception:
                    pass
        return aliases_map

    def get_nodes_by_label(self, label: str) -> List[tuple[str, Dict[str, Any]]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT id, properties FROM nodes WHERE label = ?", (label,)).fetchall()
            return [(r["id"], json.loads(r["properties"])) for r in rows]

    def get_node_properties(self, node_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT properties FROM nodes WHERE id = ?", (node_id,)).fetchone()
            if row:
                return json.loads(row["properties"])
            return None

    def get_papers_mentioning_concepts(self, concept_ids: List[str]) -> List[Tuple[str, str]]:
        if not concept_ids:
            return []
        placeholders = ",".join("?" for _ in concept_ids)
        query = f"""
            SELECT DISTINCT n.id, n.title
            FROM nodes n
            JOIN edges e ON n.id = e.source_id
            WHERE e.target_id IN ({placeholders}) AND e.type = 'MENTIONS_CONCEPT' AND n.label = 'Paper'
        """
        with self._get_connection() as conn:
            rows = conn.execute(query, concept_ids).fetchall()
            return [(r["id"], r["title"] or "") for r in rows]

    def get_concepts_for_papers(self, paper_ids: List[str]) -> List[Tuple[str, str, str]]:
        if not paper_ids:
            return []
        placeholders = ",".join("?" for _ in paper_ids)
        query = f"""
            SELECT e.source_id AS paper_id, e.target_id AS concept_id, n.title AS concept_name
            FROM edges e
            JOIN nodes n ON e.target_id = n.id
            WHERE e.source_id IN ({placeholders}) AND e.type = 'MENTIONS_CONCEPT' AND n.label = 'Concept'
        """
        with self._get_connection() as conn:
            rows = conn.execute(query, paper_ids).fetchall()
            return [(r["paper_id"], r["concept_id"], r["concept_name"] or "") for r in rows]

    def get_concept_document_frequencies(self, concept_ids: List[str]) -> Dict[str, int]:
        if not concept_ids:
            return {}
        freqs = {cid: 0 for cid in concept_ids}
        placeholders = ",".join("?" for _ in concept_ids)
        query = f"""
            SELECT e.target_id AS concept_id, COUNT(DISTINCT e.source_id) AS doc_freq
            FROM edges e
            JOIN nodes n ON e.source_id = n.id
            WHERE e.target_id IN ({placeholders}) AND e.type = 'MENTIONS_CONCEPT' AND n.label = 'Paper'
            GROUP BY e.target_id
        """
        with self._get_connection() as conn:
            rows = conn.execute(query, concept_ids).fetchall()
            for r in rows:
                freqs[r["concept_id"]] = r["doc_freq"]
        return freqs

    def get_total_paper_count(self) -> int:
        query = "SELECT COUNT(*) FROM nodes WHERE label = 'Paper'"
        with self._get_connection() as conn:
            row = conn.execute(query).fetchone()
            return row[0] if row else 0

    def get_citation_neighbors(self, paper_ids: List[str]) -> List[Tuple[str, str, str, str]]:
        if not paper_ids:
            return []
        placeholders = ",".join("?" for _ in paper_ids)
        query = f"""
            SELECT 
                e.source_id AS seed_id, 
                e.target_id AS candidate_id, 
                'seed_cites_candidate' AS direction,
                n.title AS candidate_title
            FROM edges e
            JOIN nodes n ON e.target_id = n.id
            WHERE e.source_id IN ({placeholders}) AND e.type = 'CITES' AND n.label = 'Paper'
            
            UNION ALL
            
            SELECT 
                e.target_id AS seed_id, 
                e.source_id AS candidate_id, 
                'candidate_cites_seed' AS direction,
                n.title AS candidate_title
            FROM edges e
            JOIN nodes n ON e.source_id = n.id
            WHERE e.target_id IN ({placeholders}) AND e.type = 'CITES' AND n.label = 'Paper'
        """
        params = paper_ids + paper_ids
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [(r["seed_id"], r["candidate_id"], r["direction"], r["candidate_title"] or "") for r in rows]

    def search_chunks_within_papers(self, query_embedding: List[float], paper_ids: List[str], limit_per_paper: int = 1) -> List[Tuple[Chunk, float]]:
        if not paper_ids:
            return []
            
        if not query_embedding:
            import logging
            logging.getLogger(__name__).warning("Query embedding is unavailable for search_chunks_within_papers. Falling back to default chunk retrieval with similarity 0.0.")
            query_embedding = [0.0] * 384
        
        placeholders = ",".join("?" for _ in paper_ids)
        query = f"""
            SELECT c.id, c.paper_id, c.text_content, c.page_number, c.embedding, c.id_hash, c.parent_id, p.text_content AS parent_text
            FROM chunks c
            LEFT JOIN parent_chunks p ON c.parent_id = p.id
            WHERE c.paper_id IN ({placeholders})
        """
        
        with self._get_connection() as conn:
            rows = conn.execute(query, paper_ids).fetchall()
            
        import numpy as np
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
            
        chunks_by_paper = {}
        for r in rows:
            paper_id = r["paper_id"]
            emb_blob = r["embedding"]
            if not emb_blob:
                continue
            emb_array = np.frombuffer(emb_blob, dtype=np.float32)
            emb_norm = np.linalg.norm(emb_array)
            if emb_norm == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(q_vec, emb_array) / (q_norm * emb_norm))
                
            chunk = Chunk(
                id=r["id"],
                paper_id=paper_id,
                text_content=r["text_content"],
                page_number=r["page_number"],
                embedding=emb_array.tolist(),
                parent_id=r["parent_id"],
                parent_text=r["parent_text"]
            )
            
            if paper_id not in chunks_by_paper:
                chunks_by_paper[paper_id] = []
            chunks_by_paper[paper_id].append((chunk, similarity))
            
        results = []
        for paper_id, chunk_sims in chunks_by_paper.items():
            chunk_sims.sort(key=lambda x: x[1], reverse=True)
            results.extend(chunk_sims[:limit_per_paper])
            
        return results

    def get_neighbor_papers(self, seed_paper_ids: List[str], order: int = 2, allowed_edge_types: List[str] = None) -> List[str]:
        if not seed_paper_ids:
            return []
        neighbor_paper_ids = set()
        for pid in seed_paper_ids:
            neighbors = self.get_neighbors(pid, max_depth=order, allowed_edge_types=allowed_edge_types)
            for src_id, src_label, _, tgt_id, tgt_label, _ in neighbors:
                if src_label in ("Paper", "UserNote") and src_id not in seed_paper_ids:
                    neighbor_paper_ids.add(src_id)
                if tgt_label in ("Paper", "UserNote") and tgt_id not in seed_paper_ids:
                    neighbor_paper_ids.add(tgt_id)
        return list(neighbor_paper_ids)

    def resolve_graph_nodes_to_local_papers(self, node_ids: List[str]) -> List[ResolvedPaperNode]:
        if not node_ids:
            return []
        
        # 1. Query nodes table
        placeholders = ",".join("?" for _ in node_ids)
        nodes_query = f"SELECT id, label, is_placeholder FROM nodes WHERE id IN ({placeholders})"
        
        node_map = {}
        with self._get_connection() as conn:
            rows = conn.execute(nodes_query, node_ids).fetchall()
            for r in rows:
                node_map[r["id"]] = {
                    "label": r["label"],
                    "is_placeholder": bool(r["is_placeholder"])
                }
                
        # 2. Query chunks count
        chunks_query = f"SELECT paper_id, COUNT(*) as cnt FROM chunks WHERE paper_id IN ({placeholders}) GROUP BY paper_id"
        chunks_map = {}
        with self._get_connection() as conn:
            rows = conn.execute(chunks_query, node_ids).fetchall()
            for r in rows:
                chunks_map[r["paper_id"]] = r["cnt"]
                
        # 3. Query edge type for relation
        edges_query = f"SELECT source_id, target_id, type FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})"
        edge_map = {}
        with self._get_connection() as conn:
            rows = conn.execute(edges_query, node_ids + node_ids).fetchall()
            for r in rows:
                sid, tid, etype = r["source_id"], r["target_id"], r["type"]
                if sid in node_ids:
                    edge_map[sid] = etype
                if tid in node_ids:
                    edge_map[tid] = etype
        resolved = []
        for nid in node_ids:
            info = node_map.get(nid)
            exists = info is not None
            node_type = info["label"] if exists else None
            is_placeholder = info["is_placeholder"] if exists else False
            
            is_paper = node_type in ("Paper", "UserNote")
            canonical_paper_id = nid if is_paper else None
            
            chunks_count = chunks_map.get(nid, 0)
            source_relation_type = edge_map.get(nid, None)
            
            resolved.append(ResolvedPaperNode(
                original_node_id=nid,
                canonical_paper_id=canonical_paper_id,
                node_type=node_type,
                exists_in_papers_table=is_paper and exists,
                chunks_count=chunks_count,
                is_placeholder=is_placeholder,
                source_relation_type=source_relation_type
            ))
        return resolved

    def chunks_count(self, paper_id: str) -> int:
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM chunks WHERE paper_id = ?", (paper_id,)).fetchone()
            return row[0] if row else 0

    def has_chunks(self, paper_id: str) -> bool:
        return self.chunks_count(paper_id) > 0

    def is_local_indexed_paper(self, paper_id: str) -> bool:
        return self.has_chunks(paper_id)

    def is_bridge_only_paper(self, paper_id: str) -> bool:
        with self._get_connection() as conn:
            row = conn.execute("SELECT label FROM nodes WHERE id = ?", (paper_id,)).fetchone()
            if not row or row[0] not in ("Paper", "UserNote"):
                return False
        return self.chunks_count(paper_id) == 0

    def is_retrieval_candidate_paper(self, paper_id: str) -> bool:
        return self.is_local_indexed_paper(paper_id)

    def get_derived_bridge_edges(self, seed_paper_ids: List[str]) -> List[Dict[str, Any]]:
        if not seed_paper_ids:
            return []
        placeholders = ",".join("?" for _ in seed_paper_ids)
        query = f"""
            SELECT 
                source_local_paper_id,
                target_local_paper_id,
                bridge_node_id,
                bridge_relation_type,
                source_relation_type,
                target_relation_type
            FROM derived_local_to_local_bridge_edges
            WHERE source_local_paper_id IN ({placeholders})
        """
        with self._get_connection() as conn:
            rows = conn.execute(query, seed_paper_ids).fetchall()
            return [dict(r) for r in rows]

    def get_chunks_count_by_paper_ids(self, paper_ids: List[str]) -> Dict[str, int]:
        if not paper_ids:
            return {}
        placeholders = ",".join("?" for _ in paper_ids)
        query = f"SELECT paper_id, COUNT(*) as cnt FROM chunks WHERE paper_id IN ({placeholders}) GROUP BY paper_id"
        res = {}
        with self._get_connection() as conn:
            rows = conn.execute(query, paper_ids).fetchall()
            for r in rows:
                res[r["paper_id"]] = r["cnt"]
        return res

    def filter_papers_with_chunks(self, paper_ids: List[str]) -> List[str]:
        if not paper_ids:
            return []
        counts = self.get_chunks_count_by_paper_ids(paper_ids)
        return [pid for pid in paper_ids if counts.get(pid, 0) > 0]

    def count_total_local_papers(self) -> int:
        query = "SELECT COUNT(*) FROM nodes WHERE label = 'Paper' AND is_placeholder = 0"
        with self._get_connection() as conn:
            row = conn.execute(query).fetchone()
            return row[0] if row else 0

    def get_concept_idf(self, concept_ids: List[str]) -> Dict[str, float]:
        if not concept_ids:
            return {}
        import math
        total_papers = self.count_total_local_papers()
        doc_freqs = self.get_concept_document_frequencies(concept_ids)
        idfs = {}
        for c in concept_ids:
            df = doc_freqs.get(c, 0)
            idfs[c] = math.log((1 + total_papers) / (1 + df))
        return idfs

    def get_chunks_count_by_paper_ids(self, paper_ids: List[str]) -> Dict[str, int]:
        if not paper_ids:
            return {}
        placeholders = ",".join("?" for _ in paper_ids)
        query = f"SELECT paper_id, COUNT(*) as cnt FROM chunks WHERE paper_id IN ({placeholders}) GROUP BY paper_id"
        res = {}
        with self._get_connection() as conn:
            rows = conn.execute(query, paper_ids).fetchall()
            for r in rows:
                res[r["paper_id"]] = r["cnt"]
        return res

    def filter_papers_with_chunks(self, paper_ids: List[str]) -> List[str]:
        if not paper_ids:
            return []
        counts = self.get_chunks_count_by_paper_ids(paper_ids)
        return [pid for pid in paper_ids if counts.get(pid, 0) > 0]

    def count_total_local_papers(self) -> int:
        query = "SELECT COUNT(*) FROM nodes WHERE label = 'Paper' AND is_placeholder = 0"
        with self._get_connection() as conn:
            row = conn.execute(query).fetchone()
            return row[0] if row else 0

    def get_concept_idf(self, concept_ids: List[str]) -> Dict[str, float]:
        if not concept_ids:
            return {}
        import math
        total_papers = self.count_total_local_papers()
        doc_freqs = self.get_concept_document_frequencies(concept_ids)
        idfs = {}
        for c in concept_ids:
            df = doc_freqs.get(c, 0)
            idfs[c] = math.log((1 + total_papers) / (1 + df))
        return idfs

    def find_paper_by_arxiv(self, arxiv_id: str) -> Optional[Paper]:
        if not arxiv_id:
            return None
        # Clean arxiv ID prefix
        arxiv_clean = arxiv_id.lower().strip()
        if arxiv_clean.startswith("arxiv:"):
            arxiv_clean = arxiv_clean[6:]
        # Look in properties JSON for arxiv_id or arxiv
        with self._get_connection() as conn:
            rows = conn.execute("SELECT id, properties FROM nodes WHERE label IN ('Paper', 'UserNote')").fetchall()
            for row in rows:
                props = json.loads(row["properties"])
                p_arxiv = props.get("arxiv_id") or props.get("arxiv")
                if p_arxiv:
                    p_arxiv_clean = str(p_arxiv).lower().strip()
                    if p_arxiv_clean.startswith("arxiv:"):
                        p_arxiv_clean = p_arxiv_clean[6:]
                    if p_arxiv_clean == arxiv_clean:
                        return Paper(
                            id=row["id"],
                            title=props.get("title", ""),
                            authors=props.get("authors", []),
                            year=props.get("year"),
                            doi=props.get("doi"),
                            abstract=props.get("abstract"),
                            file_path=props.get("file_path"),
                            created_at=props.get("created_at"),
                            properties=props
                        )
        return None


    def find_paper_by_url(self, url: str) -> Optional[Paper]:
        if not url:
            return None
        url_clean = url.lower().strip()
        with self._get_connection() as conn:
            rows = conn.execute("SELECT id, properties FROM nodes WHERE label IN ('Paper', 'UserNote')").fetchall()
            for row in rows:
                props = json.loads(row["properties"])
                p_url = props.get("url")
                if p_url and str(p_url).lower().strip() == url_clean:
                    return Paper(
                        id=row["id"],
                        title=props.get("title", ""),
                        authors=props.get("authors", []),
                        year=props.get("year"),
                        doi=props.get("doi"),
                        abstract=props.get("abstract"),
                        file_path=props.get("file_path"),
                        created_at=props.get("created_at"),
                        properties=props
                    )
        return None

    def get_non_placeholder_papers(self) -> List[Paper]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT id, properties FROM nodes WHERE label IN ('Paper', 'UserNote') AND is_placeholder = 0").fetchall()
            papers = []
            for row in rows:
                props = json.loads(row["properties"])
                papers.append(Paper(
                    id=row["id"],
                    title=props.get("title", ""),
                    authors=props.get("authors", []),
                    year=props.get("year"),
                    doi=props.get("doi"),
                    abstract=props.get("abstract"),
                    file_path=props.get("file_path"),
                    created_at=props.get("created_at"),
                    properties=props
                ))
            return papers

    def save_reference_corpus_stats(self, stats: List[Tuple[str, int, float, int, str]]) -> None:
        if not stats:
            return
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO reference_corpus_stats (work_id, df, idf, n_local_papers, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(work_id) DO UPDATE SET
                    df = excluded.df,
                    idf = excluded.idf,
                    n_local_papers = excluded.n_local_papers,
                    updated_at = excluded.updated_at
                """,
                stats
            )
            conn.commit()

    def save_paper_reference_vectors(self, vectors: List[Tuple[str, str, float]]) -> None:
        if not vectors:
            return
        with self._get_connection() as conn:
            # First delete vectors for the papers we are inserting
            paper_ids = list({v[0] for v in vectors})
            placeholders = ",".join("?" for _ in paper_ids)
            conn.execute(f"DELETE FROM paper_reference_vector WHERE paper_id IN ({placeholders})", paper_ids)
            conn.executemany(
                """
                INSERT OR REPLACE INTO paper_reference_vector (paper_id, work_id, weight)
                VALUES (?, ?, ?)
                """,
                vectors
            )
            conn.commit()

    def save_chunk_reference_mentions(self, mentions: List[Tuple[str, str, str, Optional[str], Optional[str], Optional[int], Optional[str], Optional[str]]]) -> None:
        if not mentions:
            return
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO chunk_reference_mentions (chunk_id, paper_id, work_id, citation_marker, context, page_number, section, raw_reference)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id, work_id) DO UPDATE SET
                    citation_marker = excluded.citation_marker,
                    context = excluded.context,
                    page_number = excluded.page_number,
                    section = excluded.section,
                    raw_reference = excluded.raw_reference
                """,
                mentions
            )
            conn.commit()

    def delete_derived_edges_by_types(self, edge_types: List[str]) -> None:
        if not edge_types:
            return
        placeholders = ",".join("?" for _ in edge_types)
        with self._get_connection() as conn:
            conn.execute(f"DELETE FROM edges WHERE type IN ({placeholders})", edge_types)
            conn.commit()

    def delete_derived_edges_for_papers(self, paper_ids: List[str], edge_types: List[str]) -> None:
        if not paper_ids or not edge_types:
            return
        placeholders_papers = ",".join("?" for _ in paper_ids)
        placeholders_types = ",".join("?" for _ in edge_types)
        with self._get_connection() as conn:
            conn.execute(f"DELETE FROM edges WHERE type IN ({placeholders_types}) AND (source_id IN ({placeholders_papers}) OR target_id IN ({placeholders_papers}))", edge_types + paper_ids + paper_ids)
            conn.commit()

    def get_reference_corpus_stats(self) -> Dict[str, Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT work_id, df, idf, n_local_papers, updated_at FROM reference_corpus_stats").fetchall()
            res = {}
            for r in rows:
                res[r["work_id"]] = {
                    "df": r["df"],
                    "idf": r["idf"],
                    "n_local_papers": r["n_local_papers"],
                    "updated_at": r["updated_at"]
                }
            return res

    def get_paper_reference_vectors(self) -> Dict[str, Dict[str, float]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT paper_id, work_id, weight FROM paper_reference_vector").fetchall()
            res = {}
            for r in rows:
                pid = r["paper_id"]
                if pid not in res:
                    res[pid] = {}
                res[pid][r["work_id"]] = r["weight"]
            return res

    def get_chunk_reference_mentions(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT chunk_id, paper_id, work_id, citation_marker, context, page_number, section, raw_reference FROM chunk_reference_mentions").fetchall()
            return [dict(r) for r in rows]

    def remap_external_work_to_local_paper(self, ext_work_id: str, local_paper_id: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE edges SET target_id = ? WHERE target_id = ? AND type IN ('CITES', 'CITES_IN_CONTEXT')",
                (local_paper_id, ext_work_id)
            )
            conn.execute(
                "UPDATE chunk_reference_mentions SET work_id = ? WHERE work_id = ?",
                (local_paper_id, ext_work_id)
            )
            conn.execute(
                "DELETE FROM nodes WHERE id = ? AND label = 'ExternalWork'",
                (ext_work_id,)
            )
            conn.commit()

    def delete_chunk_reference_mentions_for_paper(self, paper_id: str) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM chunk_reference_mentions WHERE paper_id = ?", (paper_id,))
            conn.commit()

    def delete_chunk_nodes_for_paper(self, paper_id: str) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                DELETE FROM nodes 
                WHERE label = 'Chunk' 
                  AND json_extract(properties, '$.paper_id') = ?
            """, (paper_id,))
            conn.commit()

class SQLiteVectorRepository(VectorRepository):
    def __init__(self, db_path: str, read_only: bool = False):
        self.db_path = db_path
        self.read_only = read_only
        if not self.read_only:
            self._init_db()
        self._usearch_index = None

    def _get_connection(self):
        if self.read_only:
            uri = f"file:{os.path.abspath(self.db_path)}?mode=ro&immutable=1"
            conn = sqlite3.connect(uri, uri=True, timeout=30.0)
        else:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA foreign_keys = ON;")
        if not self.read_only:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA mmap_size = 268435456;")
        conn.execute("PRAGMA cache_size = -64000;")
        conn.row_factory = sqlite3.Row
        return conn

    def _get_index(self, ndim: int = 384):
        if self._usearch_index is not None:
            if getattr(self._usearch_index, "ndim", None) == ndim:
                return self._usearch_index
            else:
                self._usearch_index = None
        
        from usearch.index import Index
        self._usearch_index = Index(ndim=ndim, metric="cos")
        self.index_path = None if self.db_path == ":memory:" else self.db_path.replace(".db", ".usearch")
        
        if self.index_path and os.path.exists(self.index_path):
            try:
                self._usearch_index.load(self.index_path)
                if getattr(self._usearch_index, "ndim", None) != ndim:
                    self._usearch_index = Index(ndim=ndim, metric="cos")
            except Exception:
                self._usearch_index = Index(ndim=ndim, metric="cos")

        if self.read_only and (not self.index_path or not os.path.exists(self.index_path) or len(self._usearch_index) == 0):
            raise RuntimeError("Read-only vector repository requires a validated persisted USEARCH sidecar")
                
        # Self-healing: if SQLite has chunks, but USearch is empty, rebuild the index
        if len(self._usearch_index) == 0:
            try:
                with self._get_connection() as conn:
                    rows = conn.execute("SELECT id, embedding FROM chunks").fetchall()
                if rows:
                    for r in rows:
                        key = stable_hash(r["id"])
                        if key not in self._usearch_index:
                            emb_array = np.frombuffer(r["embedding"], dtype=np.float32)
                            # Ensure dimension matches before adding
                            if len(emb_array) == ndim:
                                self._usearch_index.add(key, emb_array)
                    if self.index_path:
                        self._usearch_index.save(self.index_path)
            except Exception:
                pass
                
        return self._usearch_index

    def _init_db(self):
        with self._get_connection() as conn:
            # Create parent_chunks table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS parent_chunks (
                id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                text_content TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                FOREIGN KEY (paper_id) REFERENCES nodes(id) ON DELETE CASCADE
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_parent_chunks_paper ON parent_chunks(paper_id);")

            conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                text_content TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                parent_id TEXT,
                FOREIGN KEY (paper_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (parent_id) REFERENCES parent_chunks(id) ON DELETE SET NULL
            );
            """)
            # Create FTS5 virtual table for chunks
            conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                id UNINDEXED,
                text_content
            );
            """)
            
            # Create triggers to sync chunks and chunks_fts
            conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_after_insert AFTER INSERT ON chunks BEGIN
                INSERT OR REPLACE INTO chunks_fts(id, text_content) VALUES (new.id, new.text_content);
            END;
            """)
            conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_after_update AFTER UPDATE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE id = old.id;
                INSERT INTO chunks_fts(id, text_content) VALUES (new.id, new.text_content);
            END;
            """)
            conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_after_delete AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE id = old.id;
            END;
            """)
            
            # Populate existing chunks if any are missing in chunks_fts
            conn.execute("""
            INSERT INTO chunks_fts(id, text_content)
            SELECT id, text_content FROM chunks
            WHERE id NOT IN (SELECT id FROM chunks_fts);
            """)

            # Migration check: add columns if not present
            cursor = conn.execute("PRAGMA table_info(chunks);")
            columns = [row["name"] for row in cursor.fetchall()]
            if "id_hash" not in columns:
                try:
                    conn.execute("ALTER TABLE chunks ADD COLUMN id_hash INTEGER;")
                    rows = conn.execute("SELECT id FROM chunks").fetchall()
                    for r in rows:
                        h = stable_hash(r["id"])
                        conn.execute("UPDATE chunks SET id_hash = ? WHERE id = ?", (h, r["id"]))
                    conn.commit()
                except Exception:
                    pass
            if "parent_id" not in columns:
                try:
                    conn.execute("ALTER TABLE chunks ADD COLUMN parent_id TEXT;")
                    conn.commit()
                except Exception:
                    pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(id_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(parent_id);")
            conn.commit()

    def save_chunks(self, chunks: List[Chunk]) -> None:
        with self._get_connection() as conn:
            for chunk in chunks:
                if chunk.embedding is None:
                    continue
                # Ensure parent paper exists
                exists = conn.execute("SELECT 1 FROM nodes WHERE id = ?", (chunk.paper_id,)).fetchone()
                if not exists:
                    conn.execute(
                        "INSERT INTO nodes (id, label, properties) VALUES (?, ?, ?)",
                        (chunk.paper_id, "Paper", json.dumps({"title": chunk.paper_id, "placeholder": True}, ensure_ascii=False))
                    )
                # Save parent chunk if parent_id is set
                if isinstance(chunk.parent_id, str) and isinstance(chunk.parent_text, str):
                    conn.execute(
                        "INSERT OR IGNORE INTO parent_chunks (id, paper_id, text_content, page_number) VALUES (?, ?, ?, ?)",
                        (chunk.parent_id, chunk.paper_id, chunk.parent_text, chunk.page_number)
                    )
                emb_array = np.array(chunk.embedding, dtype=np.float32)
                emb_blob = emb_array.tobytes()
                parent_id_val = chunk.parent_id if isinstance(chunk.parent_id, str) else None
                conn.execute(
                    "INSERT OR REPLACE INTO chunks (id, paper_id, text_content, page_number, embedding, id_hash, parent_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (chunk.id, chunk.paper_id, chunk.text_content, chunk.page_number, emb_blob, stable_hash(chunk.id), parent_id_val)
                )
            conn.commit()

        # Update USearch index
        valid_chunks = [c for c in chunks if c.embedding is not None]
        if valid_chunks:
            ndim = len(valid_chunks[0].embedding)
            index = self._get_index(ndim)
            for chunk in valid_chunks:
                key = stable_hash(chunk.id)
                if key not in index:
                    index.add(key, np.array(chunk.embedding, dtype=np.float32))
            
            self.index_path = None if self.db_path == ":memory:" else self.db_path.replace(".db", ".usearch")
            if self.index_path:
                index.save(self.index_path)

    def save_chunks_bulk(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
            
        valid_chunks = [c for c in chunks if c.embedding is not None]
        if not valid_chunks:
            return
            
        # Compile unique paper IDs we need to ensure exist
        paper_ids = list({c.paper_id for c in valid_chunks})
        
        with self._get_connection() as conn:
            # Ensure parent paper nodes exist (insert placeholders if not present)
            paper_placeholders = [
                (pid, "Paper", json.dumps({"title": pid, "placeholder": True}, ensure_ascii=False))
                for pid in paper_ids
            ]
            conn.executemany(
                "INSERT OR IGNORE INTO nodes (id, label, properties) VALUES (?, ?, ?)",
                paper_placeholders
            )
            
            # Prepare parent chunks insert params
            parent_chunks_params = []
            seen_parent_ids = set()
            for chunk in valid_chunks:
                if isinstance(chunk.parent_id, str) and isinstance(chunk.parent_text, str) and chunk.parent_id not in seen_parent_ids:
                    seen_parent_ids.add(chunk.parent_id)
                    parent_chunks_params.append((
                        chunk.parent_id,
                        chunk.paper_id,
                        chunk.parent_text,
                        chunk.page_number
                    ))
            
            if parent_chunks_params:
                conn.executemany(
                    "INSERT OR IGNORE INTO parent_chunks (id, paper_id, text_content, page_number) VALUES (?, ?, ?, ?)",
                    parent_chunks_params
                )
                
            # Prepare chunk insert params
            params = []
            for chunk in valid_chunks:
                emb_array = np.array(chunk.embedding, dtype=np.float32)
                emb_blob = emb_array.tobytes()
                parent_id_val = chunk.parent_id if isinstance(chunk.parent_id, str) else None
                params.append((
                    chunk.id,
                    chunk.paper_id,
                    chunk.text_content,
                    chunk.page_number,
                    emb_blob,
                    stable_hash(chunk.id),
                    parent_id_val
                ))
                
            conn.executemany(
                "INSERT OR REPLACE INTO chunks (id, paper_id, text_content, page_number, embedding, id_hash, parent_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                params
            )
            conn.commit()
            
        # Update USearch index
        ndim = len(valid_chunks[0].embedding)
        index = self._get_index(ndim)
        for chunk in valid_chunks:
            key = stable_hash(chunk.id)
            if key not in index:
                index.add(key, np.array(chunk.embedding, dtype=np.float32))
                
        self.index_path = None if self.db_path == ":memory:" else self.db_path.replace(".db", ".usearch")
        if self.index_path:
            index.save(self.index_path)

    def get_chunks_for_paper(self, paper_id: str) -> List[Chunk]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.paper_id, c.text_content, c.page_number, c.embedding, c.parent_id, p.text_content AS parent_text
                FROM chunks c
                LEFT JOIN parent_chunks p ON c.parent_id = p.id
                WHERE c.paper_id = ?
                """,
                (paper_id,)
            ).fetchall()
            
            chunks = []
            for r in rows:
                emb_array = np.frombuffer(r["embedding"], dtype=np.float32)
                chunks.append(Chunk(
                    id=r["id"],
                    paper_id=r["paper_id"],
                    text_content=r["text_content"],
                    page_number=r["page_number"],
                    embedding=emb_array.tolist(),
                    parent_id=r["parent_id"],
                    parent_text=r["parent_text"]
                ))
            return chunks

    def get_all_chunks(self) -> List[Chunk]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.paper_id, c.text_content, c.page_number, c.embedding, c.parent_id, p.text_content AS parent_text
                FROM chunks c
                LEFT JOIN parent_chunks p ON c.parent_id = p.id
                """
            ).fetchall()
            
            chunks = []
            for r in rows:
                emb_array = np.frombuffer(r["embedding"], dtype=np.float32)
                chunks.append(Chunk(
                    id=r["id"],
                    paper_id=r["paper_id"],
                    text_content=r["text_content"],
                    page_number=r["page_number"],
                    embedding=emb_array.tolist(),
                    parent_id=r["parent_id"],
                    parent_text=r["parent_text"]
                ))
            return chunks

    def _build_metadata_filters(self, filters: Optional[dict]) -> tuple[str, list]:
        # TODO/WARNING: Filtering using json_extract on n.properties runs without indexes,
        # resulting in a full table scan for each query. This will become a bottleneck as the
        # database grows. In the future, we should extract fields like 'year', 'authors', and
        # 'venue' into separate indexed columns in the nodes/chunks table.
        if not filters:
            return "", []
        clauses = []
        params = []
        if "year_start" in filters and filters["year_start"] is not None:
            clauses.append("CAST(json_extract(n.properties, '$.year') AS INTEGER) >= ?")
            params.append(int(filters["year_start"]))
        if "year_end" in filters and filters["year_end"] is not None:
            clauses.append("CAST(json_extract(n.properties, '$.year') AS INTEGER) <= ?")
            params.append(int(filters["year_end"]))
        if "author" in filters and filters["author"]:
            clauses.append("json_extract(n.properties, '$.authors') LIKE ?")
            params.append(f"%{filters['author']}%")
        if "venue" in filters and filters["venue"]:
            clauses.append("json_extract(n.properties, '$.journal') LIKE ?")
            params.append(f"%{filters['venue']}%")
        if clauses:
            return " AND " + " AND ".join(clauses), params
        return "", []

    def search_similar_chunks(self, query_embedding: List[float], limit: int = 5, filters: Optional[dict] = None) -> List[tuple[Chunk, float]]:
        ndim = len(query_embedding)
        index = self._get_index(ndim)
        
        if len(index) == 0:
            return []
            
        q_vec = np.array(query_embedding, dtype=np.float32)
        # Fetch more candidates to account for deleted/ghost vectors and metadata filtering!
        search_limit = max(limit * 5, 100)
        matches = index.search(q_vec, search_limit)
        
        if len(matches) == 0:
            return []
            
        keys_list = [int(k) for k in matches.keys]
        placeholders = ",".join("?" for _ in keys_list)
        
        filter_sql, filter_params = self._build_metadata_filters(filters)
        query = f"""
            SELECT c.id, c.paper_id, c.text_content, c.page_number, c.embedding, c.id_hash, c.parent_id, p.text_content AS parent_text
            FROM chunks c
            JOIN nodes n ON n.id = c.paper_id
            LEFT JOIN parent_chunks p ON c.parent_id = p.id
            WHERE c.id_hash IN ({placeholders}) {filter_sql}
        """
        
        with self._get_connection() as conn:
            rows = conn.execute(query, keys_list + filter_params).fetchall()
            
        key_to_dist = {int(k): float(d) for k, d in zip(matches.keys, matches.distances)}
        
        results = []
        for r in rows:
            h = r["id_hash"]
            if h not in key_to_dist:
                continue
            emb_array = np.frombuffer(r["embedding"], dtype=np.float32)
            chunk = Chunk(
                id=r["id"],
                paper_id=r["paper_id"],
                text_content=r["text_content"],
                page_number=r["page_number"],
                embedding=emb_array.tolist(),
                parent_id=r["parent_id"],
                parent_text=r["parent_text"]
            )
            dist = key_to_dist[h]
            similarity = 1.0 - dist
            results.append((chunk, similarity))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def search_text_fts5(self, query: str, limit: int = 10, filters: Optional[dict] = None) -> List[tuple[Chunk, float]]:
        import re
        from spacy.lang.ru.stop_words import STOP_WORDS as RU_STOP_WORDS
        from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS
        
        words = re.findall(r'\w+', query)
        if not words:
            return []
            
        stop_words = RU_STOP_WORDS.union(EN_STOP_WORDS)
        filtered_words = [w for w in words if w.lower() not in stop_words]
        if not filtered_words:
            filtered_words = words
            
        fts_query = " OR ".join(filtered_words)
        
        filter_sql, filter_params = self._build_metadata_filters(filters)
        
        with self._get_connection() as conn:
            try:
                # bm25 returns negative values where lower is better.
                # So we sort by bm25(...) ASC and return -bm25(...) as the score.
                query_str = f"""
                    SELECT c.id, c.paper_id, c.text_content, c.page_number, c.embedding, f.score, c.parent_id, p.text_content AS parent_text
                    FROM (
                        SELECT id, -bm25(chunks_fts) as score
                        FROM chunks_fts
                        WHERE chunks_fts MATCH ?
                    ) f
                    JOIN chunks c ON c.id = f.id
                    JOIN nodes n ON n.id = c.paper_id
                    LEFT JOIN parent_chunks p ON c.parent_id = p.id
                    WHERE 1=1 {filter_sql}
                    ORDER BY f.score DESC
                    LIMIT ?
                """
                rows = conn.execute(
                    query_str,
                    [fts_query] + filter_params + [limit]
                ).fetchall()
            except sqlite3.OperationalError:
                return []
                
        results = []
        for r in rows:
            emb_array = np.frombuffer(r["embedding"], dtype=np.float32)
            chunk = Chunk(
                id=r["id"],
                paper_id=r["paper_id"],
                text_content=r["text_content"],
                page_number=r["page_number"],
                embedding=emb_array.tolist(),
                parent_id=r["parent_id"],
                parent_text=r["parent_text"]
            )
            results.append((chunk, float(r["score"])))
        return results

