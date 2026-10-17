# Database architecture

No PostgreSQL, Qdrant, Neo4j or remote vector service appears in the inspected B6 path. SQLite stores chunks and parents, metadata, float32 vector BLOBs, FTS5, graph nodes/edges and auxiliary tables. Adjacent .usearch file is a cosine vector index. Implementation: SQLiteGraphRepository in back/src/repository/sqlite_impl.py:49-80; SQLiteVectorRepository :1648-1701; FTS5 :1729 onward; vector BLOBs :1871-1900; cosine setting :1671-1673.

Observed SQLite is 3.50.4; user_version 0 and quick_check ok. FTS5 row counts match chunk counts in both inspected databases. The external graph database has 2,700 Concept nodes and 3,814 Paper nodes among its 9,680 nodes; the small database has 7 Concept nodes. Exact stored-vector row counts were not separately isolated from chunk counts, so remain UNKNOWN. No separate DB service or port is required. Semantic foreign-key completeness and stale-record audit were not run. USEARCH cardinality was not verified because package is absent.

Constructors initialize schema/set WAL; USEARCH may rebuild/save. Application open is not guaranteed read-only. Use consistent backup, immutable source and writable clone. Cross-OS sidecar compatibility is unverified; rebuild from stored vector BLOBs using pinned target version. Schema initialization should fail closed on unsupported schema rather than silently mutate.
