# Preserved smoke restore identity diagnosis

The first smoke implementation raised `RuntimeError: Restored smoke database identity differs from the selected snapshot` at stage `snapshot_restore` because it required byte-for-byte equality after a SQLite backup restore.

A read-only diagnostic restore produced SHA-256 `bc54d962762d0ec9c150853f10a4fed7b4cec1fa42cdef064ec2d87a9fb7b79d` from selected snapshot SHA-256 `c7fdd7c6fed69b788563896efaa8cbfe204216d6ff82043ff3a7414267aba8f0`. The restored DB passed SQLite integrity, FTS5, vector validation, and the selected snapshot's logical summary: 22,068 chunks/vectors, 768 dimensions, 346 documents, 2,845 parents, 9,680 graph nodes, 11,712 edges, and matching table, node-label, and edge-type counts.

SQLite's backup restore can change physical page layout while preserving logical contents. The smoke now validates those logical contents, records both hashes, and sends the restored DB's hash to the pipeline. It also checks that rebuilding USEARCH does not change the restored DB and that the selected snapshot file remains unchanged.
