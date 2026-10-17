# Snapshot feasibility

Recommended candidate: SQLite online backup API, for example sqlite3.Connection.backup, with writers quiesced or consistent-read semantics. Do not copy live .db alone while WAL may be active. Record DB source, journal mode, schema/counts and hash.

Restore on Linux, validate SQLite/FTS5/schema/quick_check/integrity/counts, vector dimensions and IDs. Rebuild USEARCH sidecar from stored float32 blobs using pinned target version, then compare candidate IDs, scores and ordering on fixed queries. Keep immutable source and run writable clone if app initializes schema or WAL. Record DB and sidecar build identities.

SQLite file is plausibly OS portable subject to FTS and version compatibility; binary USEARCH sidecar portability is unverified. No July snapshot was found. Restore success does not imply historical identity or retrieval equivalence.
