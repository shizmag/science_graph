# Snapshot restore validation plan

Export with SQLite online backup; quiesce writers or use consistent API semantics. Capture schema, SQLite version, journal state, counts and source/export hashes. Never copy active DB alone. Include query manifest, note hashes and model/runtime manifests; exclude credentials.

Restore to clean volume, verify hash and FTS5; run quick_check/integrity_check. Compare chunks, parents, nodes by type, edges by type, FTS rows and vector count. Confirm finite 768-float32 vectors. Rebuild USEARCH and hash output. Compare stable IDs, representative text hashes, vector norms/checksums, graph neighborhoods, and fixed lexical/dense retrieval candidates/scores/order. Use score tolerances for device differences and report ties. Compare selected contexts and prompt hashes. Record pass/fail per layer; database open alone is not retrieval equivalence. Keep source immutable; use a clone for application startup.
