# Historical index provenance

## Classification: CURRENT_INDEX_ONLY for accessible data; July identity is INSUFFICIENT_EVIDENCE

July run manifest captures code/model/query count, not SQLite file hash, backup ID, USEARCH checksum, or complete invocation. Config snapshot names external DB path but subprocess logic can split DBs. Current DBs and June backups have distinct counts and no link to run.

External current DB has 22,068 chunks, 9,680 nodes, 11,712 edges; October DB mtime and July sidecar mtime. Timestamp discrepancy suggests a sidecar question but proves neither corruption nor historical relation. Repo DB has 2,331 chunks. June backups have 22,110 chunks and 14,281 nodes. Similar counts are not identity. Saved retrieval traces cannot recreate vectors, corpus, FTS and graph required for new arms.

To establish July snapshot, recover invocation, resolved DB path, consistent DB/WAL snapshot and hash, sidecar version/hash, source corpus/chunk IDs/text, E5 revision/preprocess and graph inputs. If unavailable, approve a newly fingerprinted baseline; do not call it July reproduction.
