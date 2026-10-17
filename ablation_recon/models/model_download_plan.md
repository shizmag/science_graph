# Model download plan

Core public checkpoints can be acquired at immutable revisions: OCC-RAG e7d9f8b912dc1ac9a66ab42f0d07a7663d6ed58c; E5 d128750597153bb5987e10b1c3493a34e5a4502a; Cross-Encoder 1427fd652930e4ba29e8149678df786c240d8825. Local cache evidence exists. No full download was attempted. A downloader should pin revision, fetch tokenizer/config/template, resume, verify hashes and persist cache outside image.

spaCy en_core_web_sm is unpinned and may download implicitly; pin/stage or use an explicit tested fallback. NER/PDF assets are ingestion-only unless rebuilding. No private custom generator was found.

Author-side staging remains required for the 375 manifest, accepted SQLite snapshot, notes/license metadata and possibly raw source provenance. These have no verified remote source. Keep credentials separate from manifests and logs.
