# Results roundtrip plan

Archive needs query ID; effective arm config; dataset/index/model/tokenizer/template/runtime/image fingerprints; per-stage retrieval candidates/scores and graph lookups; selected contexts; prompt and hash; raw answer; token IDs/text; stop reason; parser output; defined entropy and selected-token logprob; raw logits or approved representation; GPU diagnostics; timing; errors/completion/resume; per-file bytes and SHA256. Never include credentials.

Importer should extract to isolated staging without executing contents, validate schema/path safety/size/hash, compare identities, enforce unique query-arm rows/completion states, validate trace and telemetry, convert to later independent Judge input without calling Judge, preserve original archive and emit import record.

Historical one-run artifact was about 1.1GB with compact telemetry. Full logits may enlarge it; measure one-query package before full run. No importer built.
