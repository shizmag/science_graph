# CUDA end-to-end acceptance plan

Future gate, not completed execution. Run one real frozen query on restored snapshot. Require:
1. Automatic asset manifest with revisions, file hashes, image digest and acquisition status.
2. SQLite restore/integrity/count checks, vector dimensions and target-built USEARCH.
3. Frozen query ID/text/hash and exact E5 revision/prefix/pooling/truncation/dtype/vector.
4. FTS and dense retrieval executed with candidate IDs/scores; fusion recorded.
5. Effective graph flags, graph SQL/lookups/candidates and static/dynamic context outputs.
6. CE revision, CUDA device placement, pair count/input lengths/scores at every call.
7. Generator revision/tokenizer, CUDA placement, GPU, dtype, prompt hash, greedy settings, output tokens and stop reason.
8. Real score-derived entropy with explicit definition and selected-token logprob; raw logits or approved sufficient representation.
9. Full raw answer, parsed output, timing/errors/completion and evidence IDs.
10. Checksummed archive imported on author machine with no judge request.

Record GPU name/capability, Torch CUDA build, driver, availability, parameter placement and activity during actual CE/generator inference. Device detection alone is not success.

If graph candidates are absent, pipeline smoke may pass but graph-path acceptance fails. The query must demonstrate the path when graph behavior is an acceptance requirement.
