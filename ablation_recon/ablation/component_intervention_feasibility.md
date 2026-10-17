# Component intervention feasibility

Reference mapping is Paper B5 = internal B6; code B5 is different. Archived traces have 1,053 graph-neighbor labels, zero concept/bridge candidate counts, zero graph nodes, and no graph chunks surviving final context. Candidate generation alone does not show evidence reached prompt.

Plausible independent main effects after config normalization:
- Lexical off: remove FTS candidates, rerun retrieval/generation.
- Dense off: remove E5 candidates and assert no fallback.
- RRF: change fusion while preserving upstream candidates.
- Score blend: isolate formula with Cross-Encoder scores and candidates held.
- Context trim: meaningful only when budget exceeded; record token counts and removed chunks.
- Source card: post-generation presentation effect, not generated quality/entropy.

Coupled or invalid currently:
- Final reranker and graph expander share CE. Split switches.
- graph_expansion controls static context and dynamic GraphExpander. Split.
- ontology flag controls only alias path; spaCy concept work remains.
- concept/bridge OR leakage and master auto-enable defeat simple component-off.
- parent dedup may happen at multiple stages; expose separately.
- structured prompt shares code with dynamic context; freeze context to test format only.
- graph-neighbor-in-RRF is guarded off when master graph is on.
- historical graph candidates did not survive into prompt.

Test main effects singly and trace them. Retrieval arms can interact with RRF, score blend, graph expansion and trimming; later interaction experiments may be necessary. A toggle with no executed effect is a no-op, not a negative scientific result.
