# Scientific integrity risks

- July DB identity is unknown; counts cannot establish equality.
- Retrieval/generation may use different DBs in one default run.
- Runtime sampling can change population; freeze exact 375.
- Nested OR graph config can leave feature active when component says off.
- graph-neighbor-in-RRF is inactive behind graph master guard.
- Graph candidate counts do not prove prompt inclusion; archived graph chunks never survive final context.
- Shared CE couples final rerank and graph scoring.
- graph_expansion couples static/dynamic context.
- Ontology flag is partial.
- Historical top-k entropy is truncated/renormalized, not full entropy; current telemetry differs from historical.
- MLX and CUDA numerics can change close rankings/tokens.
- 768D does not prove E5 vector lineage.
- Rebuild may change chunk IDs, FTS tokenization, vectors and graph edges.
- Historical run is ~1.1GB; full logits increase output volume.
- Resume/cache keys must include arm, config, models, index and prompt.
- No cloud Judge during smoke; archive raw evidence for later independent evaluation.

Mitigation: immutable inputs/config, complete traces, observed two-arm effect, fixed entropy definition and verified archive import.
