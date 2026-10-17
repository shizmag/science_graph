# Configuration resolution audit

Built-in defaults are in back/src/config.py:11-117 and embedded YAML :221-240,:247-381. External user YAML is ~/.config/pdf-graph-analyzer/config.yaml (:7-9,457-469). RAG_GRAPH_RETRIEVAL can override nested master setting (:958-962). Baseline flags are in back/benchmarks/rag/core/config.py:355-408; runtime mutation and Expander creation in core/generation.py:192-226. Historical snapshot says provider MLX, OCC path, E5, CE, context 12000/output 768; nested graph master/concept/bridge enabled, selected card/trace enabled. Top-level concept/bridge component values are false.

| Flag | Implementation behavior and consequence |
|---|---|
| graph_retrieval | environment/nested accessor in config.py:958-962; master activates service path |
| concept_retrieval / bridge_retrieval | accessors at :981-994 use OR semantics; false baseline value does not defeat nested true. If graph master on and both children false, service auto-enables both at rag_service.py:1617-1626 |
| graph_neighbors_in_rrf | B6 forces true in core/config.py:382-390 but service branch requires graph master false at rag_service.py:1502-1554; inactive in B6 |
| graph_expansion | rag_service.py:400-464,2184-2222 and generation.py:330-385; one flag couples static context and dynamic expander |
| graph_ontology_lookup | rag_service.py:576-595 only controls alias lookup for queries <=2 tokens; separate spaCy concept path :983-1056 remains |
| reranker | B6 forces true; final reranking at rag_service.py:1376,2112; same CE injected into expander at generation.py:207-216 |
| score_blending | scoring path rag_service.py:1391-1592; isolate only with fixed candidates and CE scores |
| context_trimming | context builder rag_service.py:400-464; effect conditional on token budget |
| selected source card | OR accessor and append at rag_service.py:2253-2259; post-generation output, not generated entropy |
| dynamic_alpha_blending | rag_service.py:1567-1579; historical false |
| RRF | rag_service.py:1582-1592; active |
| parent dedup / structured prompt | multiple candidate/context/prompt call sites; requires stage-specific counts and template isolation |
| HyDE | B6 forced false in core/config.py:382-390 |

The resolved effective config must be serialized per arm. Input YAML alone is not evidence of execution. OR accessors and master-child rules mean that some flags cannot be toggled independently.
