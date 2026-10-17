# Phase 1 — Repository and execution topology

## Repository and run

[VERIFIED_FROM_RUNTIME] One Git checkout at /Users/vladimirkasterin/python/graph, branch main, HEAD f53b1999bf006d826e6284b006bb7081b4581f36. It was clean before this report. Root pyproject.toml includes workspace back and Python >=3.12 (lines 1-18). Dataset, graphs, DBs and models are ignored/untracked, not packaged in Git.

[VERIFIED_FROM_ARTIFACT] July run directory is back/benchmarks/rag/graphs/run_20260729_211324_OCC-RAG-1.7B; manifest commit b4f3322d8cae3395bcbf0b5a1f2e47a904d1591a. Later generation telemetry/resume changes are present. Retrieval/graph implementation in inspected diff was unchanged.

## Executable dependency map

run_pipeline.py CLI/config resolver
-> retrieval subprocess run_custom_retrieve.py
-> dataset loader and query preprocessing
-> RAGService: E5 query embedding, USEARCH and SQLite vector blobs, FTS5, graph lookups, fusion, Cross-Encoder
-> static context/trim and dynamic ExperimentalGraphExpander
-> prompt construction
-> generator and telemetry
-> parse/persist/checkpoint
-> optional external evaluator

| Stage | Code | Input/output and state |
|---|---|---|
| CLI/config | run_pipeline.py main, active DB at line 100; core/config.py:530-565 | resolves path and run directory |
| Retrieval process | run_pipeline.py:342-394, run_custom_retrieve.py:76-110 | dataset to retrieved contexts/traces; SQLite, USEARCH, E5 |
| Baseline | core/config.py:355-408, generation.py:192-226 | B6 flags and expander creation |
| Query prep | rag_service.py:575-595,983-1056; normalization_pipeline.py:52-130 | normalization/concepts; spaCy conditional |
| Lexical | sqlite_impl.py FTS5 creation :1729 onward; rag_service.py:1391 onward | query terms to FTS candidates |
| Dense | vector_search.py:14-123; sqlite_impl.py:1648-1701 | query vector to cosine candidates |
| Graph | rag_service.py:1502-1681 | graph candidates from SQLite |
| Fusion/rank | rag_service.py:1391-1592 | RRF/blend and rank |
| Rerank | rag_service.py:295-306,1376,2112 | CE scores; currently MPS or CPU |
| Context | rag_service.py:400-464,2184-2222; graph_expander.py:41-730 | static and dynamic graph context |
| Generate | mlx_impl.py:557-650; generation.py helpers | historical MLX answer and compact telemetry |
| Parse/eval | run_pipeline.py:478-525 plus evaluator stage | local outputs; optional remote judge, skip it |

## DB path split

[VERIFIED_FROM_CODE] run_pipeline.py resolves active_db at line 100. Retrieval subprocess receives --db-path only if args.db_path exists (:342-394). Generation subprocess is built separately and receives no db-path (:478-525). run_custom_retrieve.py resolves explicit path at :76-80. Default app config points to external ~/.local/share/pdf-graph-analyzer/graph.db (back/src/config.py:7-12,506-507). With default arguments retrieval may select repo-local back/science_graph.db and generation external graph.db. Explicit CLI/config can change this; July invocation is not recorded enough to settle it.

[VERIFIED_FROM_RUNTIME] Repo DB is 14,065,664 bytes with 3,926,936-byte sidecar. External DB is 185,470,976 bytes with 37,180,228-byte sidecar and different counts/graph structure.

## Mutation safety

SQLite repositories initialize schema and WAL (sqlite_impl.py:49-80,1648-1663); USEARCH lazy path can rebuild and save sidecar (:1671-1697). A query may mutate DB/sidecar. No probe was run against the originals.

July manifest records MLX, model paths, output cap 768, context 12000 and 375 rows but omits DB checksum and complete invocation. Paper B5 is internal B6; internal B5 is different.
