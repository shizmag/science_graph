# Cross-Encoder GPU audit

Model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 at revision 1427fd652930e4ba29e8149678df786c240d8825, 470,592,698 bytes, XLM-R classifier, max input512, Apache-2.0 public.

RAGService._get_reranker in back/src/services/rag_service.py:295-306 chooses MPS when available else CPU; no CUDA branch. Final calls occur at :1376 and :2112. B6 generation.py:207-216 injects same instance into ExperimentalGraphExpander; graph_expander.py uses it at :435,:543,:667. Thus the model is shared between final ranking and graph scoring.

Future code must request CUDA explicitly, fail if unavailable, and log model revision/device/dtype, pair count, batch size and encoded lengths at both call sites. Separate final-rerank and graph-scorer switches before ablation. On RTX assert parameter placement and capture CUDA activity around real scoring; compare scores/order with CPU reference. No CUDA use tested.
