# RAG Custom Retrieval Benchmark Report

This report displays the retrieval stage performance of your custom configuration compared against active baselines.

## ⚙️ Custom Run Configuration Overrides

### Component Settings (vs B6 Full Pipeline)

| Component | Custom Value | B6 Default | Status |
| :--- | :---: | :---: | :---: |
| `citation_repair` | `False` | `False` | Unchanged |
| `context_trimming` | `True` | `True` | Unchanged |
| `deduplicate_parent_chunks` | `True` | `True` | Unchanged |
| `dense_search` | `True` | `True` | Unchanged |
| `dynamic_alpha_blending` | `False` | `False` | Unchanged |
| `graph_bridge_retrieval` | `False` | `False` | Unchanged |
| `graph_concept_retrieval` | `False` | `False` | Unchanged |
| `graph_expansion` | `True` | `True` | Unchanged |
| `graph_neighbors_in_rrf` | `True` | `True` | Unchanged |
| `graph_ontology_lookup` | `True` | `True` | Unchanged |
| `graph_retrieval_trace` | `True` | `True` | Unchanged |
| `graph_selected_sources_card` | `True` | `True` | Unchanged |
| `hyde` | `False` | `False` | Unchanged |
| `intent_classifier` | `False` | `False` | Unchanged |
| `lexical_search` | `True` | `True` | Unchanged |
| `llm_query_expansion` | `False` | `False` | Unchanged |
| `reranker` | `True` | `True` | Unchanged |
| `rrf` | `True` | `True` | Unchanged |
| `score_blending` | `True` | `True` | Unchanged |
| `shannon_estimator_enabled` | `True` | `True` | Unchanged |

### Hyperparameter Overrides (vs System Defaults)

| Parameter | Custom Value | Default Value | Status |
| :--- | :---: | :---: | :---: |
| *None* | | | | 


## 📊 Retrieval Performance Summary

| Baseline | Success Rate | Mean Recall | Context Precision | Mean Latency |
| :--- | :---: | :---: | :---: | :---: |
| B2 | 100.0% | 1.0000 | 1.0000 | 0.056s |


> [!NOTE]
> - **Retrieval Recall**: proportion of expected papers retrieved.
> - **Context Precision**: Mean Average Precision of the retrieved chunks/papers.
