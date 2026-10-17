# Portable question manifest plan

Extract from archived retrieved_contexts.yaml; cross-check each row against golden YAML and sampled_300.json. Preserve exact IDs, question text, gold values, answerability, source PMID and original order. Add source dataset, source revision, construction category and historical run membership. Proposed fields: query_id, question, gold_answer, answerability_label, source_pmid, source_dataset, source_revision, construction_category, historical_run_membership, row_index.

Validate 375 unique IDs, 300 answerable and 75 negative, exact string crosswalk and one immutable file SHA256. Keep negative empty-string convention or explicitly record normalization. Future startup reads this file directly without sampling. Do not include retrieved context or generated answers.

[UNKNOWN] Original PubMedQA source revision/license and synthetic-negative generation procedure need author/source investigation; record UNKNOWN rather than infer.
