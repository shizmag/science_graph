# Frozen B6 smoke fixture

- Selection date: 2026-10-17, before the new model execution.
- Canonical query ID: `pubmedqa_21084567` (row 138 of 375).
- Label: `answerable`.
- Question: Are stroke patients' reports of home blood pressure readings reliable?
- Expected source: PMID `21084567`; paper ID `pubmed_21084567_are_stroke_patients_reports_of_home_blood_pressure_readings_reliable`.
- Exact archived reference: Yes. This group of elderly stroke patients were able to record their BPs reliably at home. Any bias was small and would be unlikely to affect management. Since BP readings in a GP surgery are often a poor indication of true BP, GPs might consider using hypertensive patients' records of home BP monitoring to help guide treatment decisions.
- Selected snapshot: `external_db`, SHA-256 `c7fdd7c6fed69b788563896efaa8cbfe204216d6ff82043ff3a7414267aba8f0`.
- Indexed source coverage: exact paper ID and PMID both present in chunked `nodes`/`chunks` metadata (`{"answerability_label": "answerable", "exact_paper_id_present": true, "expected_paper_ids": ["pubmed_21084567_are_stroke_patients_reports_of_home_blood_pressure_readings_reliable"], "query_id": "pubmedqa_21084567", "source_document_present": true, "source_pmid": "21084567", "source_pmid_present": true}`).

## Selection evidence

The July archived B6 metrics for this query mark graph retrieval enabled and record 2 graph-neighbor chunks retrieved, with 0 graph-neighbor chunks in final context. This identifies an exercised graph candidate pathway while preserving the distinction between a candidate and final generator-visible evidence.

The preserved three-row October retrieval artifact also contains this question. Its B6 record has 20 lexical seeds and 20 dense seeds, 10 candidates before and 5 after reranking, and 5 final chunks. The expected paper is in the saved top-five paper IDs: `True`. This older record is used only to establish fixture suitability; it is not treated as the new run's result or as proof of July database identity.

## Frozen observation plan

Run exactly this frozen record against the selected snapshot. Record separately: graph candidates produced; graph candidates selected into final evidence; graph relationship text inserted into the generator prompt; indirect ranking effects. A graph candidate count alone will not be called graph influence. Do not replace the fixture based on the new generation outcome.

## Observed local B6 smoke

The frozen fixture was run successfully on 2026-10-17. The expected source was
the first of five final evidence chunks; Recall and Context Precision recomputed
from this one expected document were both 1.0. The graph trace listed 322 graph
candidate papers; the stage log selected two neighbors at hop 1 and one at hop
2, but zero graph-neighbor chunks survived into final context. Although the
retrieval artifact contains graph relationship text, the captured final prompt
did not include that graph block. See
[`smoke_execution_report.md`](smoke_execution_report.md) for the run's separate
candidate, selection, prompt, and timing evidence.
