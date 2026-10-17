# Database author decision

**Status:** author-selected new baseline.

**Selected candidate:** `external_db` (`c7fdd7c6fed69b788563896efaa8cbfe204216d6ff82043ff3a7414267aba8f0`).
The decision record is `baseline_decision.json`.

**Evidence:** the July database identity is unproven. Available October B2 saved
retrieval chunks match both candidates exactly by chunk ID, paper ID and text hash
(11/11 chunks). No July canonical retrieval file is present in the checkout.

**Reason:** the external candidate contains 300 PubMedQA notes and a broader graph
with citation and concept edges. This is a new-corpus design choice, not a
reconstruction of the July database.

**Alternative:** `repo_db` contains 300 document IDs and 2,331 chunks with only
`HAS_TAG` graph edges and seven concepts. It is the narrower source-corpus option.

All new baseline arms must use this exact snapshot. Do not mix July scores with
new-baseline runs. The one-query smoke is still gated on restoring the canonical
July 375-query artifact so the actual historical question can be frozen first.
