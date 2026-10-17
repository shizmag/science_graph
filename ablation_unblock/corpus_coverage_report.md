# Source article coverage in the selected frozen corpus

- Database: `/Users/vladimirkasterin/python/graph/ablation_portability/snapshots/candidates/external_db/v1/database.db`
- SHA-256: `c7fdd7c6fed69b788563896efaa8cbfe204216d6ff82043ff3a7414267aba8f0`
- Chunked documents: 346
- Unique source PMIDs: 300
- Unique source PMIDs represented: 300
- Unique source PMIDs absent: 0

## Per-query source availability

| Population | Queries | Source documents present | Exact paper IDs present | PMID matches |
|---|---:|---:|---:|---:|
| answerable | 300 | 300 | 300 | 300 |
| designated_negative | 75 | 75 | 75 | 75 |
| **All** | **375** | **375** |  |  |

Coverage uses indexed `nodes.id`/`nodes.properties` and requires a matching `chunks.paper_id`; it does not inspect raw article text.

## Missing sources

None. All archived source PMIDs resolve to chunked documents in the selected snapshot.
