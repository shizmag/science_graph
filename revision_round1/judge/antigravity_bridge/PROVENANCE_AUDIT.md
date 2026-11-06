# R2 provenance audit ledger

The prior `agy_r2_workspace` run was explicitly deleted by the author before this integrity upgrade. Based on the previous audit report, the deleted state contained 200 schema-valid files: 154 matched `evaluate_all.py` heuristic output, 3 had unknown provenance, and 43 were provisionally attributed to the author-labelled Gemini session. These counts are historical observations, not current accepted results. They are not recoverable from the present result directory and are excluded from scientific exports.

The integrity workflow never promotes a record merely because its JSON validates. Author acceptance requires a record hash, reviewer identity, provenance evidence, and grounding review. Suspicious heuristic signatures are flags requiring review, not proof of generation method. The current standalone workspace begins with zero results.
