# Remaining work for the CUDA portability phase

The local acceptance gate is complete, so remote CUDA portability work can begin:
the exact dataset is frozen, `external_db` is the author-selected new baseline,
coverage is 375/375, and a one-query B6 run retrieved the expected source and
generated an answer. No full CUDA ablation runner or benchmark matrix was started.

## Required for the remote CUDA smoke

1. Transfer and verify the frozen `external_db` snapshot, canonical dataset, and
   smoke package by SHA-256. Rebuild USEARCH on the target machine from persisted
   vectors; do not copy a local binary index as authoritative state.
2. Provision a pinned CUDA/PyTorch runtime and record its versions, driver, GPU,
   and visible devices. The current Cross-Encoder selects MPS or CPU and has no
   CUDA path; implement and verify explicit CUDA selection in the remote phase.
3. Verify the locked generator, E5, and Cross-Encoder assets on the target host,
   then run the same frozen query against the same snapshot identity with no cloud
   judge. Capture retrieval candidates, prompt, output, device evidence, timings,
   telemetry available from that backend, and the portable result archive.
4. Confirm that the target run has nonempty lexical/dense retrieval, the expected
   source article, actual Cross-Encoder scoring, and a persisted local generation
   response. A zero-output process exit is not sufficient.

## Known issues and scope limits

- The user-supplied all-zero/37.566-second report is not present in scoped local
  artifacts, so its query population, source document, routing, and stage timing
  remain unresolved. Obtain its exact report and per-query trace before assigning
  a cause. The verified smoke demonstrates the new baseline works for one
  canonical fixture, not that every report arm or query works.
- Local MLX query expansion failed JSON validation four times and fell back to
  the original question. The B6 retrieval and generation completed, but this
  structured-output issue should be tracked during backend portability.
- The smoke showed graph candidates and intermediate selections, but no graph
  neighbor survived final context and graph text was not inserted into the final
  generator prompt. No counterfactual establishes an indirect ranking effect.
- July database identity is not established by the author's selection of
  `external_db`; future ablation outputs must use the frozen selected snapshot
  and must not be compared as though they reproduce July's historical database.
- Windows/WSL2 behavior and CUDA model execution have not been tested. Build the
  full ablation harness only after the remote one-query CUDA smoke passes.
