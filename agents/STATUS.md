# Agent status

| role | state | assignment | evidence | blocker |
| --- | --- | --- | --- | --- |
| lead researcher | working | Finish the approved G1 aggregate under the full-horizon annealed-schedule rule | User approved the two-run full-H15 correction on 2026-08-26 | — |
| researcher | assigned | Implement exact full-horizon reuse and two-run correction launcher; do not launch | Reuse only H15 artifacts with stopped_epoch=15; rerun 6L central and 8L second-random under new identities | — |
| optimizer | blocked | Run the current selected-model A100 utilization gate after training capacity returns | Backlog audit found concurrent probes and sub-second multi-GPU dispatch after admission; no serialization defect | Every GPU has foreign compute above the admission threshold |
| refactorer | available | — | — | — |
| reviewer | available | — | Recovery-wave correction lineage is fail-closed; focused candidate tests pass | — |
| g2 eSASRec researcher | reviewing | Present the revised metric-first RQ2 tables for user acceptance | Seven generated two-row tables; 33 focused tests pass; blind review found no issues | — |
| axis auditor | available | — | Two layers with attention window 50 cap usable history at 99 tokens; raw runs remain ineligible for reporting | — |
