# Evaluation Results

## Ablation Table

| System | Unsupported match rate | Hallucinated quote rate | Cost / report |
|---|---|---|---|
| Phase 1 — single call, no RAG | TBD | TBD | TBD |
| + planning + hybrid RAG | TBD | TBD | TBD |
| + evidence validation | TBD | TBD | TBD |
| + reflection | TBD | TBD | TBD |
| + confidence routing | TBD | TBD | TBD |

## Per-stage Model Comparison

| Stage | Calls / report | Cheap model | Strong model |
|---|---|---|---|
| JD → requirements | 1 | TBD | TBD |
| Rerank candidates | N | TBD | TBD |
| Score requirement | N | TBD | TBD |
| Reflection | 1 | TBD | TBD |

## Escalation Precision Curve

Run `python evaluation/sweep_threshold.py <reports.json>` to populate this section.

| Threshold | Escalation rate | Notes |
|---|---|---|
| 0.5 | TBD | |
| 0.6 | TBD | |
| 0.7 | TBD | Default |
| 0.8 | TBD | |
| 0.9 | TBD | |

## Key Metrics

- `hallucinated_quote_rate`: TBD
- `unsupported_match_rate`: TBD
- `retrieval_miss_rate`: TBD
- `reflection_correction_rate`: TBD (raised / lowered split)
- `escalation_rate`: TBD
- `escalation_precision`: TBD
