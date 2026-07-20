# Known Failures and Limitations

This document separates observed failures from risks that have not yet been
measured. A measured limitation is more useful than an unsupported claim that
the system works reliably.

## Evaluation Scope

The current baseline was run on July 19, 2026 using five privacy-safe synthetic
resumes. All five cases completed, but this sample is deliberately small and
does not represent real recruiting traffic, unusual resume formats, provider
variance, or demographic fairness. The results are diagnostic, not a production
accuracy claim.

| Metric | Current result |
|---|---:|
| Completed cases | 5/5 |
| Verdict accuracy | 100% |
| Score mean absolute error | 1.12 points |
| Escalation rate | 40% |
| False accepts | 0 |
| False rejects | 0 |
| Hallucinated quote rate | 0% |
| Unsupported positive-match rate | 0% |
| Reflection corrections | 0 raised, 0 lowered |

The raw reports and metric definitions are in `evaluation/results.json` and
`evaluation/run_evaluation.py`.

The initial diagnostic run achieved only 20% exact verdict accuracy. After
aligning the scorer and reflector evidence scales, preserving the JD's original
requirement strength, normalizing presentation-only quote delimiters, and adding
guarded confidence calibration, the same five labels reached 100%. This is a
useful regression result, not evidence of 100% real-world accuracy.

## Observed and Residual Risks

### Raw model confidence is not calibrated

The initial baseline's largest failure was confidence routing. In multiple cases, the
scorer assigned a requirement a score of `1.0`, described the evidence as clear
or explicit, but reported confidence between `0.0` and `0.1`. The default
`CONFIDENCE_FLOOR=0.7` therefore escalated otherwise well-supported matches to
human review.

The application now treats raw model confidence as only one input. After
reflection, `calibrate_confidence()` raises confidence to the routing floor only
when evidence state is valid, retrieval was not flagged as weak, and the
full-resume audit completed. Invalid evidence, weak retrieval, and failed audits
remain uncertain.

This is a guarded decision rule, not probability calibration. Its safety depends
on retrieval-confidence detection and the critic's ability to notice full-resume
evidence, both of which remain imperfect.

The current threshold sweep measures escalation rate only:

| Confidence floor | Escalation rate |
|---:|---:|
| 0.5 | 40% |
| 0.6 | 40% |
| 0.7 | 40% |
| 0.8 | 60% |
| 0.9 | 60% |

It does not yet calculate escalation precision or verdict accuracy at each
threshold, so it cannot identify an optimal floor.

### Weak-versus-absent evidence remains a judgment boundary

The initial run rejected an education-only case that was labeled for review.
The scorer and reflector now consistently treat relevant keyword-only, academic,
or adjacent evidence as weak (`0.25`) rather than absent (`0.0`). That fixed the
synthetic regression, but the distinction still relies on model interpretation
and needs validation on real resumes.

### Reflection recovery is unproven

Across the current five-case run, reflection changed no scores. That is too
little data to claim that reflection reliably recovers retrieval misses or
corrects inflation. The evaluator also does not yet count proposed corrections
that were rejected by the reflector's invariant checks.

### Provider output is not guaranteed to be deterministic

Temperature zero reduces randomness but does not guarantee identical provider
responses. The saved metrics describe one run. Repeated-run variance is not yet
measured, and a five-case result can change sharply when a single verdict moves.

### Evidence validation checks quotation, not meaning

The baseline produced no fabricated quotes or unsupported positive matches,
which confirms the substring guard worked on these five cases. It does not prove
that cited evidence supports the model's interpretation. For example,
`"Kubernetes"` can be a genuine resume substring without demonstrating hands-on
production experience.

## Structural Limitations

### Requirement extraction is the ceiling

If the planner omits, duplicates, or misclassifies a job requirement, every
downstream stage evaluates the wrong plan. There is no independent check of the
planner output against the original job description.

### A retrieval miss can become a rejection

The router cannot distinguish "the resume does not contain this" from "retrieval
did not surface it." Low confidence and unverifiable evidence escalate, but a
confident zero caused by poor retrieval can still reject a candidate. The
current evaluation does not measure how often this happens.

The `LOW_RETRIEVAL_FLOOR` is based on reciprocal-rank-fusion agreement. It does
not directly measure whether the retrieved chunks are semantically relevant.

### Routing thresholds are not calibrated

`REQUIRED_ACCEPT_SCORE=0.75`, `PARTIAL_MATCH_SCORE=0.25`, and
`CONFIDENCE_FLOOR=0.7` remain defaults rather than thresholds selected against a
representative labeled dataset. The five synthetic cases are insufficient for
calibration.

### The overqualified problem

A staff engineer can score highly against a mid-level job description even when
a recruiter would consider the candidate unsuitable. This is a product-policy
question that evidence matching alone cannot answer.

### Scores are not comparable across candidates

The system evaluates one resume against one job description. It has no
cross-candidate calibration, so a score of 80 in one report should not be treated
as equivalent to 80 in another.

## Retrieval and Input Limitations

### Skills-section bias

A dense skills list can outrank experience evidence because it contains more
matching terms. The reranker is prompted to prefer demonstrated capability, but
the baseline is too small to quantify how often it succeeds.

### Short resumes

Resumes with fewer than three or four distinct entries provide little retrieval
diversity. BM25 and cosine similarity can both return rankings even when none
of the chunks are useful.

### Non-standard PDF layouts

Docling provides layout-aware reading order and structure detection, but complex
two-column resumes, decorative templates, and dense tables can still produce
incorrect element ordering or heading associations. These formats are not yet
represented in the synthetic evaluation set.

### OCR quality and cold-start cost

Docling enables OCR for scanned resumes, but recognition accuracy depends on
scan quality, language, fonts, and the available OCR runtime. The first
conversion in a fresh environment may also be slow while Docling initializes
its document models.

## Open Work

- [ ] Expand evaluation to representative anonymized or consented resumes.
- [ ] Calibrate confidence against representative human-reviewed outcomes.
- [ ] Report verdict accuracy and escalation precision at each confidence floor.
- [ ] Calibrate required, partial-match, and confidence thresholds jointly.
- [ ] Measure rejected reflection corrections and their rejection reasons.
- [ ] Measure retrieval-caused rejection separately from genuine qualification gaps.
- [ ] Add two-column, table-based, short, malformed, and image-only PDF cases.
- [ ] Measure provider variance, latency, token usage, and cost per report.
- [ ] Evaluate fairness and disparate error rates before any hiring use.
