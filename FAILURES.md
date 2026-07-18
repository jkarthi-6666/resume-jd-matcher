# Known Failures and Limitations

This document is maintained honestly. "We built X and measured it barely helps" is a stronger portfolio claim than an unmeasured claim that everything works.

## Structural Limitations

### The overqualified problem
A staff engineer scores highly against a mid-level JD. The system says match; a recruiter says no. This is a product question the scoring model cannot answer — it is not solved here.

### Requirement extraction is the ceiling
If the planner misreads the JD, everything downstream is confidently wrong about the wrong thing. The system has no way to detect this.

### Evidence validation catches fabrication, not misinterpretation
A real quote can still be the wrong quote. "Familiar with Kubernetes" is a real substring, but it does not demonstrate hands-on production use. Reflection catches some of this; not all.

### A retrieval miss on a required item now rejects
The router rejects on a single missing required qualification. It cannot distinguish "the resume does not contain this" from "retrieval did not surface it and the scorer was confident about the wrong chunks." Low confidence and unverifiable evidence escalate instead of rejecting, which covers the honest-uncertainty case, but a confident zero from bad retrieval still rejects. This is unmeasured — the escalation/rejection split needs the evaluation set before the thresholds mean anything.

### The routing thresholds are unmeasured
`REQUIRED_ACCEPT_SCORE=0.75` and `PARTIAL_MATCH_SCORE=0.25` are inherited defaults, not calibrated values. They are configurable so they can be swept once labelled cases exist. Until then, treat the accept/review/reject boundaries as arbitrary.

### Single-resume, single-JD
No cross-candidate calibration. Scores are not comparable between reports.

### Confidence is self-reported
The model estimating its own confidence is not a calibrated probability. The escalation precision curve is the only thing keeping it honest.

## Retrieval Failures

### Skills-section bias
A dense skills list (`Python, Docker, Kubernetes, Terraform`) often out-ranks an experience bullet that actually demonstrates the skill — because it matches more terms. The reranker corrects this, but not always.

### Short resumes
Resumes under 3-4 distinct entries provide insufficient retrieval diversity. BM25 and FAISS both degrade when the corpus is tiny.

## Known Open Items

- [ ] Measure actual hallucinated quote rate across the eval set
- [ ] Measure whether reflection raises scores as often as it lowers them
- [ ] Measure how often reflection corrections are rejected by the invariant checks, and why
- [ ] Calibrate the 0.7 confidence floor with the escalation precision curve
- [ ] Calibrate `REQUIRED_ACCEPT_SCORE` and `PARTIAL_MATCH_SCORE` against labelled verdicts
- [ ] Measure how often a required-item rejection is caused by retrieval failure rather than a genuine gap
- [ ] Section detection fails on non-standard resume formats (two-column layouts, tables)
- [ ] No OCR support — image-only PDFs fail clearly but cannot be processed
