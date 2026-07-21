# Known Failures and Limitations

This document separates observed failures from risks that have not yet been
measured. A measured limitation is more useful than an unsupported claim that
the system works reliably.

## Evaluation Scope

Run against all eight privacy-safe synthetic resumes in
`evaluation/labeled_set.jsonl`. Each run's timestamp is recorded in the
`generated_at` field of the results files.

**The full-pipeline column is post-fix**, produced after the reflector guard
described below. The naive column predates it and was not re-run: `run_naive()`
is a single model call that never invokes the reflector, so the change cannot
affect it.

| Metric | Full pipeline (post-fix) | Naive baseline |
|---|---:|---:|
| Completed cases | 8/8 | 7/8 |
| Verdict accuracy | 100% (8/8) | n/a — emits no verdict |
| Score mean absolute error | 0.8 points | 10.39 points |
| Escalation rate | 50% (4/8) | n/a |
| False accepts / false rejects | 0 / 0 | n/a |
| Hallucinated quote rate | 0% | not checked |
| Unsupported positive-match rate | 0% | not checked |
| Reflection corrections | 0 raised, 0 lowered, 1 refused | n/a |
| Planner hallucination rate | 0% | n/a |

**100% on eight curated synthetic cases is not an accuracy claim.** Eight cases
built to exercise known routing paths, generated from one template by one
script, cannot establish a rate — one case moving swings this figure by 12.5
points. It means the set contains no known regression, nothing more.

Raw reports and metric definitions are in `evaluation/results.json`,
`evaluation/results_naive.json`, and `evaluation/run_evaluation.py`.

An earlier baseline reported five cases at 100% accuracy and 1.12 points of
error. It is superseded and not comparable: it covered five of eight cases,
omitting the three that exercise routing precedence, and it was produced before
`reflector.py`, `scorer.py`, `router.py`, `validate.py` and
`prompts/reflection.txt` changed. Its claim that aligning the scorer and
reflector scales moved the set from 20% to 100% rested on a single run of a case
now known to be non-deterministic.

## Measured Failures

### Reflection collapses weak evidence to absent — guarded, not corrected

`case_005` is labeled `needs_review`: Python appears only in coursework and a
personal project against a JD requiring two years of professional experience.
Ground truth is 0.25 — weak but real evidence.

Pre-fix the case returned `reject`. Attribution:

- **Scorer — correct.** Assigned exactly 0.25, and did so in every run observed.
- **Reflector — responsible.** Lowered 0.25 to 0.0, violating an explicit
  instruction in its own prompt: *"Do not lower 0.25 to 0.0 when a real keyword
  listing, academic project, or other relevant adjacent evidence exists."*
- **Router — correct.** 0.0 is below `PARTIAL_MATCH_SCORE`, so documented
  precedence rejects. Given that score it had no other option.

That one case produced 25.0 of the run's 30.6 total score error (MAE 4.37 with
it, 0.93 without).

**It was also intermittent.** Four runs at `temperature=0.0` with unchanged code
and inputs returned `reject` three times and `needs_review` once (raw outputs in
`evaluation/variance/`). Pre-fix, verdict accuracy on this set was therefore not
a fixed number; it oscillated between 7/8 and 8/8.

**Mechanism.** A `raised` correction must quote new verbatim evidence. A
`lowered` correction that zeroed a scored requirement had its evidence cleared
*before* validation, so it passed the evidence check with no evidence at all. The
critic could zero any scored requirement unchallenged, and zeroing is exactly
what converts `needs_review` into `reject`.

**Guard.** A lowering correction that crosses `PARTIAL_MATCH_SCORE` downward now
requires new evidence (`_drops_below_missing_floor`), and validation runs before
a zeroed requirement's evidence is discarded. Only the boundary crossing is
constrained — lowering that stays above the floor, or starts below it, cannot
change the verdict and is left to the critic. A legitimate "this evidence is
irrelevant" correction keeps a path: quote the text being disputed.

**The model behaviour is unchanged.** In the post-fix run the reflector emitted
the identical downgrade — its note still reads "Therefore the score is lowered to
0.0" — and the guard refused it, recording `unevidenced_missing_downgrade`. The
critic still attempts this. It is being caught, not corrected.

Post-fix both branches converge on `needs_review`: refused if emitted, already
0.25 if not. Each branch has been observed once. That is an argument from the
code path with two observations, not a measured stability result.

### The threshold sweep reported an unreadable resume as an accept — fixed

`sweep_threshold.py` re-routed saved `all_requirements` at each floor. For
case_007 that list is empty, because the unassessable-document guard
short-circuits before scoring, and `route([])` returns `accept`. The tool
reported the most dangerous possible outcome for a document the pipeline
actually escalated, and understated escalation — 25% against that run's true
37.5% at floor 0.7.

This was a defect in the evaluation tool. The pipeline routed case_007 correctly
throughout. A case that scored no requirements is now treated as
threshold-independent, and the sweep reproduces the evaluator's independently
computed metrics exactly at the default floor.

### The naive baseline cannot handle an unassessable document

`run_naive()` does not catch `UnassessableDocumentError`, so case_007 aborts and
the naive run completes 7/8. `run_full()` catches it and returns `needs_review`.
The full pipeline's document guard is load-bearing rather than cosmetic.

Naive is also fooled by the adversarial keyword case: on case_003 it returned 60
against a ground truth of 25, where the full pipeline returned exactly 25.0.

## Observed and Residual Risks

### Raw model confidence is not calibrated

An early baseline's largest failure was confidence routing: the scorer assigned
`1.0`, described the evidence as explicit, then reported confidence between `0.0`
and `0.1`, so `CONFIDENCE_FLOOR=0.7` escalated well-supported matches.

Raw confidence is now only one input. After reflection, `calibrate_confidence()`
raises confidence to the routing floor only when evidence is valid, retrieval was
not flagged weak, and the full-resume audit completed. This is a guarded decision
rule, not probability calibration, and its safety depends on retrieval-confidence
detection and the critic noticing full-resume evidence — both imperfect.

The sweep now reports the accuracy cost of each floor:

| Confidence floor | Verdict accuracy | False accepts | False rejects | Escalation |
|---:|---:|---:|---:|---:|
| 0.5 | 100% (8/8) | 0 | 0 | 50.0% (4/8) |
| 0.6 | 100% (8/8) | 0 | 0 | 50.0% (4/8) |
| 0.7 | 100% (8/8) | 0 | 0 | 50.0% (4/8) |
| 0.8 | 75.0% (6/8) | 0 | 0 | 75.0% (6/8) |
| 0.9 | 75.0% (6/8) | 0 | 0 | 75.0% (6/8) |

Escalation rate alone could never justify a floor, since escalating everything
scores zero on both error types. Raising the floor above 0.7 here buys nothing
and converts two correct verdicts to review.

**This table is not a basis for tuning.** Eight cases cannot calibrate a
threshold, the rows are flat across 0.5–0.7 because no case sits near the
boundary, and every row is one sample from a provider known to vary.

### Reflection has never made a correction that improved a verdict

Across both runs, reflection attempted exactly one correction: the case_005
downgrade — accepted pre-fix (causing the only verdict failure), refused
post-fix. It has raised no score, recovered no retrieval miss, and corrected no
inflation anywhere in this set.

That is not evidence the stage is useless — no case here is built to need
recovery — but nothing supports the claim that it recovers misses either.

The rejection-reason instrumentation is now exercised for the first time,
recording `unevidenced_missing_downgrade: 1`. The other reasons (substring
misses, direction errors, stale scores, bounds errors) record nothing on this set
and remain covered only by unit tests.

### Provider output is not deterministic — measured

Temperature zero does not guarantee identical responses. case_005 was run four
times at `temperature=0.0` with unchanged code and inputs and returned different
verdicts (`evaluation/variance/`). Any single-run rate from this set therefore
carries unquantified error bars. Only one case has been repeat-tested, so the
variance of the other seven is unknown.

Cosine ranking uses stable NumPy sorting, so equal similarity scores retain chunk
order deterministically. That narrows one local source of variance but does not
address provider variance. Docling's layout and OCR models are another source:
the package is pinned to `>=2.113.0,<2.114.0`, but downloaded model weights are
not, so a rebuilt cache may change extraction, reading order, or chunk
boundaries.

### Evidence validation checks quotation, not meaning

The run produced no fabricated quotes and no unsupported positive matches, which
confirms the substring guard worked on these eight cases. It does not prove cited
evidence supports the model's interpretation: `"Kubernetes"` can be a genuine
substring without demonstrating production experience. case_003 is exactly that
shape, and the pipeline scored it 0.25 rather than treating the substring as a
match.

### Token usage and cost are unmeasured

Contiguous text alone feeds the naive baseline, the reflector prompt, and
document-sufficiency checks; the larger union of contiguous text and
contextualized chunks is used only for evidence validation. Exact provider token
usage, latency, and cost still need measurement, because tokenizer behaviour and
output lengths vary by provider.

## Structural Limitations

### Requirement extraction is the ceiling

If the planner omits, duplicates, or misclassifies a requirement, every
downstream stage evaluates the wrong plan. Requirements now need a verbatim JD
source span, invalid spans are dropped and measured, and paraphrased duplicates
above the semantic threshold are merged. These guards cannot detect an omitted
requirement or prove a valid span was classified correctly. Embedding-based
deduplication can also merge distinct but unusually similar requirements; the 0.9
threshold has not been calibrated on representative job descriptions.

### A retrieval miss can become a rejection

The router cannot distinguish "the resume does not contain this" from "retrieval
did not surface it." Low confidence and unverifiable evidence escalate, but a
confident zero caused by poor retrieval can still reject a candidate. The current
evaluation does not measure how often this happens.

`LOW_RETRIEVAL_FLOOR` is based on reciprocal-rank-fusion agreement and does not
measure whether retrieved chunks are semantically relevant. Planner-provided
lexical aliases reduce abbreviation misses in BM25 but depend on planner output;
terms are capped at five and restricted by prompt to alternate names, and an
overly broad alias can still create a misleading lexical hit.

### Routing thresholds are not calibrated

`REQUIRED_ACCEPT_SCORE=0.75`, `PARTIAL_MATCH_SCORE=0.25`, and
`CONFIDENCE_FLOOR=0.7` are defaults, not thresholds selected against a
representative labeled dataset. Eight synthetic cases are insufficient: one case
moving swings verdict accuracy by 12.5 points, and no case sits near the
confidence boundary, so the sweep cannot discriminate between floors.

### The overqualified problem

A staff engineer can score highly against a mid-level job description even when a
recruiter would consider the candidate unsuitable. This is a product-policy
question that evidence matching alone cannot answer.

### Scores are not comparable across candidates

The system evaluates one resume against one job description, with no
cross-candidate calibration. A score of 80 in one report is not equivalent to 80
in another.

## Retrieval and Input Limitations

### Skills-section bias

A dense skills list can outrank experience evidence because it contains more
matching terms. The reranker is prompted to prefer demonstrated capability, but
the set is too small to quantify how often it succeeds.

Docling contextualization also prepends headings to every chunk. Under a common
heading such as "Experience", generic queries like "years of experience" give
many chunks the same BM25 contribution. Requirement-specific terms and the
independent vector arm usually differentiate, but the effect is unquantified and
deliberately not addressed with query expansion.

### Short resumes

Resumes with fewer than three or four distinct entries provide little retrieval
diversity. BM25 and cosine similarity both return rankings even when none of the
chunks are useful.

### Non-standard PDF layouts

Docling provides layout-aware reading order, but complex two-column resumes,
decorative templates, and dense tables can still produce incorrect element
ordering or heading associations. **These formats are not represented in the
evaluation set at all**, so parsing robustness is entirely unmeasured.

### OCR quality and cold-start cost

OCR accuracy depends on scan quality, language, fonts, and runtime. The first
conversion in a fresh environment is slow while Docling initializes its models.

Negligible and symbol-heavy output is detected before retrieval and scoring,
backed by a committed result: case_007 routes to `needs_review` with score 0.0
and no requirements scored. The guard does not measure linguistic coherence —
sufficiently long alphabetic gibberish can still pass — and its 100-character and
0.5 alphabetic-ratio defaults are heuristics, not calibrated thresholds.
case_007 is a deterministic synthetic fixture, not a real scan, so end-to-end OCR
quality remains unmeasured.

### Eligibility constraints require human confirmation

Location, schedule, travel, relocation, and work-authorization constraints are
separated from scored qualifications. Two committed results support this:

- **case_006** — resume silent on work authorization. Routed `needs_review` with
  match score 100.0: an unresolved gate escalates without lowering the score.
- **case_008** — unqualified candidate plus an unresolved gate. Routed `reject`:
  the gate does not suppress a scored rejection.

An explicit violation escalates by default, and deployments can opt into
rejection. That `REJECT_VIOLATED_GATES=true` path has no labeled case. The system
still relies on the planner to classify constraints as gates and cannot verify a
candidate's statement independently.

## Open Work

- [ ] Expand evaluation to representative anonymized or consented resumes.
- [ ] Add two-column, table-based, short, malformed, and image-only PDF cases.
- [ ] Repeat-run the whole set to put error bars on every reported rate; only one
      of eight cases has been repeat-tested.
- [ ] Add cases near the confidence boundary so the sweep can discriminate
      between floors.
- [ ] Calibrate confidence against representative human-reviewed outcomes.
- [ ] Calibrate required, partial-match, and confidence thresholds jointly.
- [ ] Exercise the remaining rejected-correction reasons outside unit tests.
- [ ] Cover the `REJECT_VIOLATED_GATES=true` policy path, which has no case.
- [ ] Measure retrieval-caused rejection separately from genuine qualification
      gaps.
- [ ] Measure provider variance, latency, token usage, and cost per report.
- [ ] Evaluate fairness and disparate error rates before any hiring use.
