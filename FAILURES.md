# Known Failures and Limitations

This document separates observed failures from risks that have not yet been
measured. A measured limitation is more useful than an unsupported claim that
the system works reliably.

## Evaluation Scope

The current baseline was run on July 21, 2026 against all eight privacy-safe
synthetic resumes in `evaluation/labeled_set.jsonl`. This sample is deliberately
small and does not represent real recruiting traffic, unusual resume formats, or
demographic fairness. The results are diagnostic, not a production accuracy
claim.

**These full-pipeline numbers come from post-fix code.** They were produced
after the reflector guard described under *Measured Failures* was added, and are
not comparable to the pre-fix numbers recorded there. The naive column predates
that change and was deliberately not re-run: `run_naive()` is a single model call
that never invokes the reflector, so the change cannot affect it.

| Metric | Full pipeline (post-fix) | Naive baseline |
|---|---:|---:|
| Completed cases | 8/8 | 7/8 |
| Verdict accuracy | 100% (8/8) | n/a — emits no verdict |
| Score mean absolute error | 0.8 points | 10.39 points |
| Escalation rate | 50% (4/8) | n/a |
| False accepts | 0 | n/a |
| False rejects | 0 | n/a |
| Hallucinated quote rate | 0% | not checked |
| Unsupported positive-match rate | 0% | not checked |
| Reflection corrections | 0 raised, 0 lowered, 1 refused | n/a |
| Planner hallucination rate | 0% | n/a |

**100% on eight curated synthetic cases is not an accuracy claim.** Eight cases
built to exercise known routing paths, generated from one template by one
script, cannot establish a rate. One case moving swings this figure by 12.5
points. The number means the set no longer contains a known regression; it does
not mean the system is correct.

The raw reports and metric definitions are in `evaluation/results.json`,
`evaluation/results_naive.json`, and `evaluation/run_evaluation.py`.

### These numbers are not comparable to the previous baseline

The previously committed baseline reported five cases at 100% verdict accuracy
and 1.12 points of score error. That file is superseded for two independent
reasons, and the improvement or regression between them cannot be attributed:

1. **It covered five of eight cases.** Cases 006, 007 and 008 — the three that
   exercise routing precedence — had never had committed results, so the headline
   rate described a subset of the dataset sitting beside it.
2. **It was produced by different code.** `src/reflector.py`, `src/scorer.py`,
   `src/router.py`, `src/validate.py` and `prompts/reflection.txt` all changed
   between the commit that generated it and the commit measured here.

The earlier claim that aligning the scorer and reflector evidence scales moved
the set from 20% to 100% rests on single runs of a case that is now known to be
non-deterministic (see *Reflection intermittently collapses weak evidence*).

## Measured Failures

The first three entries below were measured pre-fix and are now guarded. They
are kept because the measurement, not the fix, is the evidence — and because the
underlying model behaviour still occurs and is only being caught.

### Reflection intermittently collapses weak evidence to absent (case_005) — guarded

`case_005` is labeled `needs_review`: the resume shows Python only in coursework
and a personal project against a JD requiring two years of professional
experience. Ground truth for the requirement is 0.25 — weak but real evidence.

In the committed run the case **failed, returning `reject`**. Component
attribution:

- **Scorer — correct.** Assigned exactly 0.25, reasoning that the evidence was
  academic.
- **Reflector — responsible.** Lowered 0.25 to 0.0. `prompts/reflection.txt`
  explicitly forbids this: *"Do not lower 0.25 to 0.0 when a real keyword
  listing, academic project, or other relevant adjacent evidence exists."* The
  critic violated an explicit instruction in its own prompt.
- **Router — correct.** 0.0 is below `PARTIAL_MATCH_SCORE`, so the documented
  precedence rejects. Given the score it received, the router had no other
  option.

This single case accounted for 25.0 of the 30.6 total score error in that run.
Pre-fix full-mode score MAE was 4.37 with it and 0.93 without it.

**Status: guarded, and the model behaviour is unchanged.** A lowering correction
that crosses `PARTIAL_MATCH_SCORE` downward now requires new evidence
(`_drops_below_missing_floor` in `src/reflector.py`). In the post-fix run the
reflector emitted the identical downgrade — its note still reads "Therefore the
score is lowered to 0.0" — and the guard refused it, recording
`unevidenced_missing_downgrade`. The score held at 0.25 and the case routed to
`needs_review`.

The critic still tries to do this. It is being caught, not corrected.

### The same case returns different verdicts across identical runs

case_005 was re-run three additional times at `temperature=0.0` with unchanged
code, PDF and job description. Raw outputs are in `evaluation/variance/`.

| Run | Verdict | Reflector action |
|---|---|---|
| `results.json` (committed run) | reject | lowered 0.25 → 0.0 |
| `variance/case_005_run1.json` | reject | lowered 0.25 → 0.0 |
| `variance/case_005_run2.json` | **needs_review** | no correction |
| `variance/case_005_run3.json` | reject | lowered 0.25 → 0.0 |

Three of four runs reject; one escalates. Pre-fix, **verdict accuracy on this
eight-case set was not a fixed number** — it oscillated between 7/8 and 8/8
depending on whether the reflector fired. This also means the older run's
passing result for case_005 was not evidence that any code change fixed it.

Post-fix, both branches of that coin now converge on `needs_review`: if the
critic emits the downgrade it is refused, and if it does not the score was
already 0.25. Each branch has been observed once — the refusal in the post-fix
run, the no-correction case in `variance/case_005_run2.json`. The convergence is
an argument from the code path, supported by two observations, not a measured
stability result. Repeat runs of the whole set would be needed for that.

### A lowering correction requires no evidence — fixed

A `raised` correction must quote new verbatim evidence or it is rejected
(`reflector.py`). A `lowered` correction that zeroed a scored requirement had its
evidence cleared before validation, so it passed the evidence check with no
evidence at all. The critic could zero any scored requirement unchallenged, and
zeroing is exactly what converts `needs_review` into `reject`. That asymmetry was
the mechanism behind the case_005 failure above.

Two changes close it. A lowering correction that crosses `PARTIAL_MATCH_SCORE`
downward now requires new evidence, and evidence validation now runs *before* a
zeroed scored requirement has its evidence discarded — clearing first would have
let an evidenced downgrade cite anything on the way to 0.0 and never be held to
it.

Only the boundary-crossing case is constrained. Lowering that stays above the
floor, or that starts below it, cannot change the verdict and is still left to
the critic's judgment. A legitimate "this evidence is irrelevant" correction
keeps a path: quote the text being disputed, and the quote is substring-checked.

### The threshold sweep reports an unreadable resume as an accept — fixed

`sweep_threshold.py` re-routed saved `all_requirements` at each confidence
floor. For case_007 that list is empty, because the unassessable-document guard
short-circuits before any requirement is scored, and `route([])` returns
`accept`. The sweep therefore reported the most dangerous possible outcome for a
document the pipeline actually escalated, and understated escalation — 25%
against that run's true 37.5% at floor 0.7.

This was a defect in the evaluation tool, not the pipeline. The pipeline routed
case_007 to `needs_review` correctly throughout.

A case that scored no requirements is now treated as threshold-independent: no
confidence floor can change a verdict decided before scoring, so its recorded
verdict is used at every floor and the report names which cases those are. The
sweep now reproduces the evaluator's independently computed metrics exactly at
the default floor, and diverges from the recorded verdict on no case.

### The naive baseline cannot handle an unassessable document

`run_naive()` does not catch `UnassessableDocumentError`, so case_007 aborts
with an unhandled error and the naive run completes 7/8. `run_full()` catches it
and returns a `needs_review` report. The full pipeline's document guard is
load-bearing rather than cosmetic.

The naive baseline is also fooled by the adversarial keyword case: on case_003 it
returned 60 against a ground truth of 25, while the full pipeline returned
exactly 25.0.

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

The threshold sweep now reports the accuracy cost of each floor, not just
escalation. Against the post-fix eight-case results:

| Confidence floor | Verdict accuracy | False accepts | False rejects | Escalation |
|---:|---:|---:|---:|---:|
| 0.5 | 100% (8/8) | 0 | 0 | 50.0% (4/8) |
| 0.6 | 100% (8/8) | 0 | 0 | 50.0% (4/8) |
| 0.7 | 100% (8/8) | 0 | 0 | 50.0% (4/8) |
| 0.8 | 75.0% (6/8) | 0 | 0 | 75.0% (6/8) |
| 0.9 | 75.0% (6/8) | 0 | 0 | 75.0% (6/8) |

Escalation rate alone could never justify a floor, because escalating everything
scores zero on both error types. With the error columns beside it the tradeoff is
at least legible: raising the floor above 0.7 on this set buys nothing and costs
two correct verdicts, converting them to review.

**This table is not a basis for tuning.** Eight cases cannot calibrate a
threshold, the rows are flat across 0.5–0.7 because no case sits near the
boundary, and every row is one sample from a provider that is known to vary. The
tool is now correct; the sample it is pointed at is still too small to read.

### Weak-versus-absent evidence remains a judgment boundary

The initial run rejected an education-only case that was labeled for review. The
scorer and the reflection prompt were both changed to treat relevant
keyword-only, academic, or adjacent evidence as weak (`0.25`) rather than absent
(`0.0`).

**This is now measured as only partly effective.** The scorer holds the line: it
assigned case_005 exactly 0.25 in every run observed. The reflector does not —
it lowered that 0.25 to 0.0 in three of four runs, against an explicit
prohibition in its own prompt. The earlier claim that this change "fixed the
synthetic regression" was based on a single run of a non-deterministic case and
does not hold.

### Reflection has never made a correction that improved a verdict

Across both eight-case runs, reflection attempted exactly one correction: the
case_005 downgrade. Pre-fix it was accepted and caused the run's only verdict
failure. Post-fix the critic attempted the same downgrade and it was refused.

Reflection has therefore raised no score, recovered no retrieval miss, and
corrected no inflation anywhere in this set. Its measured contribution to date is
one wrong correction, now blocked. That is not evidence the stage is useless —
this set contains no case built to need recovery — but nothing here supports the
claim that it recovers misses either.

The structured rejection-reason instrumentation is now exercised for the first
time, recording `unevidenced_missing_downgrade: 1`. The other reasons (evidence
substring misses, direction errors, stale scores, bounds errors) still record
nothing on this set and remain covered only by unit tests.

### Provider output is not deterministic — measured

Temperature zero does not guarantee identical provider responses. This is no
longer a hypothesis on this repository: case_005 was run four times at
`temperature=0.0` with unchanged code and inputs and returned `reject` three
times and `needs_review` once. Raw outputs are committed in
`evaluation/variance/`.

The practical consequence is that any single-run rate reported from this set
carries unquantified error bars. Only one case has been repeat-tested, so the
variance of the remaining seven is still unknown, as is whether the flip rate is
stable over more than four samples.

Cosine ranking now uses stable NumPy sorting, so equal similarity scores retain
chunk order deterministically; the former FAISS `IndexFlatIP` path did not make
that tie behavior explicit. This narrows one local source of run-to-run variance
but does not address provider variance.

Docling's layout and OCR models are another nondeterminism source. The Python
package is constrained to `>=2.113.0,<2.114.0`, but downloaded model weights are
not pinned by this repository, so a rebuilt model cache may change extraction,
reading order, or chunk boundaries.

### Token usage and baseline comparability

The first cross-chunk evidence fix passed the union of contiguous text and
contextualized chunks to both the reflector and naive baseline. On the seven
readable synthetic resumes this measured 1.996 times the characters and 2.0
times the whitespace-token proxy of contiguous text. It also made the
100-alphabetic-character quality floor represent roughly 50 unique characters.

The pipeline now keeps these concerns separate: contiguous text alone feeds the
naive baseline and reflector prompt and is used for document sufficiency; the
larger union is used only for evidence validation. Exact provider token usage,
latency, and cost still need measurement because tokenizer behavior and model
output lengths vary by provider.

### Evidence validation checks quotation, not meaning

The baseline produced no fabricated quotes and no unsupported positive matches
across 13 evidence quotes, which confirms the substring guard worked on these
eight cases. It does not prove that cited evidence supports the model's
interpretation. For example, `"Kubernetes"` can be a genuine resume substring
without demonstrating hands-on production experience — case_003 is exactly that
shape, and the full pipeline scored it 0.25 rather than treating the substring
as a match.

## Structural Limitations

### Requirement extraction is the ceiling

If the planner omits, duplicates, or misclassifies a job requirement, every
downstream stage evaluates the wrong plan. Planner requirements now require a
verbatim JD source span, invalid spans are dropped and measured, and paraphrased
duplicates above the semantic threshold are merged. These guards address
hallucinated and duplicated items, but they cannot detect an omitted requirement
or prove that a valid source span was classified correctly. Embedding-based
deduplication can also merge distinct but unusually similar requirements; the
0.9 threshold has not been calibrated on representative job descriptions.

### A retrieval miss can become a rejection

The router cannot distinguish "the resume does not contain this" from "retrieval
did not surface it." Low confidence and unverifiable evidence escalate, but a
confident zero caused by poor retrieval can still reject a candidate. The
current evaluation does not measure how often this happens.

The `LOW_RETRIEVAL_FLOOR` is based on reciprocal-rank-fusion agreement. It does
not directly measure whether the retrieved chunks are semantically relevant.
Planner-provided lexical aliases now reduce abbreviation misses in BM25, but
their correctness still depends on planner output. Terms are capped at five and
restricted by prompt to alternate names; an overly broad alias can still create
a misleading lexical hit.

### Routing thresholds are not calibrated

`REQUIRED_ACCEPT_SCORE=0.75`, `PARTIAL_MATCH_SCORE=0.25`, and
`CONFIDENCE_FLOOR=0.7` remain defaults rather than thresholds selected against a
representative labeled dataset. The eight synthetic cases are insufficient for
calibration: one case moving swings verdict accuracy by 12.5 points, and the
sweep that would inform a choice of floor is itself defective (see measured
failures).

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

Docling contextualization also prepends headings to every chunk. Under a common
heading such as "Experience", generic queries like "years of experience" give
many chunks the same BM25 term contribution. This is material for requirements
whose lexical query is mostly generic wording; requirement-specific terms and
the independent vector arm usually provide differentiation, but the effect has
not been quantified and is intentionally not changed with query expansion.

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

Negligible and symbol-heavy OCR output is detected before retrieval and scoring.
This is now backed by a committed result: case_007 routes to `needs_review`
rather than a qualification-based rejection, with score 0.0 and no requirements
scored. The naive baseline has no equivalent guard and aborts on the same
document.

The content guard does not measure linguistic coherence: sufficiently long
alphabetic OCR gibberish can still pass. Its 100-character and 0.5
alphabetic-ratio defaults are deliberate heuristics, not thresholds calibrated
on representative scans. case_007 is a deterministic synthetic fixture, not a
real scan, so end-to-end OCR quality is still unmeasured.

### Eligibility constraints require human confirmation

Location, schedule, travel, relocation, and work-authorization constraints are
separated from scored qualifications. Two committed results now support this,
where previously the claim rested on cases that had never been run:

- **case_006** — resume is silent on work authorization. Routed `needs_review`
  with match score 100.0, confirming an unresolved gate escalates without
  lowering the qualification score.
- **case_008** — unqualified candidate plus an unresolved gate. Routed `reject`,
  confirming the gate does not suppress a scored rejection.

An explicit violation also escalates by default; deployments can opt into
rejection as a policy choice. That `REJECT_VIOLATED_GATES=true` path has no
labeled case and is untested by this set. The system still relies on the planner
to classify these constraints as gates, and it cannot verify a candidate's
statement independently.

## Open Work

- [x] Run and commit results for the full labeled set, including the routing
      precedence cases 006, 007 and 008.
- [x] Establish that provider output varies run to run at temperature zero
      (case_005 only; see `evaluation/variance/`).
- [x] Fix `sweep_threshold.py` so it does not re-route an unassessable document
      to `accept`, and report verdict accuracy, false accepts, and false rejects
      per threshold.
- [x] Require evidence for a lowering correction that crosses the missing-score
      floor, as raises already require it.
- [ ] Expand evaluation to representative anonymized or consented resumes.
- [ ] Repeat-run the whole set to put error bars on every reported rate; only
      one of eight cases has been repeat-tested.
- [ ] Calibrate confidence against representative human-reviewed outcomes.
- [ ] Calibrate required, partial-match, and confidence thresholds jointly.
- [ ] Exercise the rejected-correction reason instrumentation, which this set
      leaves at zero.
- [ ] Cover the `REJECT_VIOLATED_GATES=true` policy path, which has no case.
- [ ] Measure retrieval-caused rejection separately from genuine qualification gaps.
- [ ] Add two-column, table-based, short, malformed, and image-only PDF cases
      beyond the current deterministic poor-extraction fixture.
- [ ] Measure provider variance, latency, token usage, and cost per report.
- [ ] Evaluate fairness and disparate error rates before any hiring use.
