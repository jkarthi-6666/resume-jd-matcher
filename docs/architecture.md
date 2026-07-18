# Architecture

## Pipeline Flow

```
Resume PDF                                   Job Description
    ↓                                              ↓
Extract text (PyMuPDF)                    Planner extracts requirements
    ↓                                              ↓
Chunk by entry (role/project/degree)      Pydantic validation (Literal types)
    ↓                                              ↓
Prepend header to each chunk                       │
    ↓                                              │
BM25 index + FAISS index                           │
    └────────────────────┬─────────────────────────┘
                         ↓
              For each requirement:
                         ↓
        BM25 top 8  +  Vector top 8  →  RRF merge
                         ↓
              Rerank → top 3 chunks
                         ↓
              Score (score + confidence + verbatim evidence)
                         ↓
              ▸ EVIDENCE VALIDATION (substring check, Python)
                         ↓
              Weighted score, computed in Python
                         ↓
              score_pre_reflection
                         ↓
        Adversarial reflection (full resume, no scorer reasoning)
                         ↓
        Corrections — raised AND lowered — re-validated
                         ↓
              score_post_reflection
                         ↓
              Confidence router
                         ↓
        accept  |  needs_review  |  reject
```

## Design Decisions

### Why entry-based chunking, not token windows?
A token window orphans bullets from their role headers. The scorer then sees context-free text and defaults to a middling score. One chunk per job/project/degree ensures the header (role, company, dates) is always present in what the model sees.

### Why hybrid retrieval?
- BM25 guarantees exact keyword hits that embeddings can miss.
- Embeddings handle paraphrase that BM25 can miss.
- RRF merges both in 10 lines with nothing to tune.

### Why evidence validation?
The model can hallucinate plausible-sounding quotes. A substring check in Python is the only way to prove evidence is real. Without this, every "evidence" claim is just the model describing itself.

A positive score with *no* evidence is treated the same as a fabricated quote: both are unsupported claims, and both are zeroed. Only a zero score is entitled to cite nothing.

### Why adversarial reflection?
- Sees full resume (not just retrieved chunks) to catch retrieval misses.
- Does NOT see scorer's reasoning (to avoid ratification bias).
- Checks both inflation AND deflation.
- One pass only — convergence loops drift.

Every correction must survive `_is_valid_correction()` before it is applied: the score must be in range, `old_score` must match the score we actually hold (a stale value means the critic judged a version that no longer exists), the direction must agree with the arithmetic, and a raised score must bring new evidence that validates against the full resume. Invariants are enforced per-correction rather than in the Pydantic schema, so one malformed correction cannot fail validation for the whole batch and discard the valid corrections alongside it.

### Why confidence routing?
An uncertain score that routes nowhere is decoration. The router explicitly decides when a human should review, and names the specific reason.

### Routing policy
Evaluated in order; the first rule that matches wins:

```
unverifiable evidence on a required item        -> needs_review
low confidence on a required item               -> needs_review
any required item below PARTIAL_MATCH_SCORE     -> reject
any required item below REQUIRED_ACCEPT_SCORE   -> needs_review
every required item at REQUIRED_ACCEPT_SCORE+   -> accept
```

The two review rules are checked before the reject rule on purpose. A requirement the system could not assess is not evidence that the candidate lacks it, so uncertainty must escalate rather than reject.

One missing required qualification is sufficient to reject: scoring 1.0 on Python does not excuse scoring 0.0 on a required AWS qualification.

`derive_status()` and the router read the same thresholds from `config`, so a requirement badged `matched` in the UI is by construction one the router will accept. Hardcoding either independently lets the badge and the verdict disagree.

## File Map

| File | Responsibility |
|---|---|
| `src/parser.py` | PDF text extraction, ligature fixes |
| `src/chunker.py` | Entry-based resume chunking |
| `src/embeddings.py` | FAISS vector index |
| `src/retriever.py` | BM25 + FAISS + RRF |
| `src/reranker.py` | LLM rerank (cheap model) |
| `src/planner.py` | JD → requirements (cached, temp=0) |
| `src/scorer.py` | Per-requirement scoring |
| `src/validate.py` | Evidence substring check + status derivation |
| `src/calculator.py` | Weighted score (Python) |
| `src/reflector.py` | Adversarial critic |
| `src/router.py` | Verdict routing |
| `src/pipeline.py` | Orchestrates all stages |
| `src/llm.py` | Provider wrapper |
| `src/schemas.py` | Pydantic models with Literal types |
