"""Full agentic pipeline and naive baseline mode."""
import json
from dataclasses import dataclass, field
from src.schemas import (
    Chunk, Requirement, RequirementAnalysis,
    FinalReport, NaiveResult, ReflectionResult, RequirementPlan,
)
from src import docling_processor, planner, retriever as ret_mod, reranker, scorer
from src import validate as val, calculator, reflector, router, config, llm


@dataclass
class DebugInfo:
    resume_text: str = ""
    validation_corpus: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    requirements: list[Requirement] = field(default_factory=list)
    retrieval_debug: list[dict] = field(default_factory=list)  # per-requirement
    evidence_validation: list[dict] = field(default_factory=list)
    raw_model_responses: list[str] = field(default_factory=list)
    token_notes: list[str] = field(default_factory=list)
    reflection_result: ReflectionResult | None = None
    requirement_plan: RequirementPlan | None = None


def run_naive(pdf_bytes: bytes, job_description: str) -> NaiveResult:
    """Phase 1 baseline — single LLM call, no RAG."""
    if not job_description.strip():
        raise ValueError("Job description is empty.")
    resume_text, _ = docling_processor.extract_and_chunk_resume(pdf_bytes)
    prompt = (
        f"You are a recruiter. Compare the resume and job description below.\n\n"
        f"Resume:\n{resume_text}\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Return JSON with: match_score (0-100 int), matched_skills (list[str]), "
        f"missing_skills (list[str]), recommendation (str)."
    )
    return llm.call(prompt, config.NAIVE_MODEL, NaiveResult, temperature=0.0)


def run_full(
    pdf_bytes: bytes,
    job_description: str,
    generate_recommendation: bool = True,
    progress_cb=None,
) -> tuple[FinalReport, DebugInfo]:
    """Full agentic pipeline.

    progress_cb, if given, is called as progress_cb(stage: str, done: int, total: int)
    at each stage boundary so callers (e.g. the Streamlit UI) can render progress.
    """
    debug = DebugInfo()

    def _tick(stage: str, done: int, total: int):
        if progress_cb is not None:
            progress_cb(stage, done, total)

    if not job_description.strip():
        raise ValueError("Job description is empty.")

    n_stages = 6  # process PDF, index, plan, score requirements, reflect, route

    # --- 1. Docling conversion and structure-aware chunking ---
    _tick("Extracting and chunking resume", 0, n_stages)
    try:
        resume_text, chunks = docling_processor.extract_and_chunk_resume(pdf_bytes)
    except docling_processor.UnassessableDocumentError as exc:
        reason = str(exc)
        _tick("Document requires human review", 1, n_stages)
        report = FinalReport(
            verdict="needs_review",
            review_reason=reason,
            match_score=0.0,
            score_pre_reflection=0.0,
            score_post_reflection=0.0,
            reflection_delta=0.0,
            all_requirements=[],
            corrections=[],
            recommendation=(
                "The resume document could not be assessed reliably. "
                "Review the original document or request a clearer, text-based PDF."
            ),
            reflection_notes=["Scoring was skipped because resume text was unassessable."],
        )
        return report, debug
    debug.resume_text = resume_text
    validation_corpus = docling_processor.build_validation_corpus(
        resume_text, chunks
    )
    debug.validation_corpus = validation_corpus
    debug.chunks = chunks

    # --- 2. Build retriever ---
    _tick("Building hybrid retriever index", 1, n_stages)
    hybrid = ret_mod.HybridRetriever()
    hybrid.build(chunks)

    # --- 3. Requirement planning ---
    _tick("Planning requirements from job description", 2, n_stages)
    requirement_plan = planner.plan(job_description)
    debug.requirement_plan = requirement_plan
    requirements = requirement_plan.requirements
    debug.requirements = requirements

    # --- 4. Per-requirement: retrieve → rerank → score → validate (parallel) ---
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process_requirement(req: Requirement):
        bm25_ranked, vec_ranked, merged, top_rrf = hybrid.retrieve(
            req.requirement,
            query_terms=req.query_terms,
        )
        low_conf = top_rrf < config.LOW_RETRIEVAL_FLOOR
        candidate_chunks = hybrid.get_chunks(merged)
        top_chunks = reranker.rerank(req.requirement, candidate_chunks)
        analysis = scorer.score(req, top_chunks, low_retrieval_confidence=low_conf)
        analysis = val.validate_evidence(analysis, top_chunks)
        analysis = val.derive_status(analysis)
        ret_dbg = {
            "requirement_id": req.id,
            "requirement": req.requirement,
            "query_terms": req.query_terms,
            "bm25_ranked": bm25_ranked,
            "vec_ranked": vec_ranked,
            "rrf_merged": merged,
            "post_rerank": [c.chunk_id for c in top_chunks],
            "top_rrf_score": top_rrf,
            "low_retrieval_confidence": low_conf,
        }
        ev_dbg = {
            "requirement_id": req.id,
            "evidence": analysis.evidence,
            "evidence_valid": analysis.evidence_valid,
        }
        return req.id, analysis, ret_dbg, ev_dbg

    # Run all requirements concurrently; collect in original order
    total_reqs = len(requirements)
    _tick(f"Scoring requirements (0/{total_reqs})", 3, n_stages)
    result_map: dict[str, tuple] = {}
    n_done = 0
    with ThreadPoolExecutor(max_workers=min(len(requirements), 8)) as pool:
        futures = {pool.submit(_process_requirement, req): req for req in requirements}
        for future in as_completed(futures):
            req_id, analysis, ret_dbg, ev_dbg = future.result()
            result_map[req_id] = (analysis, ret_dbg, ev_dbg)
            n_done += 1
            _tick(f"Scoring requirements ({n_done}/{total_reqs})", 3, n_stages)

    analyses: list[RequirementAnalysis] = []
    for req in requirements:
        analysis, ret_dbg, ev_dbg = result_map[req.id]
        analyses.append(analysis)
        debug.retrieval_debug.append(ret_dbg)
        debug.evidence_validation.append(ev_dbg)

    # --- 6. Pre-reflection score ---
    score_pre = calculator.weighted_score(analyses)

    # --- 7. Adversarial reflection ---
    _tick("Running adversarial reflection", 4, n_stages)
    corrected_analyses, reflection_result = reflector.reflect(
        analyses,
        resume_text,
        validation_corpus=validation_corpus,
    )
    debug.reflection_result = reflection_result

    # Self-reported model confidence is not calibrated. Convert it into a
    # routing signal only after evidence validation, retrieval checks, and a
    # completed full-resume audit have all succeeded.
    corrected_analyses = [
        val.calibrate_confidence(a, reflection_result.completed)
        for a in corrected_analyses
    ]

    # --- 8. Post-reflection score ---
    score_post = calculator.weighted_score(corrected_analyses)
    delta = round(score_post - score_pre, 1)

    # --- 9. Confidence routing ---
    _tick("Routing verdict and generating recommendation", 5, n_stages)
    verdict, review_reason = router.route(corrected_analyses)

    # --- 10. Recommendation ---
    recommendation = _make_recommendation(corrected_analyses, verdict, score_post)

    _tick("Done", n_stages, n_stages)

    report = FinalReport(
        verdict=verdict,
        review_reason=review_reason,
        match_score=score_post,
        score_pre_reflection=score_pre,
        score_post_reflection=score_post,
        reflection_delta=delta,
        all_requirements=corrected_analyses,
        corrections=reflection_result.changed_requirements,
        recommendation=recommendation,
        reflection_notes=reflection_result.review_notes,
    )
    return report, debug


def _make_recommendation(
    analyses: list[RequirementAnalysis],
    verdict: str,
    score: float,
) -> str:
    matched   = [a for a in analyses if a.kind == "scored" and a.status == "matched"]
    missing   = [
        a for a in analyses
        if a.kind == "scored" and a.status == "missing" and a.importance == "required"
    ]
    uncertain = [a for a in analyses if a.status == "uncertain"]

    parts = [f"Overall match: {score:.0f}/100."]
    if matched:
        skills = ", ".join(a.requirement for a in matched[:3])
        parts.append(f"Strong evidence for: {skills}.")
    if missing:
        skills = ", ".join(a.requirement for a in missing[:3])
        parts.append(f"Missing required: {skills}.")
    if uncertain:
        parts.append(f"{len(uncertain)} requirement(s) could not be confidently assessed.")

    if verdict == "accept":
        parts.append("Recommended for next steps.")
    elif verdict == "reject":
        parts.append("Does not meet required qualifications.")
    else:
        parts.append("Recommend human review before decision.")

    return " ".join(parts)
