"""Full agentic pipeline and naive baseline mode."""
import json
from dataclasses import dataclass, field
from src.schemas import (
    Chunk, Requirement, RequirementAnalysis,
    FinalReport, NaiveResult, ReflectionResult,
)
from src import parser, chunker, planner, retriever as ret_mod, reranker, scorer
from src import validate as val, calculator, reflector, router, config, llm


@dataclass
class DebugInfo:
    resume_text: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    requirements: list[Requirement] = field(default_factory=list)
    retrieval_debug: list[dict] = field(default_factory=list)  # per-requirement
    evidence_validation: list[dict] = field(default_factory=list)
    raw_model_responses: list[str] = field(default_factory=list)
    token_notes: list[str] = field(default_factory=list)
    reflection_result: ReflectionResult | None = None


def run_naive(pdf_bytes: bytes, job_description: str) -> NaiveResult:
    """Phase 1 baseline — single LLM call, no RAG."""
    if not job_description.strip():
        raise ValueError("Job description is empty.")
    resume_text = parser.extract_text(pdf_bytes)
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
) -> tuple[FinalReport, DebugInfo]:
    """Full agentic pipeline."""
    debug = DebugInfo()

    if not job_description.strip():
        raise ValueError("Job description is empty.")

    # --- 1. PDF extraction ---
    resume_text = parser.extract_text(pdf_bytes)
    debug.resume_text = resume_text

    # --- 2. Chunking ---
    chunks = chunker.chunk_resume(resume_text)
    debug.chunks = chunks

    # --- 3. Build retriever ---
    hybrid = ret_mod.HybridRetriever()
    hybrid.build(chunks)

    # --- 4. Requirement planning ---
    requirements = planner.plan(job_description)
    debug.requirements = requirements

    # --- 5. Per-requirement: retrieve → rerank → score → validate (parallel) ---
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process_requirement(req: Requirement):
        bm25_ranked, vec_ranked, merged, top_rrf = hybrid.retrieve(req.requirement)
        low_conf = top_rrf < config.LOW_RETRIEVAL_FLOOR
        candidate_chunks = hybrid.get_chunks(merged)
        top_chunks = reranker.rerank(req.requirement, candidate_chunks)
        analysis = scorer.score(req, top_chunks, low_retrieval_confidence=low_conf)
        analysis = val.validate_evidence(analysis, top_chunks)
        analysis = val.derive_status(analysis)
        ret_dbg = {
            "requirement_id": req.id,
            "requirement": req.requirement,
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
    result_map: dict[str, tuple] = {}
    with ThreadPoolExecutor(max_workers=min(len(requirements), 8)) as pool:
        futures = {pool.submit(_process_requirement, req): req for req in requirements}
        for future in as_completed(futures):
            req_id, analysis, ret_dbg, ev_dbg = future.result()
            result_map[req_id] = (analysis, ret_dbg, ev_dbg)

    analyses: list[RequirementAnalysis] = []
    for req in requirements:
        analysis, ret_dbg, ev_dbg = result_map[req.id]
        analyses.append(analysis)
        debug.retrieval_debug.append(ret_dbg)
        debug.evidence_validation.append(ev_dbg)

    # --- 6. Pre-reflection score ---
    score_pre = calculator.weighted_score(analyses)

    # --- 7. Adversarial reflection ---
    corrected_analyses, reflection_result = reflector.reflect(
        analyses, resume_text
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
    verdict, review_reason = router.route(corrected_analyses)

    # --- 10. Recommendation ---
    recommendation = _make_recommendation(corrected_analyses, verdict, score_post)

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
    matched   = [a for a in analyses if a.status == "matched"]
    missing   = [a for a in analyses if a.status == "missing" and a.importance == "required"]
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
