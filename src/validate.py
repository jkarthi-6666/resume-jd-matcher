"""Phase 5.5 – Evidence validation. Every evidence quote must be a real substring."""
import re
from src.schemas import RequirementAnalysis, Chunk
from src import config


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def clean_evidence_quote(quote: str) -> str:
    """Remove presentation-only quote delimiters before exact validation."""
    cleaned = quote.strip()
    pairs = (("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’"))
    for opening, closing in pairs:
        if len(cleaned) >= 2 and cleaned.startswith(opening) and cleaned.endswith(closing):
            return cleaned[len(opening):-len(closing)].strip()
    return cleaned


def invalid_evidence_quotes(evidence: list[str], chunks: list[Chunk]) -> list[str]:
    """Quotes that are not substrings of the given chunks."""
    corpus = normalize(" ".join(c.embed_text for c in chunks))
    return [q for q in evidence if normalize(q) not in corpus]


def validate_evidence(
    analysis: RequirementAnalysis,
    retrieved_chunks: list[Chunk],
) -> RequirementAnalysis:
    """Every positive claim must be backed by a quote found in the resume text."""
    cleaned = [clean_evidence_quote(q) for q in analysis.evidence if q.strip()]
    analysis.evidence = cleaned

    # A positive score with nothing to back it is unsupported by definition.
    if analysis.score > 0.0 and not cleaned:
        analysis.evidence_valid = False
        analysis.score = 0.0
        analysis.confidence = 0.0
        analysis.reason = "Positive score was returned without supporting evidence."
        return analysis

    # A missing requirement is entitled to have no evidence.
    if not cleaned:
        analysis.evidence_valid = True
        return analysis

    if not retrieved_chunks:
        analysis.evidence_valid = False
        analysis.score = 0.0
        analysis.confidence = 0.0
        analysis.reason = "Evidence was returned, but no resume chunks were retrieved."
        return analysis

    bad = invalid_evidence_quotes(cleaned, retrieved_chunks)
    if bad:
        analysis.evidence_valid = False
        analysis.score = 0.0
        analysis.confidence = 0.0
        analysis.reason = (
            f"Evidence not found in retrieved resume text: {bad[0][:80]!r}"
        )
    else:
        analysis.evidence_valid = True

    return analysis


def derive_status(analysis: RequirementAnalysis) -> RequirementAnalysis:
    """Compute status in Python. The model never sets this field.

    Thresholds come from config so status and the router's verdict stay in
    agreement — a requirement badged "matched" is one the router will accept.
    """
    ev = analysis.evidence_valid
    conf = analysis.confidence
    score = analysis.score

    if ev is False or conf < config.CONFIDENCE_FLOOR:
        analysis.status = "uncertain"
    elif score >= config.REQUIRED_ACCEPT_SCORE:
        analysis.status = "matched"
    elif score >= config.PARTIAL_MATCH_SCORE:
        analysis.status = "partially_matched"
    else:
        analysis.status = "missing"

    return analysis


def calibrate_confidence(
    analysis: RequirementAnalysis,
    full_resume_audit_completed: bool,
) -> RequirementAnalysis:
    """Turn confidence into a deterministic routing signal after reflection.

    Model confidence is retained for genuinely uncertain cases. It is lifted to
    the routing floor only when all independent safeguards succeeded: evidence
    state is valid, hybrid retrieval was not flagged as weak, and the critic
    completed a full-resume audit. For a zero score, valid empty evidence means
    the scorer made no positive claim and the full-resume audit found no reason
    to correct it.
    """
    if (
        full_resume_audit_completed
        and analysis.evidence_valid is True
        and not analysis.low_retrieval_confidence
    ):
        analysis.confidence = max(analysis.confidence, config.CONFIDENCE_FLOOR)
    return derive_status(analysis)
