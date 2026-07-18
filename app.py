"""Streamlit UI for the Resume to JD Matcher."""
import streamlit as st

st.set_page_config(
    page_title="Resume → JD Matcher",
    page_icon="📄",
    layout="wide",
)

# ── Imports (after page config) ──────────────────────────────────────────────
try:
    from src.config import validate_config
    from src import pipeline
    from src.parser import ImageOnlyPDF
    CONFIG_OK = True
    try:
        validate_config()
        API_OK = True
    except EnvironmentError as e:
        API_OK = False
        API_ERROR = str(e)
except ImportError as e:
    CONFIG_OK = False
    st.error(f"Import error: {e}")
    st.stop()


# ── Render functions (defined before they are called) ────────────────────────

def _render_naive(result):
    st.subheader("Naive Baseline Result")
    st.metric("Match Score", f"{result.match_score}/100")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Matched skills**")
        for s in result.matched_skills:
            st.markdown(f"- {s}")
    with col2:
        st.markdown("**Missing skills**")
        for s in result.missing_skills:
            st.markdown(f"- {s}")
    st.info(result.recommendation)


def _verdict_badge(report):
    verdict = report.verdict
    score   = report.match_score

    if verdict == "accept":
        st.success(f"✅ **ACCEPT** — Match score: {score:.0f}/100")
    elif verdict == "reject":
        st.error(f"❌ **REJECT** — Match score: {score:.0f}/100")
    else:
        st.warning(f"⚠️ **NEEDS HUMAN REVIEW** — Match score: {score:.0f}/100")
        if report.review_reason:
            st.warning(report.review_reason)

    st.markdown(f"**Recommendation:** {report.recommendation}")

    delta = report.reflection_delta
    delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
    col1, col2, col3 = st.columns(3)
    col1.metric("Pre-reflection score", f"{report.score_pre_reflection:.0f}")
    col2.metric("Post-reflection score", f"{report.score_post_reflection:.0f}", delta=delta_str)
    col3.metric("Corrections", len(report.corrections))


def _render_requirements(report):
    for a in report.all_requirements:
        status_icon = {
            "matched": "🟢",
            "partially_matched": "🟡",
            "missing": "🔴",
            "uncertain": "⚠️",
        }.get(a.status or "uncertain", "❓")

        importance_badge = "**[required]**" if a.importance == "required" else "_[preferred]_"
        invalid = a.evidence_valid is False

        header = f"{status_icon} {a.requirement} {importance_badge}"
        with st.expander(header, expanded=invalid):
            col1, col2, col3 = st.columns(3)
            col1.metric("Score", f"{a.score:.2f}")
            col2.metric("Confidence", f"{a.confidence:.2f}")
            col3.metric("Status", a.status or "—")

            if invalid:
                st.error("⚠️ Evidence could not be verified in the resume text.")

            if a.evidence:
                st.markdown("**Evidence:**")
                for ev in a.evidence:
                    st.markdown(f"> {ev}")
            else:
                st.markdown("_No evidence found._")

            st.markdown(f"**Reason:** {a.reason}")

            if a.low_retrieval_confidence:
                st.caption("⚡ Low retrieval confidence — this requirement may have been missed by retrieval.")


def _render_reflection(report):
    if not report.corrections:
        st.info("No corrections made by the reflection stage.")
    else:
        raised  = [c for c in report.corrections if c.direction == "raised"]
        lowered = [c for c in report.corrections if c.direction == "lowered"]

        if raised:
            st.markdown("**Raised scores (retrieval misses recovered)**")
            for c in raised:
                st.success(
                    f"**{c.requirement_id}** — {c.old_score:.2f} → {c.new_score:.2f}\n\n"
                    f"{c.reason}"
                )
                if c.new_evidence:
                    for ev in c.new_evidence:
                        st.markdown(f"> {ev}")

        if lowered:
            st.markdown("**Lowered scores (inflated matches corrected)**")
            for c in lowered:
                st.warning(
                    f"**{c.requirement_id}** — {c.old_score:.2f} → {c.new_score:.2f}\n\n"
                    f"{c.reason}"
                )

    if report.reflection_notes:
        st.markdown("**Reflection notes:**")
        for note in report.reflection_notes:
            st.caption(note)


def _render_debug(debug, report):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Resume text", "Chunks", "Planner output", "Retrieval detail", "Raw report JSON"
    ])

    with tab1:
        st.text_area("Extracted resume text", debug.resume_text, height=400)

    with tab2:
        st.write(f"Total chunks: {len(debug.chunks)}")
        for chunk in debug.chunks:
            with st.expander(f"[{chunk.chunk_id}] {chunk.header} ({chunk.section})", expanded=False):
                st.text(chunk.embed_text)
                st.caption(f"header_detected={chunk.header_detected}")

    with tab3:
        st.write(f"Requirements extracted: {len(debug.requirements)}")
        for req in debug.requirements:
            st.markdown(
                f"- `{req.id}` **{req.requirement}** "
                f"({req.category} / {req.importance})"
            )

    with tab4:
        for rd in debug.retrieval_debug:
            with st.expander(f"{rd['requirement_id']}: {rd['requirement'][:60]}", expanded=False):
                st.write(f"Top RRF score: {rd['top_rrf_score']:.4f}")
                st.write(f"Low retrieval confidence: {rd['low_retrieval_confidence']}")
                col1, col2, col3, col4 = st.columns(4)
                col1.markdown("**BM25**\n" + "\n".join(rd["bm25_ranked"]))
                col2.markdown("**Vector**\n" + "\n".join(rd["vec_ranked"]))
                col3.markdown("**RRF merged**\n" + "\n".join(rd["rrf_merged"]))
                col4.markdown("**Post-rerank**\n" + "\n".join(rd["post_rerank"]))

    with tab5:
        st.json(report.model_dump())


def _render_full(report, debug):
    _verdict_badge(report)
    st.divider()
    st.subheader("Requirement Analysis")
    _render_requirements(report)
    st.divider()
    st.subheader("Reflection Changes")
    _render_reflection(report)
    st.divider()
    with st.expander("Debug panel", expanded=False):
        _render_debug(debug, report)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Resume → Job Description Matcher")
st.caption("Hybrid RAG · Evidence validation · Adversarial reflection · Confidence routing")

if not API_OK:
    st.error(f"Configuration error: {API_ERROR}")
    st.info("Copy `.env.example` to `.env` and add your API key.")
    st.stop()

# ── Mode selector ─────────────────────────────────────────────────────────────
mode = st.radio(
    "Mode",
    ["Full pipeline (RAG + reflection + routing)", "Naive baseline (single LLM call)"],
    horizontal=True,
)
use_naive = "Naive" in mode

# ── Inputs ────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

with col_r:
    job_description = st.text_area(
        "Paste Job Description",
        height=300,
        placeholder="Paste the full job posting here…",
    )

analyse_btn = st.button("Analyse", type="primary", disabled=(not uploaded_file or not job_description.strip()))

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyse_btn and uploaded_file and job_description.strip():
    pdf_bytes = uploaded_file.read()

    with st.spinner("Analysing…"):
        try:
            if use_naive:
                result = pipeline.run_naive(pdf_bytes, job_description)
                _render_naive(result)
            else:
                report, debug = pipeline.run_full(pdf_bytes, job_description)
                _render_full(report, debug)
        except ImageOnlyPDF as e:
            st.error(str(e))
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            raise
