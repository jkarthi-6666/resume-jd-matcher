"""Streamlit UI for the Resume to JD Matcher."""
from html import escape

import streamlit as st

st.set_page_config(
    page_title="Resume Screening Workspace",
    page_icon=":page_facing_up:",
    layout="wide",
)

# ── Imports (after page config) ──────────────────────────────────────────────
try:
    from src.config import validate_config
    from src import pipeline
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


# ── Design system ─────────────────────────────────────────────────────────────
# Neutral ink/paper palette with a single accent; status is conveyed through
# color + text labels, never emoji.
st.markdown(
    """
    <style>
    :root {
        --ink: #1a1d23;
        --ink-soft: #55596b;
        --ink-faint: #8a8fa3;
        --line: rgba(30, 33, 45, 0.10);
        --paper-raised: rgba(30, 33, 45, 0.025);
        --accent: #2f5fe0;
        --good: #1e7a4c;
        --good-bg: #e9f6ef;
        --bad: #b3311f;
        --bad-bg: #fbebe9;
        --warn: #8a5a10;
        --warn-bg: #fbf1de;
        --neutral-bg: #eef0f4;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --ink: #e8e9ee;
            --ink-soft: #aeb2c4;
            --ink-faint: #7d8299;
            --line: rgba(232, 233, 238, 0.12);
            --paper-raised: rgba(232, 233, 238, 0.035);
            --accent: #7ea1ff;
            --good: #5fd39a;
            --good-bg: rgba(95, 211, 154, 0.12);
            --bad: #f28b7d;
            --bad-bg: rgba(242, 139, 125, 0.12);
            --warn: #e8c07d;
            --warn-bg: rgba(232, 192, 125, 0.12);
            --neutral-bg: rgba(232, 233, 238, 0.08);
        }
    }

    html, body, [class*="css"] { font-family: "Inter", "Helvetica Neue", Arial, sans-serif; }
    .block-container {
        width: 100%;
        max-width: none;
        padding: 2rem clamp(1.25rem, 3.5vw, 4rem) 4rem;
    }

    /* Keep long model output and validation messages inside the viewport. */
    [data-testid="stAlert"] p,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stExpanderDetails"] {
        overflow-wrap: anywhere;
        word-break: normal;
    }

    /* ── Header ── */
    .app-eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--ink-faint);
        margin-bottom: 0.35rem;
    }
    .app-title {
        font-size: clamp(1.8rem, 3vw, 2.5rem);
        line-height: 1.12;
        font-weight: 750;
        color: var(--ink);
        letter-spacing: -0.035em;
    }
    .app-subtitle {
        max-width: 760px;
        font-size: 1rem;
        line-height: 1.6;
        color: var(--ink-soft);
        margin-top: 0.55rem;
    }

    /* ── Metrics ── */
    div[data-testid="stMetric"] {
        background: var(--paper-raised);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.75rem 1rem 0.6rem 1rem;
    }
    div[data-testid="stMetricLabel"] {
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.68rem !important;
        color: var(--ink-faint) !important;
    }

    /* ── Expanders / containers ── */
    div[data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 8px;
    }
    div[data-testid="stExpander"] summary { font-weight: 500; }

    /* ── Full-width result tabs ── */
    button[data-baseweb="tab"] {
        min-width: 11rem;
        padding: 0.8rem 1rem;
        font-weight: 600;
    }
    div[data-baseweb="tab-list"] {
        gap: 0.35rem;
        border-bottom: 1px solid var(--line);
    }

    /* ── Panel heading ── */
    .panel-heading {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        margin-bottom: 0.1rem;
    }
    .panel-title { font-size: 1.05rem; font-weight: 700; color: var(--ink); }
    .panel-caption { font-size: 0.82rem; color: var(--ink-faint); margin-bottom: 1rem; }

    /* ── Status pill ── */
    .pill {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        white-space: nowrap;
    }
    .pill.fresh { background: var(--good-bg); color: var(--good); }
    .pill.stale { background: var(--warn-bg); color: var(--warn); }
    .pill.empty { background: var(--neutral-bg); color: var(--ink-faint); }

    /* ── Verdict banner ── */
    .verdict-banner {
        border-radius: 10px;
        padding: 0.9rem 1.15rem;
        margin-bottom: 0.75rem;
        border: 1px solid transparent;
    }
    .verdict-banner .verdict-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        display: block;
        margin-bottom: 0.15rem;
    }
    .verdict-banner .verdict-score { font-size: 1.15rem; font-weight: 700; }
    .verdict-banner.accept { background: var(--good-bg); border-color: rgba(30, 122, 76, 0.2); }
    .verdict-banner.accept .verdict-label, .verdict-banner.accept .verdict-score { color: var(--good); }
    .verdict-banner.reject { background: var(--bad-bg); border-color: rgba(179, 49, 31, 0.2); }
    .verdict-banner.reject .verdict-label, .verdict-banner.reject .verdict-score { color: var(--bad); }
    .verdict-banner.review { background: var(--warn-bg); border-color: rgba(138, 90, 16, 0.2); }
    .verdict-banner.review .verdict-label, .verdict-banner.review .verdict-score { color: var(--warn); }

    /* ── Status dot (requirement rows) ── */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 0.55rem;
        vertical-align: middle;
    }
    .status-dot.matched { background: var(--good); }
    .status-dot.partially_matched { background: var(--warn); }
    .status-dot.missing { background: var(--bad); }
    .status-dot.uncertain { background: var(--ink-faint); }

    /* ── Section label ── */
    .section-label {
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--ink-soft);
        margin: 0.25rem 0 0.75rem 0;
    }

    /* ── Requirement importance tag ── */
    .tag {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        padding: 0.1rem 0.45rem;
        border-radius: 3px;
        margin-left: 0.5rem;
        background: var(--neutral-bg);
        color: var(--ink-soft);
    }
    .tag.required { background: rgba(47, 95, 224, 0.10); color: var(--accent); }

    /* ── Correction cards ── */
    .correction-card {
        border-left: 3px solid var(--line);
        padding: 0.55rem 0 0.55rem 0.9rem;
        margin-bottom: 0.6rem;
    }
    .correction-card.raised { border-left-color: var(--good); }
    .correction-card.lowered { border-left-color: var(--warn); }
    .correction-card .correction-id { font-weight: 600; color: var(--ink); font-size: 0.88rem; }
    .correction-card .correction-scores { color: var(--ink-faint); font-size: 0.82rem; margin-left: 0.4rem; }
    .correction-card .correction-reason { color: var(--ink-soft); font-size: 0.86rem; margin-top: 0.2rem; }

    button[kind="primary"] { font-weight: 600; }
    button[kind="secondary"] { font-weight: 500; }

    @media (max-width: 700px) {
        .block-container { padding: 1.25rem 1rem 3rem; }
        button[data-baseweb="tab"] { min-width: auto; }
        .panel-heading { align-items: center; gap: 0.75rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Small render helpers ──────────────────────────────────────────────────────

def _pill(state: str) -> str:
    label = {"fresh": "Current", "stale": "Inputs changed", "empty": "Not run"}[state]
    return f'<span class="pill {state}">{label}</span>'


def _tag(importance: str) -> str:
    cls = "required" if importance == "required" else ""
    label = {
        "required": "Required",
        "preferred": "Preferred",
        "unknown": "Unspecified",
    }.get(importance, importance.replace("_", " ").title())
    return f'<span class="tag {cls}">{label}</span>'


# ── Result renderers ──────────────────────────────────────────────────────────

def _render_naive(result):
    score = result.match_score
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Match score", f"{score}/100")
    with col2:
        st.progress(min(max(score, 0), 100) / 100)
        st.markdown(
            f'<div class="panel-caption" style="margin-top:0.5rem">'
            f'{escape(result.recommendation)}</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="section-label">Matched skills — {len(result.matched_skills)}</div>', unsafe_allow_html=True)
        if result.matched_skills:
            for s in result.matched_skills:
                st.markdown(f"- {s}")
        else:
            st.caption("None reported.")
    with col2:
        st.markdown(f'<div class="section-label">Missing skills — {len(result.missing_skills)}</div>', unsafe_allow_html=True)
        if result.missing_skills:
            for s in result.missing_skills:
                st.markdown(f"- {s}")
        else:
            st.caption("None reported.")


def _verdict_banner(report):
    verdict = report.verdict
    score = report.match_score
    cls = {"accept": "accept", "reject": "reject"}.get(verdict, "review")
    label = {"accept": "Accept", "reject": "Reject"}.get(verdict, "Needs human review")

    st.markdown(
        f'<div class="verdict-banner {cls}">'
        f'<span class="verdict-label">{label}</span>'
        f'<span class="verdict-score">{score:.0f} / 100</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if verdict not in ("accept", "reject") and report.review_reason:
        st.caption(report.review_reason)

    st.markdown(f"**Recommendation.** {report.recommendation}")

    delta = report.reflection_delta
    delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
    col1, col2, col3 = st.columns(3)
    col1.metric("Pre-reflection score", f"{report.score_pre_reflection:.0f}")
    col2.metric("Post-reflection score", f"{report.score_post_reflection:.0f}", delta=delta_str)
    col3.metric("Corrections", len(report.corrections))


def _requirement_counts(report):
    counts = {"matched": 0, "partially_matched": 0, "missing": 0, "uncertain": 0}
    for a in report.all_requirements:
        key = a.status if a.status in counts else "uncertain"
        counts[key] += 1
    return counts


def _render_requirements(report, key_prefix: str):
    counts = _requirement_counts(report)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matched", counts["matched"])
    col2.metric("Partial", counts["partially_matched"])
    col3.metric("Missing", counts["missing"])
    col4.metric("Uncertain", counts["uncertain"])

    st.write("")
    status_filter = st.multiselect(
        "Filter by status",
        ["matched", "partially_matched", "missing", "uncertain"],
        default=["matched", "partially_matched", "missing", "uncertain"],
        format_func=lambda s: s.replace("_", " ").title(),
        key=f"{key_prefix}_req_status_filter",
    )
    importance_filter = st.radio(
        "Filter by importance", ["All", "Required only", "Preferred only"],
        horizontal=True, key=f"{key_prefix}_req_importance_filter",
    )

    for a in report.all_requirements:
        status = a.status or "uncertain"
        if status not in status_filter:
            continue
        if importance_filter == "Required only" and a.importance != "required":
            continue
        if importance_filter == "Preferred only" and a.importance != "preferred":
            continue

        invalid = a.evidence_valid is False
        dot = f'<span class="status-dot {status}"></span>'
        header = f"{a.requirement}"

        with st.expander(header, expanded=invalid):
            st.markdown(
                f'<div style="margin-top:-0.5rem; margin-bottom:0.75rem">'
                f'{dot}{status.replace("_", " ").title()}{_tag(a.importance)}'
                f'</div>',
                unsafe_allow_html=True,
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Score", f"{a.score:.2f}")
            col2.metric("Confidence", f"{a.confidence:.2f}")
            col3.metric("Status", status.replace("_", " ").title())

            if invalid:
                st.error("Evidence could not be verified against the resume text.")

            if a.evidence:
                st.markdown("**Evidence**")
                for ev in a.evidence:
                    st.markdown(f"> {ev}")
            else:
                st.markdown("_No evidence found._")

            st.markdown(f"**Reason.** {a.reason}")

            if a.low_retrieval_confidence:
                st.caption("Low retrieval confidence — this requirement may have been missed by retrieval.")


def _render_reflection(report):
    if not report.corrections:
        st.caption("No corrections made by the reflection stage.")
    else:
        raised  = [c for c in report.corrections if c.direction == "raised"]
        lowered = [c for c in report.corrections if c.direction == "lowered"]

        if raised:
            st.markdown(f'<div class="section-label">Raised — retrieval misses recovered ({len(raised)})</div>', unsafe_allow_html=True)
            for c in raised:
                st.markdown(
                    f'<div class="correction-card raised">'
                    f'<span class="correction-id">{c.requirement_id}</span>'
                    f'<span class="correction-scores">{c.old_score:.2f} &rarr; {c.new_score:.2f}</span>'
                    f'<div class="correction-reason">{escape(c.reason)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if c.new_evidence:
                    for ev in c.new_evidence:
                        st.markdown(f"> {ev}")

        if lowered:
            st.markdown(f'<div class="section-label">Lowered — inflated matches corrected ({len(lowered)})</div>', unsafe_allow_html=True)
            for c in lowered:
                st.markdown(
                    f'<div class="correction-card lowered">'
                    f'<span class="correction-id">{c.requirement_id}</span>'
                    f'<span class="correction-scores">{c.old_score:.2f} &rarr; {c.new_score:.2f}</span>'
                    f'<div class="correction-reason">{escape(c.reason)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    if report.reflection_notes:
        st.markdown('<div class="section-label">Reflection notes</div>', unsafe_allow_html=True)
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
    _verdict_banner(report)
    st.divider()
    st.markdown('<div class="section-label">Requirement analysis</div>', unsafe_allow_html=True)
    _render_requirements(report, key_prefix="full")
    st.divider()
    st.markdown('<div class="section-label">Reflection changes</div>', unsafe_allow_html=True)
    _render_reflection(report)
    st.divider()
    with st.expander("Debug panel", expanded=False):
        _render_debug(debug, report)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-eyebrow">Resume Screening</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">Resume / Job Description Matcher</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Evaluate candidate fit with evidence-backed requirement analysis, '
    'then compare it with a single-shot baseline.</div>',
    unsafe_allow_html=True,
)
st.write("")

if not API_OK:
    st.error(f"Configuration error: {API_ERROR}")
    st.info("Copy .env.example to .env and add your API key.")
    st.stop()

# ── Inputs ────────────────────────────────────────────────────────────────────
with st.container(border=True):
    col_l, col_r = st.columns(2)
    with col_l:
        uploaded_file = st.file_uploader("Resume (PDF)", type=["pdf"])
    with col_r:
        job_description = st.text_area(
            "Job description",
            height=220,
            placeholder="Paste the full job posting here…",
        )
        st.caption(
            "Bullets and pasted formatting are normalized automatically. "
            "Recruitment logistics are excluded from candidate matching."
        )

    inputs_ready = bool(uploaded_file and job_description.strip())
    run_col1, run_col2, spacer = st.columns([1, 1, 2])
    run_naive_btn = run_col1.button(
        "Run naive baseline", disabled=not inputs_ready, use_container_width=True,
    )
    run_full_btn = run_col2.button(
        "Run full pipeline", type="primary", disabled=not inputs_ready, use_container_width=True,
    )
    if not inputs_ready:
        spacer.caption("Upload a resume and paste a job description to enable analysis.")

# ── Session state ─────────────────────────────────────────────────────────────
st.session_state.setdefault("naive_result", None)
st.session_state.setdefault("naive_error", None)
st.session_state.setdefault("full_result", None)
st.session_state.setdefault("full_error", None)
st.session_state.setdefault("naive_input_sig", None)
st.session_state.setdefault("full_input_sig", None)

current_sig = None
if uploaded_file is not None:
    current_sig = (uploaded_file.file_id, hash(job_description.strip()))

if run_naive_btn and inputs_ready:
    pdf_bytes = uploaded_file.getvalue()
    with st.spinner("Running naive baseline…"):
        try:
            st.session_state["naive_result"] = pipeline.run_naive(pdf_bytes, job_description)
            st.session_state["naive_error"] = None
        except ValueError as e:
            st.session_state["naive_result"] = None
            st.session_state["naive_error"] = str(e)
        st.session_state["naive_input_sig"] = current_sig

if run_full_btn and inputs_ready:
    pdf_bytes = uploaded_file.getvalue()
    progress_bar = st.progress(0, text="Starting full pipeline…")

    def _on_progress(stage: str, done: int, total: int):
        progress_bar.progress(done / total, text=f"{stage}…")

    try:
        report, debug = pipeline.run_full(pdf_bytes, job_description, progress_cb=_on_progress)
        st.session_state["full_result"] = (report, debug)
        st.session_state["full_error"] = None
    except ValueError as e:
        st.session_state["full_result"] = None
        st.session_state["full_error"] = str(e)
    finally:
        progress_bar.empty()
        st.session_state["full_input_sig"] = current_sig

# ── Results: full-width tabbed workspace ──────────────────────────────────────
st.write("")


def _panel_state(has_result: bool, has_error: bool, input_sig) -> str:
    if has_error or not has_result:
        return "empty"
    if current_sig is not None and input_sig != current_sig:
        return "stale"
    return "fresh"


naive_state = _panel_state(
    st.session_state["naive_result"] is not None, bool(st.session_state["naive_error"]),
    st.session_state["naive_input_sig"],
)
full_state = _panel_state(
    st.session_state["full_result"] is not None, bool(st.session_state["full_error"]),
    st.session_state["full_input_sig"],
)

tab_full, tab_naive = st.tabs(["Full pipeline", "Naive baseline"])

with tab_full:
    st.markdown(
        f'<div class="panel-heading"><span class="panel-title">Full pipeline analysis</span>{_pill(full_state)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel-caption">Hybrid retrieval, evidence validation, adversarial reflection, '
        'and confidence-aware routing</div>',
        unsafe_allow_html=True,
    )
    if st.session_state["full_error"]:
        st.error(st.session_state["full_error"])
    elif st.session_state["full_result"] is not None:
        if full_state == "stale":
            st.warning("Inputs changed since this result was generated — re-run to refresh.")
        report, debug = st.session_state["full_result"]
        try:
            _render_full(report, debug)
        except Exception as e:
            st.error(f"Unexpected error while rendering: {e}")
            raise
    else:
        st.info("Run the full pipeline to generate an evidence-backed candidate assessment.")

with tab_naive:
    st.markdown(
        f'<div class="panel-heading"><span class="panel-title">Naive baseline</span>{_pill(naive_state)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel-caption">Single LLM call without retrieval, evidence verification, or reflection</div>',
        unsafe_allow_html=True,
    )
    if st.session_state["naive_error"]:
        st.error(st.session_state["naive_error"])
    elif st.session_state["naive_result"] is not None:
        if naive_state == "stale":
            st.warning("Inputs changed since this result was generated — re-run to refresh.")
        _render_naive(st.session_state["naive_result"])
    else:
        st.info("Run the naive baseline to compare it with the full pipeline.")
