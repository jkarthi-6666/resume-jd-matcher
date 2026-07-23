"""Result workspace and report-specific Streamlit components."""

from html import escape
from typing import Any

import streamlit as st

from ui.state import panel_state


def _pill(state: str) -> str:
    label = {"fresh": "Current", "stale": "Inputs changed", "empty": "Not run"}[state]
    return f'<span class="pill {state}">{label}</span>'


def _tag(importance: str) -> str:
    css_class = "required" if importance == "required" else ""
    label = {
        "required": "Required",
        "preferred": "Preferred",
        "unknown": "Unspecified",
    }.get(importance, importance.replace("_", " ").title())
    return f'<span class="tag {css_class}">{label}</span>'


def _render_naive(result: Any) -> None:
    score = result.match_score
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Match score", f"{score}/100")
    with col2:
        st.progress(min(max(score, 0), 100) / 100)
        st.markdown(
            '<div class="panel-caption" style="margin-top:0.5rem">'
            f"{escape(result.recommendation)}</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="section-label">Matched skills — '
            f"{len(result.matched_skills)}</div>",
            unsafe_allow_html=True,
        )
        if result.matched_skills:
            for skill in result.matched_skills:
                st.markdown(f"- {skill}")
        else:
            st.caption("None reported.")
    with col2:
        st.markdown(
            '<div class="section-label">Missing skills — '
            f"{len(result.missing_skills)}</div>",
            unsafe_allow_html=True,
        )
        if result.missing_skills:
            for skill in result.missing_skills:
                st.markdown(f"- {skill}")
        else:
            st.caption("None reported.")


def _render_verdict_banner(report: Any) -> None:
    verdict = report.verdict
    css_class = {"accept": "accept", "reject": "reject"}.get(verdict, "review")
    label = {
        "accept": "Accept",
        "reject": "Reject",
    }.get(verdict, "Needs human review")

    st.markdown(
        f'<div class="verdict-banner {css_class}">'
        f'<span class="verdict-label">{label}</span>'
        f'<span class="verdict-score">{report.match_score:.0f} / 100</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    if verdict not in ("accept", "reject") and report.review_reason:
        st.caption(report.review_reason)

    st.markdown(f"**Recommendation.** {report.recommendation}")

    delta = report.reflection_delta
    delta_label = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
    col1, col2, col3 = st.columns(3)
    col1.metric("Pre-reflection score", f"{report.score_pre_reflection:.0f}")
    col2.metric(
        "Post-reflection score",
        f"{report.score_post_reflection:.0f}",
        delta=delta_label,
    )
    col3.metric("Corrections", len(report.corrections))


def _requirement_counts(report: Any) -> dict[str, int]:
    counts = {
        "matched": 0,
        "partially_matched": 0,
        "missing": 0,
        "uncertain": 0,
    }
    for analysis in report.all_requirements:
        key = analysis.status if analysis.status in counts else "uncertain"
        counts[key] += 1
    return counts


def _render_requirements(report: Any, key_prefix: str) -> None:
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
        format_func=lambda status: status.replace("_", " ").title(),
        key=f"{key_prefix}_req_status_filter",
    )
    importance_filter = st.radio(
        "Filter by importance",
        ["All", "Required only", "Preferred only"],
        horizontal=True,
        key=f"{key_prefix}_req_importance_filter",
    )

    for analysis in report.all_requirements:
        status = analysis.status or "uncertain"
        if status not in status_filter:
            continue
        if importance_filter == "Required only" and analysis.importance != "required":
            continue
        if importance_filter == "Preferred only" and analysis.importance != "preferred":
            continue

        invalid = analysis.evidence_valid is False
        dot = f'<span class="status-dot {status}"></span>'
        with st.expander(analysis.requirement, expanded=invalid):
            st.markdown(
                '<div style="margin-top:-0.5rem; margin-bottom:0.75rem">'
                f'{dot}{status.replace("_", " ").title()}'
                f"{_tag(analysis.importance)}</div>",
                unsafe_allow_html=True,
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Score", f"{analysis.score:.2f}")
            col2.metric("Confidence", f"{analysis.confidence:.2f}")
            col3.metric("Status", status.replace("_", " ").title())

            if invalid:
                st.error("Evidence could not be verified against the resume text.")

            if analysis.evidence:
                st.markdown("**Evidence**")
                for evidence in analysis.evidence:
                    st.markdown(f"> {evidence}")
            else:
                st.markdown("_No evidence found._")

            st.markdown(f"**Reason.** {analysis.reason}")
            if analysis.low_retrieval_confidence:
                st.caption(
                    "Low retrieval confidence — this requirement may have been "
                    "missed by retrieval."
                )


def _render_reflection(report: Any) -> None:
    if not report.corrections:
        st.caption("No corrections made by the reflection stage.")
    else:
        raised = [
            correction
            for correction in report.corrections
            if correction.direction == "raised"
        ]
        lowered = [
            correction
            for correction in report.corrections
            if correction.direction == "lowered"
        ]

        if raised:
            st.markdown(
                '<div class="section-label">Raised — retrieval misses recovered '
                f"({len(raised)})</div>",
                unsafe_allow_html=True,
            )
            for correction in raised:
                st.markdown(
                    '<div class="correction-card raised">'
                    f'<span class="correction-id">{correction.requirement_id}</span>'
                    '<span class="correction-scores">'
                    f"{correction.old_score:.2f} &rarr; {correction.new_score:.2f}"
                    "</span>"
                    '<div class="correction-reason">'
                    f"{escape(correction.reason)}</div></div>",
                    unsafe_allow_html=True,
                )
                for evidence in correction.new_evidence:
                    st.markdown(f"> {evidence}")

        if lowered:
            st.markdown(
                '<div class="section-label">Lowered — inflated matches corrected '
                f"({len(lowered)})</div>",
                unsafe_allow_html=True,
            )
            for correction in lowered:
                st.markdown(
                    '<div class="correction-card lowered">'
                    f'<span class="correction-id">{correction.requirement_id}</span>'
                    '<span class="correction-scores">'
                    f"{correction.old_score:.2f} &rarr; {correction.new_score:.2f}"
                    "</span>"
                    '<div class="correction-reason">'
                    f"{escape(correction.reason)}</div></div>",
                    unsafe_allow_html=True,
                )

    if report.reflection_notes:
        st.markdown(
            '<div class="section-label">Reflection notes</div>',
            unsafe_allow_html=True,
        )
        for note in report.reflection_notes:
            st.caption(note)


def _render_debug(debug: Any, report: Any) -> None:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Resume text",
            "Chunks",
            "Planner output",
            "Retrieval detail",
            "Raw report JSON",
        ]
    )

    with tab1:
        st.text_area("Extracted resume text", debug.resume_text, height=400)

    with tab2:
        st.write(f"Total chunks: {len(debug.chunks)}")
        for chunk in debug.chunks:
            with st.expander(
                f"[{chunk.chunk_id}] {chunk.header} ({chunk.section})",
                expanded=False,
            ):
                st.text(chunk.embed_text)

    with tab3:
        st.write(f"Requirements extracted: {len(debug.requirements)}")
        if debug.requirement_plan and debug.requirement_plan.dropped_requirement_count:
            st.warning(
                f"Dropped {debug.requirement_plan.dropped_requirement_count} "
                "planner requirement(s) with invalid job-description source spans."
            )
        for requirement in debug.requirements:
            st.markdown(
                f"- `{requirement.id}` **{requirement.requirement}** "
                f"({requirement.category} / {requirement.importance} / "
                f"{requirement.kind})"
            )
            st.caption(f"JD source: {requirement.source_span}")
            if requirement.query_terms:
                st.caption(f"BM25 aliases: {', '.join(requirement.query_terms)}")

    with tab4:
        for detail in debug.retrieval_debug:
            with st.expander(
                f"{detail['requirement_id']}: {detail['requirement'][:60]}",
                expanded=False,
            ):
                st.write(f"Top RRF score: {detail['top_rrf_score']:.4f}")
                st.write(
                    "Low retrieval confidence: "
                    f"{detail['low_retrieval_confidence']}"
                )
                col1, col2, col3, col4 = st.columns(4)
                col1.markdown("**BM25**\n" + "\n".join(detail["bm25_ranked"]))
                col2.markdown("**Vector**\n" + "\n".join(detail["vec_ranked"]))
                col3.markdown("**RRF merged**\n" + "\n".join(detail["rrf_merged"]))
                col4.markdown(
                    "**Post-rerank**\n" + "\n".join(detail["post_rerank"])
                )

    with tab5:
        st.json(report.model_dump())


def _render_full(report: Any, debug: Any) -> None:
    _render_verdict_banner(report)
    st.divider()
    st.markdown(
        '<div class="section-label">Requirement analysis</div>',
        unsafe_allow_html=True,
    )
    _render_requirements(report, key_prefix="full")
    st.divider()
    st.markdown(
        '<div class="section-label">Reflection changes</div>',
        unsafe_allow_html=True,
    )
    _render_reflection(report)
    st.divider()
    with st.expander("Debug panel", expanded=False):
        _render_debug(debug, report)


def render_results_workspace(current_input_signature: tuple | None) -> None:
    """Render both result tabs from the current Streamlit session state."""
    naive_state = panel_state(
        has_result=st.session_state["naive_result"] is not None,
        has_error=bool(st.session_state["naive_error"]),
        result_input_signature=st.session_state["naive_input_sig"],
        current_input_signature=current_input_signature,
    )
    full_state = panel_state(
        has_result=st.session_state["full_result"] is not None,
        has_error=bool(st.session_state["full_error"]),
        result_input_signature=st.session_state["full_input_sig"],
        current_input_signature=current_input_signature,
    )

    st.write("")
    tab_full, tab_naive = st.tabs(["Full pipeline", "Naive baseline"])

    with tab_full:
        st.markdown(
            '<div class="panel-heading"><span class="panel-title">'
            f"Full pipeline analysis</span>{_pill(full_state)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="panel-caption">Hybrid retrieval, evidence validation, '
            "adversarial reflection, and confidence-aware routing</div>",
            unsafe_allow_html=True,
        )
        if st.session_state["full_error"]:
            st.error(st.session_state["full_error"])
        elif st.session_state["full_result"] is not None:
            if full_state == "stale":
                st.warning(
                    "Inputs changed since this result was generated — re-run to "
                    "refresh."
                )
            report, debug = st.session_state["full_result"]
            try:
                _render_full(report, debug)
            except Exception as error:
                st.error(f"Unexpected error while rendering: {error}")
                raise
        else:
            st.info(
                "Run the full pipeline to generate an evidence-backed candidate "
                "assessment."
            )

    with tab_naive:
        st.markdown(
            '<div class="panel-heading"><span class="panel-title">'
            f"Naive baseline</span>{_pill(naive_state)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="panel-caption">Single LLM call without retrieval, '
            "evidence verification, or reflection</div>",
            unsafe_allow_html=True,
        )
        if st.session_state["naive_error"]:
            st.error(st.session_state["naive_error"])
        elif st.session_state["naive_result"] is not None:
            if naive_state == "stale":
                st.warning(
                    "Inputs changed since this result was generated — re-run to "
                    "refresh."
                )
            _render_naive(st.session_state["naive_result"])
        else:
            st.info("Run the naive baseline to compare it with the full pipeline.")
