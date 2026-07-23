"""Top-level layout components shared by the application."""

from dataclasses import dataclass
from typing import Any

import streamlit as st


@dataclass(frozen=True)
class AnalysisInputs:
    """Values and actions emitted by the input panel."""

    uploaded_file: Any
    job_description: str
    run_naive: bool
    run_full: bool
    ready: bool


def render_header() -> None:
    """Render the application heading and supporting copy."""
    st.markdown(
        '<div class="app-eyebrow">Resume Screening</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="app-title">Resume / Job Description Matcher</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="app-subtitle">Evaluate candidate fit with evidence-backed '
        "requirement analysis, then compare it with a single-shot baseline.</div>",
        unsafe_allow_html=True,
    )
    st.write("")


def render_input_panel() -> AnalysisInputs:
    """Render resume/JD inputs and return the requested analysis action."""
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

        ready = bool(uploaded_file and job_description.strip())
        run_col1, run_col2, spacer = st.columns([1, 1, 2])
        run_naive = run_col1.button(
            "Run naive baseline",
            disabled=not ready,
            use_container_width=True,
        )
        run_full = run_col2.button(
            "Run full pipeline",
            type="primary",
            disabled=not ready,
            use_container_width=True,
        )
        if not ready:
            spacer.caption(
                "Upload a resume and paste a job description to enable analysis."
            )

    return AnalysisInputs(
        uploaded_file=uploaded_file,
        job_description=job_description,
        run_naive=run_naive,
        run_full=run_full,
        ready=ready,
    )

