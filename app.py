"""Streamlit entry point for the Resume to JD Matcher."""

import streamlit as st

st.set_page_config(
    page_title="Resume Screening Workspace",
    page_icon=":page_facing_up:",
    layout="wide",
)

# Streamlit requires set_page_config to be the first UI command.
try:
    from src import pipeline
    from src.config import validate_config
    from ui.layout import render_header, render_input_panel
    from ui.results import render_results_workspace
    from ui.state import initialize_session_state, input_signature
    from ui.styles import apply_global_styles
except ImportError as error:
    st.error(f"Import error: {error}")
    st.stop()


def _configuration_error() -> str | None:
    try:
        validate_config()
    except EnvironmentError as error:
        return str(error)
    return None


def _run_naive(inputs, current_signature: tuple | None) -> None:
    if not (inputs.run_naive and inputs.ready):
        return

    with st.spinner("Running naive baseline…"):
        try:
            st.session_state["naive_result"] = pipeline.run_naive(
                inputs.uploaded_file.getvalue(),
                inputs.job_description,
            )
            st.session_state["naive_error"] = None
        except (ValueError, TimeoutError) as error:
            st.session_state["naive_result"] = None
            st.session_state["naive_error"] = str(error)
        st.session_state["naive_input_sig"] = current_signature


def _run_full(inputs, current_signature: tuple | None) -> None:
    if not (inputs.run_full and inputs.ready):
        return

    progress_bar = st.progress(0, text="Starting full pipeline…")

    def on_progress(stage: str, done: int, total: int) -> None:
        progress_bar.progress(done / total, text=f"{stage}…")

    try:
        report, debug = pipeline.run_full(
            inputs.uploaded_file.getvalue(),
            inputs.job_description,
            progress_cb=on_progress,
        )
        st.session_state["full_result"] = (report, debug)
        st.session_state["full_error"] = None
    except (ValueError, TimeoutError) as error:
        st.session_state["full_result"] = None
        st.session_state["full_error"] = str(error)
    finally:
        progress_bar.empty()
        st.session_state["full_input_sig"] = current_signature


def main() -> None:
    apply_global_styles()
    render_header()

    config_error = _configuration_error()
    if config_error:
        st.error(f"Configuration error: {config_error}")
        st.info("Copy .env.example to .env and add your API key.")
        st.stop()

    inputs = render_input_panel()
    initialize_session_state()
    current_signature = input_signature(
        inputs.uploaded_file,
        inputs.job_description,
    )

    _run_naive(inputs, current_signature)
    _run_full(inputs, current_signature)
    render_results_workspace(current_signature)


main()
