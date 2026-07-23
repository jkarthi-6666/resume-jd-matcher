"""Session-state helpers for the Streamlit application."""

from typing import Any

import streamlit as st


_SESSION_DEFAULTS = {
    "naive_result": None,
    "naive_error": None,
    "full_result": None,
    "full_error": None,
    "naive_input_sig": None,
    "full_input_sig": None,
}


def initialize_session_state() -> None:
    """Create the application-owned session keys on the first render."""
    for key, default in _SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, default)


def input_signature(uploaded_file: Any, job_description: str) -> tuple | None:
    """Return a lightweight signature used to detect stale results."""
    if uploaded_file is None:
        return None
    return uploaded_file.file_id, hash(job_description.strip())


def panel_state(
    *,
    has_result: bool,
    has_error: bool,
    result_input_signature: tuple | None,
    current_input_signature: tuple | None,
) -> str:
    """Describe whether a result panel is empty, stale, or current."""
    if has_error or not has_result:
        return "empty"
    if (
        current_input_signature is not None
        and result_input_signature != current_input_signature
    ):
        return "stale"
    return "fresh"

