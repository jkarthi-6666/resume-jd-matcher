"""Application design tokens and global Streamlit style overrides."""

import streamlit as st


_GLOBAL_STYLES = """
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

html, body, [class*="css"] {
    font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
}

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

/* Header */
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

/* Metrics */
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

/* Expanders and containers */
div[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 8px;
}

div[data-testid="stExpander"] summary {
    font-weight: 500;
}

/* Full-width result tabs */
button[data-baseweb="tab"] {
    min-width: 11rem;
    padding: 0.8rem 1rem;
    font-weight: 600;
}

div[data-baseweb="tab-list"] {
    gap: 0.35rem;
    border-bottom: 1px solid var(--line);
}

/* Panel heading */
.panel-heading {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 0.1rem;
}

.panel-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--ink);
}

.panel-caption {
    font-size: 0.82rem;
    color: var(--ink-faint);
    margin-bottom: 1rem;
}

/* Status pill */
.pill {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    white-space: nowrap;
}

.pill.fresh {
    background: var(--good-bg);
    color: var(--good);
}

.pill.stale {
    background: var(--warn-bg);
    color: var(--warn);
}

.pill.empty {
    background: var(--neutral-bg);
    color: var(--ink-faint);
}

/* Verdict banner */
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

.verdict-banner .verdict-score {
    font-size: 1.15rem;
    font-weight: 700;
}

.verdict-banner.accept {
    background: var(--good-bg);
    border-color: rgba(30, 122, 76, 0.2);
}

.verdict-banner.accept .verdict-label,
.verdict-banner.accept .verdict-score {
    color: var(--good);
}

.verdict-banner.reject {
    background: var(--bad-bg);
    border-color: rgba(179, 49, 31, 0.2);
}

.verdict-banner.reject .verdict-label,
.verdict-banner.reject .verdict-score {
    color: var(--bad);
}

.verdict-banner.review {
    background: var(--warn-bg);
    border-color: rgba(138, 90, 16, 0.2);
}

.verdict-banner.review .verdict-label,
.verdict-banner.review .verdict-score {
    color: var(--warn);
}

/* Requirement status dot */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 0.55rem;
    vertical-align: middle;
}

.status-dot.matched {
    background: var(--good);
}

.status-dot.partially_matched {
    background: var(--warn);
}

.status-dot.missing {
    background: var(--bad);
}

.status-dot.uncertain {
    background: var(--ink-faint);
}

/* Section label */
.section-label {
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--ink-soft);
    margin: 0.25rem 0 0.75rem 0;
}

/* Requirement importance tag */
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

.tag.required {
    background: rgba(47, 95, 224, 0.10);
    color: var(--accent);
}

/* Reflection correction cards */
.correction-card {
    border-left: 3px solid var(--line);
    padding: 0.55rem 0 0.55rem 0.9rem;
    margin-bottom: 0.6rem;
}

.correction-card.raised {
    border-left-color: var(--good);
}

.correction-card.lowered {
    border-left-color: var(--warn);
}

.correction-card .correction-id {
    font-weight: 600;
    color: var(--ink);
    font-size: 0.88rem;
}

.correction-card .correction-scores {
    color: var(--ink-faint);
    font-size: 0.82rem;
    margin-left: 0.4rem;
}

.correction-card .correction-reason {
    color: var(--ink-soft);
    font-size: 0.86rem;
    margin-top: 0.2rem;
}

button[kind="primary"] {
    font-weight: 600;
}

button[kind="secondary"] {
    font-weight: 500;
}

@media (max-width: 700px) {
    .block-container {
        padding: 1.25rem 1rem 3rem;
    }

    button[data-baseweb="tab"] {
        min-width: auto;
    }

    .panel-heading {
        align-items: center;
        gap: 0.75rem;
    }
}
</style>
"""


def apply_global_styles() -> None:
    """Inject the application's design system once per Streamlit render."""
    st.markdown(_GLOBAL_STYLES, unsafe_allow_html=True)
