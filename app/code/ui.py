"""Streamlit view layer for the Assignment 3 price-category app."""

import streamlit as st

from helpers import (
    CLASS_BLURB,
    NOT_SURE,
    NUMERIC_SPECS,
    SELECT_SPECS,
    format_inr,
    format_range,
    resolve_choice,
    resolve_numeric,
)

# Colour per class so the result card reads at a glance (cheap → expensive).
CLASS_COLOR = {
    "Budget": "#2A9D8F",
    "Mid-range": "#457B9D",
    "Premium": "#E9C46A",
    "Luxury": "#E76F51",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 760px; }
        .hero-title { font-size: 2.1rem; font-weight: 800; margin-bottom: .2rem; }
        .hero-sub { color: #6b7280; font-size: 1.05rem; margin-bottom: 1.4rem; }
        .section-title { font-size: 1.25rem; font-weight: 700; margin: .4rem 0 .1rem; }
        .section-sub { color: #6b7280; margin-bottom: 1rem; }
        .result-card { border-radius: 16px; padding: 1.6rem 1.8rem; color: white; }
        .result-label { font-size: 2rem; font-weight: 800; }
        .result-range { font-size: 1.15rem; opacity: .95; margin-top: .2rem; }
        .result-conf { font-size: .95rem; opacity: .85; margin-top: .6rem; }
        .dots { letter-spacing: .3rem; color: #cbd5e1; margin-bottom: .6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def site_header() -> None:
    st.markdown('<div class="hero-title">🚗 Chaky\'s Price Garage — A3</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Tell us about a used car and we\'ll predict its '
        '<b>price category</b> with a from-scratch multinomial logistic regression.</div>',
        unsafe_allow_html=True,
    )


def homepage() -> None:
    st.markdown(
        "This app classifies a used car into one of four price bands learned from the "
        "Indian used-car dataset:"
    )
    for label, blurb in CLASS_BLURB.items():
        st.markdown(f"- **{label}** — {blurb}")
    st.markdown("Answer a couple of quick steps — skip anything you don't know.")


def step_dots(current: int, total: int) -> None:
    dots = " ".join("●" if i <= current else "○" for i in range(1, total + 1))
    st.markdown(f'<div class="dots">{dots}</div>', unsafe_allow_html=True)


def render_field(field: str) -> None:
    """Render one numeric or select field and store its resolved value."""
    answers = st.session_state.answers
    if field in NUMERIC_SPECS:
        label, emoji, low, high, default, step = NUMERIC_SPECS[field]
        skipped = st.checkbox(f"{NOT_SURE} — {label}", key=f"skip_{field}")
        value = st.number_input(
            f"{emoji} {label}", min_value=low, max_value=high, value=default,
            step=step, key=f"num_{field}", disabled=skipped,
        )
        answers[field] = resolve_numeric(skipped, value)
    else:
        label, emoji, options = SELECT_SPECS[field]
        choice = st.selectbox(f"{emoji} {label}", [NOT_SURE, *options], key=f"sel_{field}")
        answers[field] = resolve_choice(choice)


def render_result(result: dict, answers: dict) -> None:
    """Show the predicted price class as a coloured card."""
    label = result["label"]
    color = CLASS_COLOR.get(label, "#457B9D")
    range_text = format_range(result["low"], result["high"])
    st.markdown(
        f"""
        <div class="result-card" style="background:{color};">
          <div>Predicted price category</div>
          <div class="result-label">{label}</div>
          <div class="result-range">Typical price: {range_text}</div>
          <div class="result-conf">Model confidence: {result['confidence']*100:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("How sure is the model about each class?"):
        for class_id, prob in enumerate(result["probabilities"]):
            name = ["Budget", "Mid-range", "Premium", "Luxury"][class_id]
            st.write(f"{name}")
            st.progress(min(1.0, float(prob)))
    st.caption(
        "Class boundaries are the price quartiles of the training data; "
        "predictions come from a hand-implemented softmax classifier."
    )
