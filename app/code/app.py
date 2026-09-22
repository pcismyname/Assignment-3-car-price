"""Streamlit price-category app for AT82.03 Assignment 3.

A short two-step wizard collects car details and predicts one of four price
classes using the deployed from-scratch multinomial logistic regression bundle.
"""

from pathlib import Path

import streamlit as st

import ui
from helpers import STEP_BASICS, STEP_DETAILS
from predict import build_input_row, load_model, predict_category

st.set_page_config(page_title="Chaky's Price Garage — A3", page_icon="🚗", layout="centered")

MODEL_DIRECTORY = Path(__file__).resolve().parent
TOTAL_INPUT_STEPS = 2


@st.cache_resource
def _get_model():
    """Load the classifier bundle once per Streamlit process."""
    return load_model(MODEL_DIRECTORY)


bundle = _get_model()

ui.inject_css()
ui.site_header()

st.session_state.setdefault("step", 0)
st.session_state.setdefault("answers", {})


def go(step: int) -> None:
    st.session_state.step = step
    st.rerun()


def predict_from_answers() -> dict:
    answers = st.session_state.answers
    inputs = {field: answers.get(field) for field in STEP_BASICS + STEP_DETAILS}
    row = build_input_row(inputs)
    return predict_category(bundle, row)


step = st.session_state.step

if step == 0:
    ui.homepage()
    _, middle, _ = st.columns([1, 2, 1])
    if middle.button("Start an estimate", use_container_width=True):
        go(1)

elif step == 1:
    ui.step_dots(1, TOTAL_INPUT_STEPS)
    st.markdown(
        '<div class="section-title">The essentials</div>'
        '<div class="section-sub">Start with the details most buyers notice. '
        "Skip anything you don't know.</div>",
        unsafe_allow_html=True,
    )
    for field in STEP_BASICS:
        ui.render_field(field)
    back, next_step = st.columns(2)
    if back.button("Back", use_container_width=True):
        go(0)
    if next_step.button("Continue", use_container_width=True):
        go(2)

elif step == 2:
    ui.step_dots(2, TOTAL_INPUT_STEPS)
    st.markdown(
        '<div class="section-title">Under the bonnet</div>'
        '<div class="section-sub">These fields sharpen the estimate, but every one is optional.</div>',
        unsafe_allow_html=True,
    )
    for field in STEP_DETAILS:
        ui.render_field(field)
    back, calculate = st.columns(2)
    if back.button("Back", use_container_width=True):
        go(1)
    if calculate.button("Predict category", use_container_width=True):
        go(3)

else:
    with st.spinner("Checking the specifications and classifying the car..."):
        result = predict_from_answers()
    ui.render_result(result, st.session_state.answers)
    _, middle, _ = st.columns([1, 2, 1])
    if middle.button("Estimate another car", use_container_width=True):
        st.session_state.answers = {}
        go(0)
