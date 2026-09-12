import pandas as pd
import streamlit as st

from utils import api_post

st.set_page_config(page_title="RNN vs LSTM", page_icon="\U0001F9E0")
st.title("Project 2 — RNN vs LSTM: does gated memory actually help?")
st.caption(
    "Teacher-forced, zero-randomness comparison: both models read the exact same real "
    "text, character by character, and report their own probability for the actual next "
    "character at every step. Architecture is the only variable."
)

default_excerpt = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
    "All:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?"
)
text = st.text_area("Text to read (teacher-forced)", value=default_excerpt, height=140)

if st.button("Run the read-along", type="primary"):
    with st.spinner("Running both models' real forward pass..."):
        data = api_post("/rnn-lstm/read-along", {"text": text})
    df = pd.DataFrame({
        "RNN confidence": data["rnn_confidence"],
        "LSTM confidence": data["lstm_confidence"],
    })
    st.line_chart(df)
    c1, c2 = st.columns(2)
    c1.metric("RNN avg confidence", f"{data['rnn_avg']:.3f}")
    c2.metric("LSTM avg confidence", f"{data['lstm_avg']:.3f}")
    st.caption(
        "This is an honest, unfiltered run on whatever text you pasted — not a "
        "pre-selected excerpt. On short or unusual text the LSTM won't always win; "
        "see INTERVIEW_PREP.md for why that's the correct, non-cherry-picked finding."
    )

st.markdown("---")
st.subheader("Generate text")
arch = st.radio("Architecture", ["lstm", "rnn"], horizontal=True)
col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.1)
with col2:
    length = st.slider("Length", 50, 400, 200, 10)

if st.button("Generate"):
    with st.spinner("Sampling..."):
        data = api_post("/rnn-lstm/generate", {"arch": arch, "length": length, "temperature": temperature})
    st.code(data["text"])
    st.caption(f"train_loss={data['train_loss']}")
