import streamlit as st

from utils import api_post, render_scored_text

st.set_page_config(page_title="MLP Playground", page_icon="\U0001F9E0")
st.title("Project 1 — MLP: fixed 10-character window")
st.caption(
    "The model only ever sees the last 10 characters. Type a sentence and every "
    "character gets retroactively graded: 'if I only saw the 10 characters before "
    "this one, would I have guessed it?' Real forward pass, real trained weights."
)

text = st.text_area("Type a sentence", value="To be or not to be, that is the question.", height=100)

if st.button("Score this text", type="primary"):
    with st.spinner("Running the real NumPy forward pass..."):
        data = api_post("/mlp/score", {"text": text})
    render_scored_text(data["results"])
    st.caption("Colored by rank of the true next character among the model's 66-way prediction — green = top guess, red = far down the list.")

st.markdown("---")
st.subheader("Continue writing")
col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.1)
with col2:
    num_samples = st.slider("Samples", 1, 5, 3)

if st.button("Generate"):
    with st.spinner("Sampling..."):
        data = api_post("/mlp/generate", {"num_samples": num_samples, "temperature": temperature})
    for s in data["samples"]:
        st.code(s)
    st.caption(f"val_loss={data['val_loss']}  val_acc={data['val_acc']}")
