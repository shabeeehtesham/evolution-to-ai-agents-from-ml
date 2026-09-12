import plotly.express as px
import streamlit as st

from utils import api_post

st.set_page_config(page_title="Transformer Attention", page_icon="\U0001F9E0", layout="wide")
st.title("Project 3 — Transformer: real self-attention weights")
st.caption(
    "Real post-softmax attention weights from the same forward pass that powers "
    "generation — pick a layer/head and see exactly which earlier characters the "
    "model looked at to predict each next one."
)

prompt = st.text_input("Prompt (max 64 chars, causal window)", value="To be or not to")

if st.button("Compute attention", type="primary"):
    with st.spinner("Running the real causal self-attention forward pass..."):
        data = api_post("/transformer/attention", {"prompt": prompt})
    st.session_state["attn_data"] = data

if "attn_data" in st.session_state:
    data = st.session_state["attn_data"]
    tokens = data["tokens"]
    n_layer = len(data["attentions"])
    n_head = len(data["attentions"][0]) if n_layer else 0

    c1, c2 = st.columns(2)
    layer = c1.selectbox("Layer", list(range(n_layer)))
    head = c2.selectbox("Head", list(range(n_head)))

    matrix = data["attentions"][layer][head]
    fig = px.imshow(
        matrix,
        x=[repr(t)[1:-1] for t in tokens],
        y=[repr(t)[1:-1] for t in tokens],
        labels=dict(x="key (attended to)", y="query (attending from)", color="attention"),
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Lower-triangular by construction: causal masking gives every position an "
        "attention weight of exactly 0 toward anything after it, so the upper "
        "triangle is always blank."
    )

st.markdown("---")
st.subheader("Continue writing")
col1, col2, col3 = st.columns(3)
with col1:
    gen_prompt = st.text_input("Seed prompt", value="\n")
with col2:
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
with col3:
    top_k = st.slider("Top-k", 1, 30, 10)

if st.button("Generate"):
    with st.spinner("Sampling autoregressively..."):
        data = api_post("/transformer/generate", {"prompt": gen_prompt, "temperature": temperature, "top_k": top_k})
    st.code(data["text"])
    st.caption(f"train_loss={data['train_loss']}")
