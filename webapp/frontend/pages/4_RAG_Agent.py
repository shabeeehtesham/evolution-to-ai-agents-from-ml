import streamlit as st

from utils import api_post

st.set_page_config(page_title="RAG Agent", page_icon="\U0001F9E0")
st.title("Project 4 — RAG Agent: grounded, not hallucinated")
st.caption(
    "Retrieve (TF-IDF cosine similarity over 17 real, cross-checked Shakespeare facts) "
    "→ Augment (retrieved facts pasted into the actual prompt template) → "
    "Generate (answer using only what was retrieved)."
)

question = st.text_input("Ask a question", value="Who is Iago and what does he do?")

if st.button("Ask", type="primary"):
    with st.spinner("Retrieving and generating..."):
        data = api_post("/rag/query", {"question": question})

    st.subheader("1. Retrieve")
    for r in data["retrieved"]:
        st.markdown(f"**score {r['score']:.4f}** — {r['text']}")

    st.subheader("2 & 3. Augment + Generate")
    st.write(data["answer"])

st.markdown("---")
st.caption(
    "Try a question outside the 17-fact database (e.g. 'What is the capital of "
    "France?') — it correctly reports no match instead of inventing an answer."
)
