import pandas as pd
import streamlit as st

from utils import api_get

st.set_page_config(page_title="Evolution to AI Agents", page_icon="\U0001F9E0", layout="wide")

st.title("Evolution to AI Agents — From Scratch")
st.caption(
    "Same character-generation task, four architectures, all running real server-side "
    "Python inference on real trained weights. Pick a page from the sidebar."
)

st.markdown(
    """
    | Project | Architecture | What it removes | Loss on the same corpus |
    |---|---|---|---|
    | 1 | MLP (NumPy, fixed 10-char window) | — baseline | 1.99 |
    | 2 | RNN / LSTM (PyTorch, hidden state) | fixed context window | 1.74 |
    | 3 | Transformer (PyTorch, self-attention) | sequential bottleneck | 1.24 |
    | 4 | RAG Agent (TF-IDF retrieval) | frozen/hallucinated knowledge | grounded, verified |

    Every number above comes from the exact same trained weights this app loads and runs -
    see the sidebar pages for live generation, live scoring, and a real attention heatmap.
    """
)

try:
    data = api_get("/metrics/comparison")
    st.subheader("Measured loss / perplexity, monotonically improving")
    df = pd.DataFrame(data["architectures"]).set_index("arch")[["perplexity"]]
    st.bar_chart(df)
except Exception as e:
    st.warning(f"Backend not reachable yet ({e}). Start it with `uvicorn backend.main:app` from the webapp/ folder.")

st.markdown("---")
st.caption(
    "Source: [github.com/shabeeehtesham](https://github.com/shabeeehtesham) — "
    "see INTERVIEW_PREP.md in the repo for the full write-up of every design decision below."
)
