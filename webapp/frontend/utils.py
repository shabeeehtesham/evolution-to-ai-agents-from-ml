import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def api_post(path, json_body, timeout=30):
    resp = requests.post(f"{BACKEND_URL}{path}", json=json_body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def api_get(path, timeout=15):
    resp = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def rank_color(rank):
    """Mirrors mlpRankColor from the original browser demo: green for a
    top guess, fading through orange to red the further down the model's
    ranked predictions the true character fell."""
    if rank == 1:
        return "#22c55e"
    if rank <= 3:
        return "#84cc16"
    if rank <= 10:
        return "#eab308"
    if rank <= 20:
        return "#f97316"
    return "#ef4444"


def render_scored_text(results):
    """Renders a list of {"char", "rank", ...} dicts as HTML spans colored
    by rank - the server-side twin of the live browser demo's colored,
    retroactively-graded text."""
    spans = []
    for r in results:
        ch = r["char"]
        display = ch if ch not in ("\n", " ") else ("&#10;" if ch == "\n" else "&nbsp;")
        if ch == "\n":
            spans.append("<br/>")
            continue
        color = rank_color(r["rank"])
        title = f"rank #{r['rank']}, p={r['prob']:.3f}"
        spans.append(f'<span title="{title}" style="background:{color}22;border-bottom:2px solid {color};">{display}</span>')
    html = "".join(spans)
    st.markdown(
        f'<div style="font-family: ui-monospace, monospace; font-size:1.05rem; line-height:1.8; white-space:pre-wrap;">{html}</div>',
        unsafe_allow_html=True,
    )
