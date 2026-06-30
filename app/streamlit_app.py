"""Streamlit UI for the Conversation Facet Evaluator.

Two modes:
  • Browse — pick a pre-scored conversation and explore its turn × facet matrix.
  • Live   — type a turn and score it on the spot (needs Ollama running).

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the package importable when run via `streamlit run` (no PYTHONPATH).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet_eval import config  # noqa: E402

SCORES_DIR = config.OUTPUT_DIR / "scores"

st.set_page_config(page_title="Conversation Facet Evaluator", layout="wide")


# --- helpers -------------------------------------------------------------

def _scored_conversations() -> list[str]:
    return sorted(p.stem for p in SCORES_DIR.glob("*.json") if not p.name.startswith("_"))


def _matrix_df(result: dict) -> pd.DataFrame:
    """facet (row) × turn (col) score matrix, most-active facets first."""
    cells: dict[str, dict] = {}
    label: dict[str, str] = {}
    for t in result["turns"]:
        for r in t["scores"]:
            fid = r["facet_id"]
            label[fid] = f"{fid} · {r['facet_name']}"
            cells.setdefault(fid, {})[t["index"]] = r["score"]
    rows = sorted(cells, key=lambda f: max((s or 0) for s in cells[f].values()), reverse=True)
    data = {label[f]: {i: cells[f].get(i) for i in range(result["n_turns"])} for f in rows}
    df = pd.DataFrame(data).T
    df.columns = [f"turn {i}" for i in df.columns]
    return df


def _color(v):
    if pd.isna(v):
        return "background-color:#f5f5f5;color:#bbb"
    g = int(40 + 215 * (1 - (v - 1) / 4))  # 1->light, 5->deep
    return f"background-color:rgb(255,{g},{g})"


# --- sidebar -------------------------------------------------------------

st.sidebar.title("Facet Evaluator")
mode = st.sidebar.radio("Mode", ["Browse scored", "Live score a turn"])
st.sidebar.caption(
    f"model `{config.OLLAMA_MODEL}` · top-K {config.ROUTER_TOP_K} · "
    f"safety floor {config.ROUTER_SAFETY_FLOOR}"
)


# --- browse mode ---------------------------------------------------------

if mode == "Browse scored":
    convs = _scored_conversations()
    if not convs:
        st.warning("No scored conversations found in output/scores/. "
                   "Run `python -m facet_eval.pipeline --all` first.")
        st.stop()

    cid = st.sidebar.selectbox("Conversation", convs)
    result = json.loads((SCORES_DIR / f"{cid}.json").read_text())

    st.title(f"🗂️ {cid}")
    rescued = sum(len(t.get("rescued_by_safety_floor", [])) for t in result["turns"])
    missing = sum(t["n_missing"] for t in result["turns"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("turns", result["n_turns"])
    c2.metric("facet-scores", sum(len(t["scores"]) for t in result["turns"]))
    c3.metric("safety-floor rescues", rescued)
    c4.metric("unscored (None)", missing)

    st.subheader("Conversation")
    for t in result["turns"]:
        st.markdown(f"**{t['index']} · {t['speaker']}** — {t['text']}")

    st.subheader("Turn × facet score matrix")
    st.caption("1 = absent … 5 = dominant · blank = facet not routed for that turn")
    df = _matrix_df(result)
    st.dataframe(df.style.map(_color).format("{:.0f}", na_rep=""),
                 width="stretch", height=560)

    with st.expander("Per-turn detail (sorted by score, with confidence + provenance)"):
        ti = st.slider("turn", 0, result["n_turns"] - 1, 0)
        turn = result["turns"][ti]
        st.markdown(f"> {turn['speaker']}: {turn['text']}")
        rows = sorted(turn["scores"],
                      key=lambda r: (-(r["score"] or 0), -(r["confidence"] or 0)))
        st.dataframe(pd.DataFrame([
            {"facet_id": r["facet_id"], "facet": r["facet_name"],
             "category": r["category"], "score": r["score"],
             "confidence (self-reported)": r["confidence"], "via": r["via"]}
            for r in rows
        ]), width="stretch", height=420, hide_index=True)
        st.caption("Confidence is the model's self-report (uncalibrated). Rows are "
                   "sorted highest-score first.")


# --- live mode -----------------------------------------------------------

else:
    st.title("⚡ Live score a turn")
    st.caption("Routes the turn (top-K + safety floor) and scores it with the local LLM. "
               "Requires Ollama running with the configured model.")
    turn = st.text_area("Conversation turn", height=120,
                        value="I can't stop shaking and I feel completely hopeless.")
    if st.button("Score", type="primary"):
        from facet_eval.router import route          # imported lazily (loads models)
        from facet_eval.scorer import score
        with st.spinner("routing + scoring…"):
            batch = route(turn)
            results = score(turn, batch)
        meta = {f["facet_id"]: f for f in batch}
        rows = sorted(results, key=lambda r: (-(r["score"] or 0), -(r["confidence"] or 0)))
        st.success(f"scored {len(results)} routed facets — highest-scoring first")
        df = pd.DataFrame([
            {"facet_id": r["facet_id"], "facet": r["facet_name"],
             "category": meta[r["facet_id"]]["category"],
             "via": meta[r["facet_id"]]["via"], "score": r["score"],
             "confidence (self-reported)": r["confidence"]}
            for r in rows
        ])
        st.markdown("**Top 10 facets**")
        st.dataframe(df.head(10), width="stretch", hide_index=True)
        with st.expander(f"Show all {len(df)} routed facets"):
            st.dataframe(df, width="stretch", height=500, hide_index=True)
        st.caption("⚠️ Confidence is the model's **self-report and uncalibrated** "
                   "(varies but skewed high). On a 3B, expect some marginal facets "
                   "to over-score (4–5); the high-signal facets lead the sorted list.")
