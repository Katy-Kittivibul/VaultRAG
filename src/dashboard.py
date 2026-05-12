import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="RAG Evaluation Framework", layout="wide")

COSINE_CSV = Path("results/eval_cosine.csv")
HYBRID_CSV = Path("results/eval_hybrid.csv")
COMPARISON_JSON = Path("results/retriever_comparison.json")

METRICS = {
    "context_recall": "Context Recall",
    "faithfulness": "Faithfulness",
    "factual_correctness": "Factual Correctness",
    "answer_relevancy": "Answer Relevancy",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("RAG Evaluation Framework")
st.sidebar.divider()
st.sidebar.markdown("**Models**")
st.sidebar.markdown("- **LLM**: Ollama Mistral 7B")
st.sidebar.markdown("- **Embeddings**: MiniLM-L6-v2")
st.sidebar.divider()
st.sidebar.markdown("**Vault Stats**")
st.sidebar.markdown("- **Files**: 76")
st.sidebar.markdown("- **Chunks**: 178")
st.sidebar.divider()
st.sidebar.markdown("Run `python -m src.evaluator` to regenerate results.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _available_metrics(df: pd.DataFrame) -> list[str]:
    return [k for k in METRICS if k in df.columns]


# ── Section 1: Evaluation Results ────────────────────────────────────────────
st.header("Evaluation Results")

cosine_df = _load_csv(COSINE_CSV)
hybrid_df = _load_csv(HYBRID_CSV)

if cosine_df is None and hybrid_df is None:
    st.warning("No evaluation results found. Run `python -m src.evaluator` first.")
else:
    for label, df in [("Cosine Retriever", cosine_df), ("Hybrid Retriever", hybrid_df)]:
        if df is None:
            continue
        st.subheader(label)
        cols = st.columns(4)
        for col, (key, display) in zip(cols, METRICS.items()):
            val = df[key].mean() if key in df.columns else None
            col.metric(display, f"{val:.3f}" if val is not None else "N/A")

    if cosine_df is not None and hybrid_df is not None:
        shared = [k for k in METRICS if k in cosine_df.columns and k in hybrid_df.columns]
        if shared:
            st.subheader("Cosine vs Hybrid — Mean Scores")
            chart_df = pd.DataFrame(
                {
                    "Cosine": [cosine_df[m].mean() for m in shared],
                    "Hybrid": [hybrid_df[m].mean() for m in shared],
                },
                index=[METRICS[m] for m in shared],
            )
            st.bar_chart(chart_df)

st.divider()

# ── Section 2: Retriever Comparison ──────────────────────────────────────────
st.header("Retriever Comparison")

if COMPARISON_JSON.exists():
    raw = json.loads(COMPARISON_JSON.read_text(encoding="utf-8"))
    table = pd.DataFrame([
        {
            "Question": row["question"],
            "Cosine Sources": ", ".join(
                Path(s).name for s in row.get("cosine_sources", [])
            ),
            "Hybrid Sources": ", ".join(
                Path(s).name for s in row.get("hybrid_sources", [])
            ),
        }
        for row in raw
    ])
    st.dataframe(table, use_container_width=True)
else:
    st.info("No retriever comparison found. Run `python src/compare_retrievers.py` first.")

st.divider()

# ── Section 3: Raw Results ────────────────────────────────────────────────────
st.header("Raw Results")

for label, path, df in [
    ("Cosine", COSINE_CSV, cosine_df),
    ("Hybrid", HYBRID_CSV, hybrid_df),
]:
    with st.expander(f"{label} — {path.name}"):
        if df is not None:
            st.dataframe(df, use_container_width=True)
        else:
            st.info(f"{path} not found.")

st.caption("Evaluated using RAGAS | Mistral 7B judge | MiniLM embeddings")
