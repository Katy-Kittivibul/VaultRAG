import chromadb
from chromadb.config import Settings
chromadb.Client(Settings(anonymized_telemetry=False))

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall, ResponseRelevancy

from src.agent import ask, build_chain, build_hybrid_chain

load_dotenv()

TESTSET_PATH = Path("results/testset.json")
RESULTS_DIR = Path("results")


def _load_testset() -> list[dict]:
    return json.loads(TESTSET_PATH.read_text(encoding="utf-8"))


def _collect_rag_outputs(chain, testset: list[dict], cache=None) -> list[dict]:
    rows = []
    total = len(testset)
    for i, item in enumerate(testset, start=1):
        print(f"  [{i}/{total}] {item['question']}")
        result = ask(chain, item["question"], cache)
        rows.append({
            "user_input": item["question"],
            "response": result["answer"],
            "retrieved_contexts": [doc.page_content for doc in result["sources"]],
            "reference": item["ground_truth"],
        })
    return rows


def _reground_references(rows: list[dict], llm: Ollama) -> list[dict]:
    print("Regrounding references from retrieved contexts...")
    regrounded = []
    for row in rows:
        contexts = "\n\n".join(row["retrieved_contexts"])
        prompt = (
            f"Question: {row['user_input']}\n\n"
            f"Retrieved contexts:\n{contexts}\n\n"
            "Write a factual answer to the question using ONLY information explicitly "
            "present in the retrieved contexts above. Do not mention any tool, library, "
            "framework, or concept that does not appear in the contexts."
        )
        answer = llm.invoke(prompt)
        regrounded.append({**row, "reference": answer})
        print(f"  regrounded: {row['user_input'][:60]}")
    return regrounded


class _CombinedResult:
    """Wraps a merged DataFrame so helper functions work unchanged."""
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df.copy()

    @property
    def scores(self):
        non_metric = {"user_input", "response", "retrieved_contexts", "reference"}
        cols = [c for c in self._df.columns if c not in non_metric]
        return self._df[cols].to_dict(orient="records")


def _metric_cols(result) -> list[str]:
    non_meta = {"user_input", "response", "retrieved_contexts", "reference"}
    return [c for c in result.to_pandas().columns if c not in non_meta]


def _run_ragas(rows: list[dict], reground: bool = False) -> _CombinedResult:
    judge_llm = LangchainLLMWrapper(Ollama(model="mistral", temperature=0, timeout=120))
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    metrics = [
        LLMContextRecall(llm=judge_llm),
        Faithfulness(llm=judge_llm),
        FactualCorrectness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings),
    ]

    if reground:
        rows = _reground_references(rows, Ollama(model="mistral"))

    eval_dataset = EvaluationDataset.from_list(rows)

    first = rows[0]
    print("--- Debug: first dataset item ---")
    print(f"  user_input:         {first['user_input']}")
    print(f"  response:           {first['response'][:80]}")
    print(f"  retrieved_contexts: {len(first['retrieved_contexts'])} chunk(s)")
    print(f"  reference:          {first['reference'][:80]}")
    print(f"Dataset size: {len(eval_dataset)}")

    merged_df = None
    for i, metric in enumerate(metrics):
        print(f"  [{i + 1}/{len(metrics)}] Evaluating {type(metric).__name__}...")
        try:
            result = evaluate(
                dataset=eval_dataset,
                metrics=[metric],
                llm=judge_llm,
                embeddings=judge_embeddings,
                batch_size=1,
            )
        except Exception as exc:
            print(f"evaluate() raised {type(exc).__name__}: {exc}")
            raise

        metric_df = result.to_pandas()
        if merged_df is None:
            merged_df = metric_df
        else:
            non_meta = {"user_input", "response", "retrieved_contexts", "reference"}
            new_cols = [c for c in metric_df.columns if c not in non_meta]
            merged_df = merged_df.join(metric_df[new_cols])

    return _CombinedResult(merged_df)


def _save_results(result: _CombinedResult, rows: list[dict], csv_path: Path) -> list[dict]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = result.to_pandas()
    df.to_csv(csv_path, index=False)

    scored = []
    for i, row in enumerate(rows):
        scores = {}
        for col in _metric_cols(result):
            val = df.iloc[i][col]
            scores[col] = (
                None if (val is None or (isinstance(val, float) and math.isnan(val)))
                else float(val)
            )
        scored.append({**row, "scores": scores})
    return scored


def _print_comparison(cosine_result: _CombinedResult, hybrid_result: _CombinedResult) -> None:
    cosine_df = cosine_result.to_pandas()
    hybrid_df = hybrid_result.to_pandas()
    metrics = _metric_cols(cosine_result)

    label_w, col_w = 32, 10
    sep = "-" * (label_w + col_w * 2 + 6)
    print("\nEvaluation Comparison")
    print(sep)
    print(f"  {'Metric':<{label_w}} {'Cosine':>{col_w}} {'Hybrid':>{col_w}}")
    print(sep)
    for col in metrics:
        c_mean = cosine_df[col].mean()
        h_mean = hybrid_df[col].mean() if col in hybrid_df.columns else float("nan")
        print(f"  {col:<{label_w}} {c_mean:>{col_w}.4f} {h_mean:>{col_w}.4f}")
    print(sep)
    print()


def _save_comparison(cosine_scored: list[dict], hybrid_scored: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    merged = []
    for c_row, h_row in zip(cosine_scored, hybrid_scored):
        merged.append({
            "question": c_row["user_input"],
            "cosine": {"answer": c_row["response"], "scores": c_row["scores"]},
            "hybrid": {"answer": h_row["response"], "scores": h_row["scores"]},
        })
    (RESULTS_DIR / "eval_comparison.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reground", action="store_true")
    args = parser.parse_args()

    testset = _load_testset()[:3]
    print(f"Loaded testset: {len(testset)} pairs")

    print("\n--- Cosine chain ---")
    cosine_chain, cosine_cache = build_chain()
    print("Collecting RAG outputs...")
    cosine_rows = _collect_rag_outputs(cosine_chain, testset, cosine_cache)
    print("Running RAGAS evaluation...")
    cosine_result = _run_ragas(cosine_rows, reground=args.reground)
    cosine_scored = _save_results(cosine_result, cosine_rows, RESULTS_DIR / "eval_cosine.csv")

    print("\n--- Hybrid chain ---")
    hybrid_chain, hybrid_cache = build_hybrid_chain()
    print("Collecting RAG outputs...")
    hybrid_rows = _collect_rag_outputs(hybrid_chain, testset, hybrid_cache)
    print("Running RAGAS evaluation...")
    hybrid_result = _run_ragas(hybrid_rows, reground=args.reground)
    hybrid_scored = _save_results(hybrid_result, hybrid_rows, RESULTS_DIR / "eval_hybrid.csv")

    _print_comparison(cosine_result, hybrid_result)
    _save_comparison(cosine_scored, hybrid_scored)

    print("Saved → results/eval_cosine.csv")
    print("Saved → results/eval_hybrid.csv")
    print("Saved → results/eval_comparison.json")


if __name__ == "__main__":
    main()
