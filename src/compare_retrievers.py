import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.agent import ask, build_chain, build_hybrid_chain

load_dotenv()

TESTSET_PATH = Path("results/testset.json")
RESULTS_DIR = Path("results")


def main() -> None:
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(testset)} questions\n")

    print("Building cosine chain...")
    cosine_chain, cosine_cache = build_chain()
    print("Building hybrid chain...")
    hybrid_chain, hybrid_cache = build_hybrid_chain()

    rows = []
    for i, item in enumerate(testset, start=1):
        question = item["question"]
        print(f"\n[{i}/{len(testset)}] {question}")

        cosine_result = ask(cosine_chain, question, cosine_cache)
        hybrid_result = ask(hybrid_chain, question, hybrid_cache)

        print(f"  [cosine] {cosine_result['answer'][:120]}")
        print(f"  [hybrid] {hybrid_result['answer'][:120]}")

        rows.append({
            "question": question,
            "cosine_answer": cosine_result["answer"],
            "hybrid_answer": hybrid_result["answer"],
            "cosine_sources": [
                doc.metadata.get("source", "") for doc in cosine_result["sources"]
            ],
            "hybrid_sources": [
                doc.metadata.get("source", "") for doc in hybrid_result["sources"]
            ],
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retriever_comparison.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _print_summary(rows)
    print("Saved → results/retriever_comparison.json")


def _print_summary(rows: list[dict]) -> None:
    sep = "=" * 70
    thin = "-" * 70
    print(f"\n{sep}")
    print(f"  Retriever Comparison ({len(rows)} questions)")
    print(sep)
    for i, row in enumerate(rows, start=1):
        q = row["question"][:60] + ("…" if len(row["question"]) > 60 else "")
        c = row["cosine_answer"][:80] + ("…" if len(row["cosine_answer"]) > 80 else "")
        h = row["hybrid_answer"][:80] + ("…" if len(row["hybrid_answer"]) > 80 else "")
        print(f"[{i}] {q}")
        print(f"     cosine: {c}")
        print(f"     hybrid: {h}")
        if i < len(rows):
            print(thin)
    print(sep)


if __name__ == "__main__":
    main()
