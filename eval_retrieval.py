"""
eval/eval_retrieval.py

Measures retrieval quality instead of just assuming it. Specifically: for
each question in eval_set.json (none of which name a company by name), runs
it through the *global* fallback path - the part of retrieve_context that
has no company hint and has to rely purely on semantic similarity - and
checks whether the retrieved chunks actually come from the expected company.

This is the metric that matters most: the per-company path is close to
trivially correct once a company is named in the query, but the global path
is where retrieval quality is actually being tested.

Run from the repo root:
    python eval/eval_retrieval.py

Requires internet access on first run, since SentenceTransformer downloads
the all-MiniLM-L6-v2 weights from Hugging Face the first time it's used -
same as the live app already does. If you're running this somewhere with no
internet (a locked-down CI runner, for instance), pre-download the model or
point HF_HOME at a directory that already has it cached.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval_core import RetrievalEngine  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), "eval_set.json")
TOP_K = 10


def load_engine() -> RetrievalEngine:
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return RetrievalEngine(
        faiss_path=os.path.join(REPO_ROOT, "financialIndex.faiss"),
        chunks_path=os.path.join(REPO_ROOT, "companyChunks.pkl"),
        vectors_path=os.path.join(REPO_ROOT, "financialVectors.npy"),
        embedding_model=embedding_model,
    )


def global_retrieve(engine: RetrievalEngine, question: str, k: int = TOP_K) -> list[dict]:
    """Bypasses target_companies entirely - always goes through the FAISS
    global index, regardless of whether the question happens to name a
    company. This is what actually tests embedding quality."""
    query_vector = engine.embedding_model.encode([question]).astype("float32")
    D, I = engine.index.search(query_vector, k)
    return [engine.chunks[int(i)] for i in I[0] if 0 <= int(i) < len(engine.chunks)]


def evaluate(engine: RetrievalEngine, eval_set: list[dict]) -> dict:
    per_question = []

    for item in eval_set:
        question = item["question"]
        expected = item["expected_company"]

        retrieved = global_retrieve(engine, question)
        companies_retrieved = [c.get("company", "Unknown") for c in retrieved]
        company_counts = Counter(companies_retrieved)

        top_company, top_count = company_counts.most_common(1)[0]
        purity = top_count / len(companies_retrieved) if companies_retrieved else 0.0
        correct_majority = top_company == expected

        # also useful: did the expected company show up at all in top-k,
        # even if it wasn't the majority
        expected_present_rate = company_counts.get(expected, 0) / len(companies_retrieved) if companies_retrieved else 0.0

        per_question.append({
            "question": question,
            "expected_company": expected,
            "majority_company": top_company,
            "correct_majority": correct_majority,
            "majority_purity": round(purity, 2),
            "expected_company_share": round(expected_present_rate, 2),
            "company_breakdown": dict(company_counts),
        })

    accuracy = sum(1 for r in per_question if r["correct_majority"]) / len(per_question)
    avg_expected_share = sum(r["expected_company_share"] for r in per_question) / len(per_question)

    return {
        "top_k": TOP_K,
        "num_questions": len(per_question),
        "majority_company_accuracy": round(accuracy, 3),
        "avg_expected_company_share_of_topk": round(avg_expected_share, 3),
        "per_question": per_question,
    }


def main():
    with open(EVAL_SET_PATH) as f:
        eval_set = json.load(f)

    print(f"Loading retrieval engine and embedding model ({len(eval_set)} eval questions)...")
    engine = load_engine()

    results = evaluate(engine, eval_set)

    print()
    print(f"Top-k:                                {results['top_k']}")
    print(f"Questions evaluated:                  {results['num_questions']}")
    print(f"Majority-company accuracy:            {results['majority_company_accuracy']:.1%}")
    print(f"Avg. expected-company share of top-k: {results['avg_expected_company_share_of_topk']:.1%}")
    print()
    print("Per-question detail:")
    for r in results["per_question"]:
        mark = "PASS" if r["correct_majority"] else "FAIL"
        print(f"  [{mark}] expected={r['expected_company']:<10} got={r['majority_company']:<10} "
              f"purity={r['majority_purity']:.0%}  | {r['question'][:70]}")

    out_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
