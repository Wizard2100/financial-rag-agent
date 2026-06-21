"""
retrieval_core.py

Pure retrieval logic for the financial RAG agent, deliberately kept free of
streamlit/genai imports so it can be:
  - unit tested in isolation (tests/test_retrieval_core.py)
  - driven by the offline retrieval evaluator (eval/eval_retrieval.py)
  - imported by app.py and wrapped with @st.cache_resource there

This is the same retrieval logic that used to live inline in app.py, just
pulled into its own module per the "split the monolith" refactor.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import faiss
import numpy as np

# ---------------------------------------------------------------------------
# Company detection - generalizes to any company present in the data, not
# just the three currently loaded. Add an explicit entry here when you know
# a company has common abbreviations/aliases; everything else falls back to
# matching the company's own lowercased name, which is usually good enough.
# ---------------------------------------------------------------------------

COMPANY_ALIASES = {
    "NVIDIA": ["nvidia", "nvda"],
    "Microsoft": ["microsoft", "msft"],
    "Reliance": ["reliance", "ril", "jio"],
    "Tesla": ["tesla", "tsla"],
}

COMPARISON_WORDS = [
    "compare", "comparison", "vs", "versus", "against", "between",
    "difference", "which company", "who has", "rank", "better than",
    "higher", "lower", "outperform",
]


@dataclass
class RetrievalEngine:
    """Loads the FAISS index + chunk metadata once and answers retrieval
    queries against them. Embedding model is injected rather than loaded
    here, so tests can swap in a fake encoder with no network access."""

    faiss_path: str
    chunks_path: str
    vectors_path: str
    embedding_model: object  # anything exposing .encode(list[str]) -> np.ndarray

    index: faiss.Index = field(init=False, repr=False)
    chunks: list = field(init=False, repr=False)
    vectors: np.ndarray = field(init=False, repr=False)
    positions_by_company: dict = field(init=False, repr=False)
    vectors_by_company: dict = field(init=False, repr=False)
    companies: list = field(init=False, repr=False)

    def __post_init__(self):
        self.index = faiss.read_index(self.faiss_path)

        with open(self.chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        self.vectors = np.load(self.vectors_path)

        positions_by_company: dict[str, list[int]] = {}
        for i, chunk in enumerate(self.chunks):
            company = chunk.get("company", "Unknown") if isinstance(chunk, dict) else "Unknown"
            positions_by_company.setdefault(company, []).append(i)

        self.positions_by_company = positions_by_company
        self.vectors_by_company = {
            company: self.vectors[positions] for company, positions in positions_by_company.items()
        }
        self.companies = sorted(positions_by_company.keys())

    # ------------------------------------------------------------------
    # query understanding
    # ------------------------------------------------------------------

    def detect_companies(self, query: str) -> list[str]:
        q = query.lower()
        return [
            c for c in self.companies
            if any(alias in q for alias in COMPANY_ALIASES.get(c, [c.lower()]))
        ]

    def is_comparison_query(self, query: str) -> bool:
        q = query.lower()
        return any(w in q for w in COMPARISON_WORDS)

    def target_companies(self, query: str) -> list[str]:
        mentioned = self.detect_companies(query)
        if mentioned:
            return mentioned
        if self.is_comparison_query(query):
            return self.companies
        return []

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------

    def retrieve_for_company(self, query_vector: np.ndarray, company: str, k: int) -> list[dict]:
        company_vectors = self.vectors_by_company[company]
        dists = np.linalg.norm(company_vectors - query_vector, axis=1)
        top_k = np.argsort(dists)[:k]
        return [self.chunks[self.positions_by_company[company][i]] for i in top_k]

    def retrieve_context(self, query: str, k_total: int = 30, k_global: int = 20) -> list[dict]:
        query_vector = self.embedding_model.encode([query]).astype("float32")
        targets = self.target_companies(query)

        if targets:
            k = max(8, k_total // len(targets))
            retrieved = []
            for company in targets:
                retrieved += self.retrieve_for_company(query_vector, company, k)
            return retrieved

        # no company named and not a comparison -> fall back to the global index
        D, I = self.index.search(query_vector, k_global)
        return [self.chunks[int(i)] for i in I[0] if 0 <= int(i) < len(self.chunks)]
