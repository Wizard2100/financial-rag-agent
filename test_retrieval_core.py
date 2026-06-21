import os

import numpy as np
import pytest

from retrieval_core import RetrievalEngine

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")


class FakeEncoder:
    """Deterministic stand-in for SentenceTransformer.encode - lets these
    tests run with zero network access. It is NOT semantically meaningful,
    so these tests only check pipeline correctness (right shapes, right
    companies, no crashes), not retrieval *quality*. For a real quality
    measurement, see eval/eval_retrieval.py, which uses the actual model."""

    def encode(self, texts):
        rng = np.random.default_rng(42)
        return rng.standard_normal((len(texts), 384)).astype("float32")


@pytest.fixture(scope="module")
def engine():
    return RetrievalEngine(
        faiss_path=os.path.join(DATA_DIR, "financialIndex.faiss"),
        chunks_path=os.path.join(DATA_DIR, "companyChunks.pkl"),
        vectors_path=os.path.join(DATA_DIR, "financialVectors.npy"),
        embedding_model=FakeEncoder(),
    )


def test_companies_loaded(engine):
    assert engine.companies == ["Microsoft", "NVIDIA", "Reliance"]


def test_detect_companies_single(engine):
    assert engine.detect_companies("What is NVIDIA's revenue?") == ["NVIDIA"]


def test_detect_companies_alias(engine):
    assert engine.detect_companies("how is msft doing") == ["Microsoft"]


def test_detect_companies_multiple(engine):
    found = engine.detect_companies("compare reliance and nvidia")
    assert set(found) == {"Reliance", "NVIDIA"}


def test_is_comparison_query(engine):
    assert engine.is_comparison_query("nvidia vs microsoft") is True
    assert engine.is_comparison_query("what is nvidia's revenue") is False


def test_target_companies_falls_back_to_all_on_bare_comparison(engine):
    # comparison language with no company named -> compare everything
    assert engine.target_companies("which company performed better") == engine.companies


def test_target_companies_empty_when_no_signal(engine):
    assert engine.target_companies("what are the main risks") == []


def test_retrieve_for_company_only_returns_that_companys_chunks(engine):
    query_vector = FakeEncoder().encode(["irrelevant"])[0]
    results = engine.retrieve_for_company(query_vector, "NVIDIA", k=8)
    assert len(results) == 8
    assert all(r.get("company") == "NVIDIA" for r in results)


def test_retrieve_context_single_company_query_stays_in_company(engine):
    results = engine.retrieve_context("What are NVIDIA's growth drivers?")
    assert all(r.get("company") == "NVIDIA" for r in results)


def test_retrieve_context_comparison_query_spans_companies(engine):
    results = engine.retrieve_context("compare revenue across all companies")
    companies_present = {r.get("company") for r in results}
    assert len(companies_present) > 1


def test_retrieve_context_global_fallback_still_returns_something(engine):
    results = engine.retrieve_context("what are the main risks mentioned in these filings")
    assert len(results) > 0
