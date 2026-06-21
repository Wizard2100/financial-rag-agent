"""
Headless smoke test using streamlit.testing.v1.AppTest - boots the real
app.py (against the real FAISS index/chunks/vectors) with the embedding
model mocked out, since this sandbox has no internet access to Hugging
Face. Confirms the app renders without an unhandled exception and that
each tab's static content (the parts that don't need a button click)
comes up clean.

This does NOT replace running the app for real in an environment with
internet access (Streamlit Cloud, Colab, your own machine) - it's a fast
check for "does importing/rendering this file blow up", not a full
behavioral test.
"""
import os
import sys
import types as pytypes

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeEncoder:
    def encode(self, texts):
        rng = np.random.default_rng(7)
        return rng.standard_normal((len(texts), 384)).astype("float32")


fake_st_module = pytypes.ModuleType("sentence_transformers")
fake_st_module.SentenceTransformer = lambda *a, **k: FakeEncoder()
sys.modules["sentence_transformers"] = fake_st_module

FAKE_INFO = {
    "NVDA": {"currency": "USD", "currentPrice": 140.0, "marketCap": 3.4e12, "trailingPE": 45.0,
              "forwardPE": 32.0, "priceToBook": 40.0, "enterpriseValue": 3.39e12, "enterpriseToEbitda": 38.0,
              "enterpriseToRevenue": 25.0, "returnOnEquity": 0.95, "returnOnAssets": 0.55, "debtToEquity": 12.0,
              "currentRatio": 4.2, "profitMargins": 0.558, "operatingMargins": 0.624,
              "freeCashflow": 60_000_000_000, "totalDebt": 10_000_000_000, "totalCash": 40_000_000_000,
              "sharesOutstanding": 24_500_000_000, "fiftyTwoWeekHigh": 180.0, "fiftyTwoWeekLow": 80.0},
    "MSFT": {"currency": "USD", "currentPrice": 470.0, "marketCap": 3.5e12, "trailingPE": 35.0,
              "forwardPE": 30.0, "priceToBook": 12.0, "returnOnEquity": 0.35, "debtToEquity": 45.0,
              "profitMargins": 0.361, "operatingMargins": 0.456, "freeCashflow": 70_000_000_000,
              "totalDebt": 60_000_000_000, "totalCash": 80_000_000_000, "sharesOutstanding": 7_400_000_000,
              "fiftyTwoWeekHigh": 500.0, "fiftyTwoWeekLow": 380.0},
    "RELIANCE.NS": {"currency": "INR", "currentPrice": 1400.0, "marketCap": 18.9e12, "trailingPE": 24.0,
              "forwardPE": 22.0, "priceToBook": 2.1, "returnOnEquity": 0.09, "debtToEquity": 38.0,
              "profitMargins": 0.076, "freeCashflow": -50_000_000_000, "totalDebt": 3_000_000_000_000,
              "totalCash": 2_000_000_000_000, "sharesOutstanding": 6_766_000_000,
              "fiftyTwoWeekHigh": 1608.0, "fiftyTwoWeekLow": 1115.0},
}


class FakeTicker:
    def __init__(self, symbol):
        self.symbol = symbol

    @property
    def info(self):
        return FAKE_INFO.get(self.symbol, {})

    @property
    def fast_info(self):
        return {}

    def history(self, period="1y"):
        idx = pd.date_range(end=pd.Timestamp.today(), periods=60, freq="D")
        base = FAKE_INFO.get(self.symbol, {}).get("currentPrice", 100.0)
        rng = np.random.default_rng(abs(hash(self.symbol)) % (2**32))
        closes = base + rng.standard_normal(60).cumsum()
        return pd.DataFrame({
            "Open": closes, "High": closes * 1.01, "Low": closes * 0.99, "Close": closes,
            "Volume": rng.integers(1_000_000, 5_000_000, 60),
        }, index=idx)


fake_yf_module = pytypes.ModuleType("yfinance")
fake_yf_module.Ticker = FakeTicker
sys.modules["yfinance"] = fake_yf_module

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app.py")
at = AppTest.from_file(APP_PATH)
at.run(timeout=60)

print("Unhandled exceptions:", len(at.exception))
for e in at.exception:
    print(" -", e.value)

print("Number of tabs found:", len(at.tabs))
print("Number of markdown blocks:", len(at.markdown))
print("Number of dataframes rendered:", len(at.dataframe))
print("Number of plotly charts rendered:", len(at.get("plotly_chart")) if at.get("plotly_chart") else 0)

assert len(at.exception) == 0, "App raised an unhandled exception on initial render"
print("\nSMOKE TEST PASSED — app.py boots and renders all tabs with no unhandled exception.")
