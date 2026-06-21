# Financial Research Agent

RAG + agentic tool-use over NVIDIA, Microsoft, and Reliance FY25 annual reports, with a
deterministic, zero-API analysis mode alongside the LLM-backed one.

## What's deterministic vs. LLM-generated (read this before a demo or interview)

| Feature | Source | Can it fail/hallucinate? |
|---|---|---|
| Verified Snapshot tab (`financials.py`) | Hand-checked annual-report figures | No - plain arithmetic on fixed numbers |
| Ratio & DCF Lab (`market_tools.py`) | Live Yahoo Finance data via `yfinance`, no key needed | Numbers no; DCF *output* only as good as the WACC/growth you choose |
| Bull vs Bear (`bull_bear.py`) | Threshold rules over the two sources above | No - fixed rules, no model |
| Direct Analysis tab, Verified-Only mode | Retrieval (`retrieval_core.py`) + template narrative (`financials.py`) | No |
| Direct Analysis tab, Live AI mode | Retrieval + Gemini, asked to only state numbers present in context | Yes, in principle - mitigated by grounding in retrieved context, not by the model's own knowledge |
| Agent Mode | Gemini with tool-calling (report search / live data / calculator) | Yes - same caveat as above |

The sidebar mode toggle switches Direct Analysis between the deterministic and LLM paths.
If no `GEMINI_API_KEY` is configured, the app detects that at startup and runs in
Verified-Only mode automatically - Agent Mode is unavailable in that case, everything else
still works.

## Project layout

```
app.py              - Streamlit UI, wires everything together
retrieval_core.py    - FAISS + chunk retrieval logic, no streamlit/genai import (testable)
financials.py        - verified annual-report figures + deterministic ratio engine + template narrative
market_tools.py       - live Yahoo Finance data, comps table, DCF calculator, price history
bull_bear.py          - rule-based bull/bear case generator
export_utils.py       - Excel/PDF export helpers
companyChunks.pkl, financialIndex.faiss, financialVectors.npy  - existing RAG data, unchanged
tests/                - pytest unit tests + headless Streamlit smoke tests
eval/                 - retrieval quality evaluation (real hit-rate, not assumed)
```

## Running tests

```
pip install -r requirements.txt
pytest tests/test_financials.py tests/test_retrieval_core.py -v
python tests/smoke_test_app.py            # headless render check (mocks the embedding model + yfinance)
python tests/smoke_test_interactions.py   # exercises buttons/chat input end-to-end
```

## Running the retrieval evaluation

```
python eval/eval_retrieval.py
```

Needs internet on first run (downloads `all-MiniLM-L6-v2` from Hugging Face, same as the
app already does). Measures how often the *global* semantic fallback - the retrieval path
used when a query doesn't name a company - actually returns chunks from the right company,
against a 15-question hand-labeled set in `eval/eval_set.json`. Results are written to
`eval/eval_results.json`.

## Known gap (next step, not done here)

`financials.py`'s FY24/FY23 fields are intentionally left as `None` - filling them in from
the real FY24/FY23 annual reports is what unlocks multi-year trend lines everywhere in the
app. Don't estimate these numbers; the ratio/narrative code already handles missing years
correctly by omitting them rather than guessing.
