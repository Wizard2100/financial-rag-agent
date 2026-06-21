import streamlit as st
import faiss
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import yfinance as yf
import re
import json
import ast
import operator
import time

from collections import Counter
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from google.genai import errors

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Financial Research Agent",
    page_icon="📈",
    layout="wide"
)

# =====================================
# GEMINI
# =====================================

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# free-tier daily quotas are tracked per model, so falling back to a second
# model on 429 gets you a separate bucket instead of just failing
FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]


def extract_retry_seconds(api_error):
    try:
        for detail in api_error.details["error"]["details"]:
            if detail.get("@type", "").endswith("RetryInfo"):
                return float(detail["retryDelay"].rstrip("s"))
    except Exception:
        pass
    return None


def generate_with_fallback(contents, config):
    last_error = None

    for i, model_name in enumerate(FALLBACK_MODELS):
        try:
            return client.models.generate_content(model=model_name, contents=contents, config=config)

        except errors.ClientError as e:
            last_error = e
            if e.code != 429:
                raise

            # first model, short cooldown -> worth one retry before giving up on it
            wait = extract_retry_seconds(e)
            if i == 0 and wait and wait <= 30:
                time.sleep(wait + 1)
                try:
                    return client.models.generate_content(model=model_name, contents=contents, config=config)
                except errors.ClientError as e2:
                    last_error = e2

    raise last_error


def render_api_error(e):
    if isinstance(e, errors.ClientError) and e.code == 429:
        wait = extract_retry_seconds(e)
        wait_text = f" Retry in about {int(wait)}s." if wait else ""
        st.error(
            "Hit Gemini's free-tier daily request limit for this project.{} "
            "Free-tier quotas reset at midnight Pacific time (~12:30 PM IST). "
            "For uninterrupted demos, enable billing in Google AI Studio - Gemini "
            "2.5 Flash is inexpensive even under heavy testing.".format(wait_text)
        )
    elif isinstance(e, errors.APIError):
        st.error(f"Gemini API error ({e.status}): {e.message}")
    else:
        st.error(f"Error: {str(e)}")

# =====================================
# LOAD MODELS (cached so reruns don't reload everything from disk)
# =====================================

@st.cache_resource
def load_resources():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("financialIndex.faiss")

    with open("companyChunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    vectors = np.load("financialVectors.npy")

    # pre-split vectors by company so comparison questions don't get
    # drowned out by whichever company happens to have the most chunks
    positions_by_company = {}
    for i, chunk in enumerate(chunks):
        company = chunk.get("company", "Unknown") if isinstance(chunk, dict) else "Unknown"
        positions_by_company.setdefault(company, []).append(i)

    vectors_by_company = {
        company: vectors[positions] for company, positions in positions_by_company.items()
    }

    return embedding_model, index, chunks, vectors, positions_by_company, vectors_by_company


embedding_model, index, chunks, vectors, positions_by_company, vectors_by_company = load_resources()
companies = sorted(positions_by_company.keys())

TICKERS = {
    "NVIDIA": "NVDA",
    "Microsoft": "MSFT",
    "Reliance": "RELIANCE.NS"
}

# =====================================
# QUERY UNDERSTANDING (used by Direct Analysis tab)
# =====================================

COMPANY_ALIASES = {
    "NVIDIA": ["nvidia", "nvda"],
    "Microsoft": ["microsoft", "msft"],
    "Reliance": ["reliance", "ril", "jio"]
}

COMPARISON_WORDS = [
    "compare", "comparison", "vs", "versus", "against", "between",
    "difference", "which company", "who has", "rank", "better than",
    "higher", "lower", "outperform"
]


def detect_companies(query):
    q = query.lower()
    found = [c for c in companies if any(a in q for a in COMPANY_ALIASES.get(c, [c.lower()]))]
    return found


def is_comparison_query(query):
    q = query.lower()
    return any(w in q for w in COMPARISON_WORDS)


def target_companies(query):
    mentioned = detect_companies(query)
    if mentioned:
        return mentioned
    if is_comparison_query(query):
        return companies
    return []

# =====================================
# RETRIEVAL (shared by both tabs)
# =====================================

def retrieve_for_company(query_vector, company, k):
    company_vectors = vectors_by_company[company]
    dists = np.linalg.norm(company_vectors - query_vector, axis=1)
    top_k = np.argsort(dists)[:k]
    return [chunks[positions_by_company[company][i]] for i in top_k]


def retrieve_context(query, query_vector):
    targets = target_companies(query)

    if targets:
        # split a fixed budget of ~30 chunks evenly across target companies
        # so e.g. Reliance's 1218 chunks can't crowd out Microsoft's 241
        k = max(8, 30 // len(targets))
        retrieved = []
        for company in targets:
            retrieved += retrieve_for_company(query_vector, company, k)
        return retrieved

    # no company named and not a comparison -> fall back to the global index
    D, I = index.search(query_vector.astype("float32"), 20)
    return [chunks[int(i)] for i in I[0] if 0 <= int(i) < len(chunks)]

# =====================================
# AGENT TOOLS
# =====================================

def tool_search_annual_report(company, query):
    if company not in companies:
        return {"error": f"Unknown company '{company}'. Valid options: {companies}"}

    query_vector = embedding_model.encode([query]).astype("float32")
    found = retrieve_for_company(query_vector, company, k=8)
    return {"results": [c.get("text", "") for c in found]}


@st.cache_data(ttl=300)
def tool_get_live_market_data(company):
    ticker_symbol = TICKERS.get(company)
    if not ticker_symbol:
        return {"error": f"No ticker mapped for '{company}'. Valid options: {list(TICKERS.keys())}"}

    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        return {
            "ticker": ticker_symbol,
            "currency": info.get("currency"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow")
        }

    except Exception:
        # Yahoo Finance occasionally rate-limits or changes its response shape -
        # fall back to the lighter fast_info endpoint rather than failing outright
        try:
            fast = yf.Ticker(ticker_symbol).fast_info
            return {
                "ticker": ticker_symbol,
                "currency": fast.get("currency"),
                "current_price": fast.get("last_price"),
                "market_cap": fast.get("market_cap"),
                "fifty_two_week_high": fast.get("year_high"),
                "fifty_two_week_low": fast.get("year_low")
            }
        except Exception as e:
            return {"error": f"Live market data unavailable right now: {str(e)}"}


SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPS:
        return SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPS:
        return SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("expression contains something other than numbers and + - * / ()")


def tool_calculate(expression):
    try:
        tree = ast.parse(expression, mode="eval")
        return {"result": _safe_eval(tree.body)}
    except Exception as e:
        return {"error": str(e)}


TOOL_REGISTRY = {
    "search_annual_report": tool_search_annual_report,
    "get_live_market_data": tool_get_live_market_data,
    "calculate": tool_calculate
}

AGENT_TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="search_annual_report",
        description=(
            "Search one company's FY25 annual report for qualitative or quantitative "
            "information - strategy, risks, segments, or specific financial figures."
        ),
        parameters={
            "type": "object",
            "properties": {
                "company": {"type": "string", "enum": companies},
                "query": {"type": "string", "description": "what to look for in the report"}
            },
            "required": ["company", "query"]
        }
    ),
    types.FunctionDeclaration(
        name="get_live_market_data",
        description="Get current live stock price, market cap and valuation multiples (P/E) for a company.",
        parameters={
            "type": "object",
            "properties": {
                "company": {"type": "string", "enum": companies}
            },
            "required": ["company"]
        }
    ),
    types.FunctionDeclaration(
        name="calculate",
        description=(
            "Evaluate a simple arithmetic expression, e.g. to compute a ratio like "
            "'130.5 / 72.9' or a percentage change. Use this instead of doing math yourself."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }
    )
])


@st.cache_data(ttl=3600, show_spinner=False)
def run_agent(query, max_steps=6):
    system_text = (
        "You are an equity research agent covering NVIDIA, Microsoft and Reliance. "
        "You have three tools: search_annual_report (FY25 report content), "
        "get_live_market_data (live price/market cap/P-E), and calculate (arithmetic). "
        "Call whatever tools you need, in whatever order makes sense, before answering. "
        "Never do arithmetic yourself - always call calculate for it. "
        "Write the final answer in markdown with bold figures and bullet points, and say "
        "which tool each fact came from."
    )

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=f"{system_text}\n\nQuestion: {query}")])]
    trace = []

    for step in range(max_steps):

        response = generate_with_fallback(
            contents=contents,
            config=types.GenerateContentConfig(tools=[AGENT_TOOLS], temperature=0.2)
        )

        candidate = response.candidates[0]
        contents.append(candidate.content)

        function_calls = [p.function_call for p in candidate.content.parts if p.function_call]

        if not function_calls:
            return response.text or "", trace

        response_parts = []

        for fc in function_calls:
            args = dict(fc.args or {})
            handler = TOOL_REGISTRY.get(fc.name)
            result = handler(**args) if handler else {"error": f"unknown tool '{fc.name}'"}

            trace.append({"step": step + 1, "tool": fc.name, "args": args, "result": result})

            response_parts.append(types.Part(function_response=types.FunctionResponse(
                id=fc.id, name=fc.name, response=result
            )))

        contents.append(types.Content(role="user", parts=response_parts))

    return "Hit the tool-call limit without a final answer - try a narrower question.", trace


@st.cache_data(ttl=3600, show_spinner=False)
def analyze_query(query):
    query_vector = embedding_model.encode([query]).astype("float32")

    retrieved_chunks = retrieve_context(query, query_vector)

    companies_used = set()
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        if isinstance(chunk, dict):
            company = chunk.get("company", "Unknown")
            text = chunk.get("text", "")
        else:
            company = "Unknown"
            text = str(chunk)

        companies_used.add(company)
        context_parts.append(f"[Source {i} | {company}]\n{text}")

    if not context_parts:
        return None, "", [], set(), ""

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a senior equity research analyst. You have been given excerpts from FY25 annual
reports of NVIDIA, Microsoft and Reliance Industries.

Answer the question using ONLY the context below. Do not invent numbers that aren't present
in the context.

CONTEXT:
{context}

QUESTION:
{query}

Respond with ONLY raw JSON, no markdown fences, in exactly this structure:

{{
  "summary": "2-4 sentence executive summary answering the question",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "comparison_table": [
    {{"metric": "Revenue", "company": "NVIDIA", "value": 130.5, "unit": "USD Billion", "period": "FY25"}}
  ],
  "segment_breakdown": [
    {{"company": "NVIDIA", "segment": "Data Center", "value": 115.2, "unit": "USD Billion"}}
  ],
  "risks": ["risk 1", "risk 2"]
}}

Rules:
- comparison_table: only include a row if it's backed by an actual number in the context.
  Use the SAME metric name across companies when they mean the same thing (always "Revenue",
  never mix it with "Total Revenue" or "Net Sales") so values can be grouped on a chart later.
  Keep the unit exactly as reported in the source - NVIDIA and Microsoft report in USD,
  Reliance reports in INR Crore. Never convert currencies, just label "unit" correctly.
- segment_breakdown: only fill this in if the question is actually about how a company's
  revenue or business breaks down into parts (segments, geographies, product lines).
  Otherwise leave it as an empty list.
- risks: only include items the context actually mentions as risks or concerns. Otherwise
  leave it as an empty list.
- If a number isn't directly supported by the context, leave it out rather than estimating it.
"""

    response = generate_with_fallback(
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3
        )
    )

    raw = response.text.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw.strip()).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None

    return data, response.text, retrieved_chunks, companies_used, context


# =====================================
# HEADER
# =====================================

st.title("📈 Financial Research Agent")
st.markdown(
    "Analyze NVIDIA, Microsoft and Reliance FY25 annual reports using Retrieval-Augmented Generation (RAG)."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Companies", len(companies))

with col2:
    st.metric("Knowledge Chunks", len(chunks))

with col3:
    st.metric("Model", "Gemini 2.5 Flash")

st.divider()

tab1, tab2 = st.tabs(["📊 Direct Analysis", "🤖 Agent Mode"])

# =====================================
# TAB 1 - DIRECT ANALYSIS (RAG -> structured JSON -> table/charts)
# =====================================

with tab1:

    st.markdown("**Try a comparison**")

    examples = [
        "Compare revenue and net income across NVIDIA, Microsoft and Reliance",
        "Compare R&D spending across all three companies",
        "What is NVIDIA's revenue breakdown by segment?",
        "What are the key risk factors for Reliance?"
    ]

    example_cols = st.columns(4)
    for col, example in zip(example_cols, examples):
        if col.button(example, width='stretch'):
            st.session_state.query_text = example


    query = st.text_input("Ask a financial question", key="query_text")

    if query:

        try:

            with st.spinner("Analyzing reports..."):
                data, raw_response_text, retrieved_chunks, companies_used, context = analyze_query(query)

            if not retrieved_chunks:
                st.error("No relevant information found.")
                st.stop()

            # ---------- answer ----------

            if data is None:
                st.subheader("📊 Financial Analysis")
                st.warning("Couldn't parse a structured response this time, showing the raw answer instead.")
                st.write(raw_response_text)

            else:
                st.subheader("📊 Financial Analysis")
                st.info(data.get("summary", ""))

                findings = data.get("key_findings") or []
                if findings:
                    st.markdown("**Key Findings**")
                    for finding in findings:
                        st.markdown(f"- {finding}")

                # ---------- comparison table + chart ----------

                table = data.get("comparison_table") or []

                if table:
                    df = pd.DataFrame(table)

                    for col in ["metric", "company", "value", "unit"]:
                        if col not in df.columns:
                            df[col] = None

                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df = df.dropna(subset=["value", "metric", "company"])

                    if not df.empty:
                        df["unit"] = df["unit"].fillna("")
                        df["metric_label"] = df["metric"] + df["unit"].apply(lambda u: f" ({u})" if u else "")

                        st.subheader("📋 Comparison Table")
                        pivot = df.pivot_table(
                            index="metric_label", columns="company", values="value", aggfunc="first"
                        )
                        st.dataframe(pivot.style.format("{:,.2f}", na_rep="—"), width='stretch')

                        st.subheader("📊 Visual Comparison")

                        n_panels = df["metric_label"].nunique()
                        fig = px.bar(
                            df, x="company", y="value", color="company",
                            facet_col="metric_label", facet_col_wrap=3,
                            text_auto=".2s"
                        )
                        # each metric keeps its own y-axis - NVIDIA/Microsoft (USD) and
                        # Reliance (INR Crore) live on completely different scales
                        fig.update_yaxes(matches=None)
                        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                        fig.update_layout(
                            showlegend=False,
                            height=320 * ((n_panels - 1) // 3 + 1)
                        )
                        st.plotly_chart(fig, width='stretch')

                        if df["unit"].nunique() > 1:
                            st.caption(
                                "Each panel uses the currency/unit as reported - NVIDIA and Microsoft "
                                "report in USD, Reliance reports in INR Crore, so bars are only "
                                "directly comparable within the same panel."
                            )

                # ---------- segment breakdown pies ----------

                segments = data.get("segment_breakdown") or []

                if segments:
                    seg_df = pd.DataFrame(segments)

                    for col in ["segment", "company", "value"]:
                        if col not in seg_df.columns:
                            seg_df[col] = None

                    seg_df["value"] = pd.to_numeric(seg_df["value"], errors="coerce")
                    seg_df = seg_df.dropna(subset=["value", "segment", "company"])

                    if not seg_df.empty:
                        st.subheader("🥧 Segment Breakdown")
                        seg_companies = seg_df["company"].unique()
                        pie_cols = st.columns(len(seg_companies))

                        for pie_col, company in zip(pie_cols, seg_companies):
                            g = seg_df[seg_df["company"] == company]
                            pie = px.pie(g, names="segment", values="value", title=company, hole=0.35)
                            pie.update_traces(textinfo="percent+label", showlegend=False)
                            pie_col.plotly_chart(pie, width='stretch')

                # ---------- risks ----------

                risks = data.get("risks") or []
                if risks:
                    st.subheader("⚠️ Risks Mentioned")
                    for risk in risks:
                        st.warning(risk)

            # ---------- sources ----------

            st.subheader("📚 Companies Referenced")
            st.write(" • ".join(sorted(companies_used)))

            with st.expander("View retrieved source chunks"):
                for i, chunk in enumerate(retrieved_chunks, start=1):
                    company = chunk.get("company", "Unknown") if isinstance(chunk, dict) else "Unknown"
                    text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                    st.markdown(f"**Source {i} — {company}**")
                    st.text(text[:500])

            # ---------- themes chart ----------

            words = re.findall(r"\b[a-zA-Z]{4,}\b", context.lower())

            stopwords = {
                "company", "revenue", "income", "would", "could", "their", "there",
                "which", "these", "those", "about", "after", "before", "other",
                "using", "from", "have", "been", "were", "with"
            }

            words = [w for w in words if w not in stopwords]
            top_words = Counter(words).most_common(10)

            if top_words:
                with st.expander("📈 Key Terms in Retrieved Context"):
                    theme_df = pd.DataFrame(top_words, columns=["Keyword", "Frequency"])
                    st.bar_chart(theme_df.set_index("Keyword"))

        except Exception as e:
            render_api_error(e)

# =====================================
# TAB 2 - AGENT MODE (tool use: report search + live market data + calculator)
# =====================================

with tab2:

    st.markdown(
        "Ask something that might need live market data, multiple lookups, or a calculation - "
        "the agent decides which tools to call and in what order."
    )

    agent_examples = [
        "Is NVIDIA fairly valued compared to Microsoft based on their P/E ratios?",
        "What is Reliance's current market cap vs its FY25 revenue?",
        "How far is NVIDIA's live stock price from its 52-week high?"
    ]

    agent_cols = st.columns(3)
    for col, example in zip(agent_cols, agent_examples):
        if col.button(example, width='stretch', key=f"agent_{example}"):
            st.session_state.agent_query_text = example

    agent_query = st.text_input("Ask the agent", key="agent_query_text")

    if agent_query:

        try:

            with st.spinner("Agent is working..."):
                answer, trace = run_agent(agent_query)

            if trace:
                st.subheader("🔧 Agent Trace")
                for t in trace:
                    with st.expander(f"Step {t['step']}: called `{t['tool']}`"):
                        st.json(t["args"])
                        st.json(t["result"])

            market_calls = [
                t for t in trace
                if t["tool"] == "get_live_market_data" and "error" not in t["result"]
            ]

            if market_calls:
                st.subheader("📡 Live Market Snapshot")
                snap_cols = st.columns(len(market_calls))

                for snap_col, t in zip(snap_cols, market_calls):
                    r = t["result"]
                    company = t["args"].get("company", r.get("ticker", "?"))
                    price = r.get("current_price")
                    currency = r.get("currency") or ""
                    pe = r.get("trailing_pe")

                    snap_col.metric(
                        f"{company} ({r.get('ticker', '')})",
                        f"{currency} {price:,.2f}" if price else "—",
                        f"P/E {pe:.1f}" if pe else None
                    )

            st.subheader("📋 Answer")
            st.markdown(answer)

        except Exception as e:
            render_api_error(e)
