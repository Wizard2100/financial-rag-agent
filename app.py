import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import json
import ast
import operator
import time
import os
import hashlib

from sentence_transformers import SentenceTransformer

import financials as fin
import market_tools as mkt
import export_utils as exp
import bull_bear as bb
from retrieval_core import RetrievalEngine

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Financial Research Agent",
    page_icon="📈",
    layout="wide"
)

# =====================================
# GEMINI (optional - the app must keep working without it)
# =====================================
# If GEMINI_API_KEY isn't set, or the genai package import fails, the app
# falls back to Verified-Only mode automatically instead of crashing on
# startup. This is what makes the "Mode" toggle in the sidebar meaningful
# rather than cosmetic.

GEMINI_AVAILABLE = False
client = None

try:
    from google import genai
    from google.genai import types
    from google.genai import errors

    if "GEMINI_API_KEY" in st.secrets:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

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
            "Switching to Verified-Only mode for this question instead.".format(wait_text)
        )
    elif isinstance(e, errors.APIError):
        st.error(f"Gemini API error ({e.status}): {e.message}")
    else:
        st.error(f"Error: {str(e)}")


# =====================================
# DEMO CACHE
# =====================================
# Disk-backed cache, separate from st.cache_data's in-memory one. Once a
# question has been answered successfully, the real response is saved to a
# file under demo_cache/. Every later run - including a cold container or
# the live API being down - serves that exact question instantly with zero
# API calls.

DEMO_CACHE_DIR = "demo_cache"
os.makedirs(DEMO_CACHE_DIR, exist_ok=True)


def _cache_path(prefix, query):
    key = hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
    return os.path.join(DEMO_CACHE_DIR, f"{prefix}_{key}.json")


def load_demo_cache(prefix, query):
    path = _cache_path(prefix, query)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_demo_cache(prefix, query, payload):
    path = _cache_path(prefix, query)
    try:
        with open(path, "w") as f:
            json.dump(payload, f, default=str)
    except Exception:
        pass


# =====================================
# LOAD RETRIEVAL ENGINE (cached so reruns don't reload everything from disk)
# =====================================

@st.cache_resource
def load_engine():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return RetrievalEngine(
        faiss_path="financialIndex.faiss",
        chunks_path="companyChunks.pkl",
        vectors_path="financialVectors.npy",
        embedding_model=embedding_model,
    )


engine = load_engine()
companies = engine.companies

TICKERS = {
    "NVIDIA": "NVDA",
    "Microsoft": "MSFT",
    "Reliance": "RELIANCE.NS",
}

# A few extra peers for the comps table only - they're not in the RAG index,
# they're purely for market-multiple comparison in the Ratio & DCF Lab.
PEER_TICKERS = {
    "AMD": "AMD",
    "Alphabet": "GOOGL",
    "TCS": "TCS.NS",
}

# =====================================
# AGENT TOOLS (only registered if Gemini is available)
# =====================================

def tool_search_annual_report(company, query):
    if company not in companies:
        return {"error": f"Unknown company '{company}'. Valid options: {companies}"}
    query_vector = engine.embedding_model.encode([query]).astype("float32")
    found = engine.retrieve_for_company(query_vector, company, k=8)
    return {"results": [c.get("text", "") for c in found]}


@st.cache_data(ttl=300)
def tool_get_live_market_data(company):
    ticker_symbol = TICKERS.get(company)
    if not ticker_symbol:
        return {"error": f"No ticker mapped for '{company}'. Valid options: {list(TICKERS.keys())}"}
    return mkt.get_live_snapshot(ticker_symbol)


SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
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
    "calculate": tool_calculate,
}

if GEMINI_AVAILABLE:
    AGENT_TOOLS = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="search_annual_report",
            description="Search one company's FY25 annual report for qualitative or quantitative information.",
            parameters={
                "type": "object",
                "properties": {
                    "company": {"type": "string", "enum": companies},
                    "query": {"type": "string", "description": "what to look for in the report"},
                },
                "required": ["company", "query"],
            },
        ),
        types.FunctionDeclaration(
            name="get_live_market_data",
            description="Get current live stock price, market cap and valuation multiples for a company.",
            parameters={
                "type": "object",
                "properties": {"company": {"type": "string", "enum": companies}},
                "required": ["company"],
            },
        ),
        types.FunctionDeclaration(
            name="calculate",
            description="Evaluate a simple arithmetic expression. Use this instead of doing math yourself.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        ),
    ])


@st.cache_data(ttl=3600, show_spinner=False)
def run_agent(query, max_steps=6):
    cached = load_demo_cache("agent", query)
    if cached:
        return cached["answer"], cached["trace"], True

    system_text = (
        f"You are an equity research agent covering {', '.join(companies)}. "
        "You have three tools: search_annual_report (FY25 report content), "
        "get_live_market_data (live price/market cap/valuation multiples), and calculate (arithmetic). "
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
            config=types.GenerateContentConfig(tools=[AGENT_TOOLS], temperature=0.2),
        )
        candidate = response.candidates[0]
        contents.append(candidate.content)
        function_calls = [p.function_call for p in candidate.content.parts if p.function_call]

        if not function_calls:
            answer = response.text or ""
            save_demo_cache("agent", query, {"answer": answer, "trace": trace})
            return answer, trace, False

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

    return "Hit the tool-call limit without a final answer - try a narrower question.", trace, False


@st.cache_data(ttl=3600, show_spinner=False)
def analyze_query_llm(query):
    """The Gemini-backed structured-JSON analysis path - unchanged in spirit
    from the previous version of this app, just retrieving through
    `engine` instead of inline FAISS calls."""
    cached = load_demo_cache("direct", query)
    if cached:
        return (
            cached["data"], cached["raw_text"], cached["retrieved_chunks"],
            set(cached["companies_used"]), cached["context"], True,
        )

    retrieved_chunks = engine.retrieve_context(query)
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
        context_parts.append(f"[Source {i}] Company: {company}\n{text}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a senior equity research analyst.

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
- comparison_table: only include a row backed by an actual number in the context. Use the same
  metric name across companies. Keep the unit exactly as reported in the source.
- segment_breakdown: only fill in if the question is about how revenue breaks into parts.
- risks: only include items the context actually mentions as risks.
- If a number isn't directly supported by the context, leave it out rather than estimating it.
"""

    response = generate_with_fallback(
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.3),
    )

    raw = response.text.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw.strip()).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None

    if data is not None:
        save_demo_cache("direct", query, {
            "data": data, "raw_text": response.text, "retrieved_chunks": retrieved_chunks,
            "companies_used": sorted(companies_used), "context": context,
        })

    return data, response.text, retrieved_chunks, companies_used, context, False


def analyze_query_verified(query):
    """The zero-API path: retrieval still runs (so 'sources' stay honest),
    but the written answer comes entirely from financials.py's template
    narrative generator instead of an LLM call."""
    retrieved_chunks = engine.retrieve_context(query)
    companies_used = {c.get("company", "Unknown") for c in retrieved_chunks if isinstance(c, dict)}

    targets = engine.target_companies(query) or list(companies_used) or fin.companies()
    targets = [t for t in targets if t in fin.companies()]

    if engine.is_comparison_query(query) and len(targets) > 1:
        metric = "Revenue"
        for candidate in ["Net Income", "Operating Income", "Revenue Growth %"]:
            if candidate.lower() in query.lower():
                metric = candidate
                break
        summary = fin.generate_comparison_narrative(metric)
        table = [
            {"metric": metric, "company": c, "value": fin.get_year(c).get(metric), "unit": "USD Billion", "period": fin.LATEST_YEAR}
            for c in targets if fin.get_year(c) and fin.get_year(c).get(metric) is not None
        ]
    else:
        summary = " ".join(fin.generate_narrative(c) for c in targets) if targets else \
            "No company could be confidently matched to this question - try naming one directly."
        table = []

    data = {
        "summary": summary,
        "key_findings": [],
        "comparison_table": table,
        "segment_breakdown": [],
        "risks": [],
    }
    context = "\n\n".join(c.get("text", "") for c in retrieved_chunks if isinstance(c, dict))
    return data, summary, retrieved_chunks, companies_used, context, False


# =====================================
# SIDEBAR
# =====================================

with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Analysis mode",
        ["🔒 Verified-Only (no API, can't fail)", "🤖 Live AI (Gemini)"],
        index=0 if not GEMINI_AVAILABLE else 1,
        disabled=not GEMINI_AVAILABLE,
        help="Verified-Only uses only hand-checked annual-report figures and a template "
             "narrative generator - zero LLM calls. Live AI uses Gemini for open-ended analysis.",
    )
    verified_only = mode.startswith("🔒") or not GEMINI_AVAILABLE

    if not GEMINI_AVAILABLE:
        st.caption("No Gemini API key found - running in Verified-Only mode.")

    st.divider()
    st.caption(
        "Verified-Only numbers come from hand-checked annual-report text. "
        "Live market data comes from Yahoo Finance via yfinance. "
        "Neither path ever asks an LLM to extract a number."
    )

# =====================================
# HEADER
# =====================================

st.title("📈 Financial Research Agent")
st.markdown(f"Analyze {', '.join(companies)} FY25 annual reports using Retrieval-Augmented Generation (RAG).")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Companies", len(companies))
with col2:
    st.metric("Knowledge Chunks", len(engine.chunks))
with col3:
    st.metric("Mode", "Verified-Only" if verified_only else "Gemini 2.5 Flash")

st.divider()

tab0, tab1, tab_ratio, tab_bull, tab2 = st.tabs([
    "🔒 Verified Snapshot", "💬 Direct Analysis", "📐 Ratio & DCF Lab", "🐂🐻 Bull vs Bear", "🤖 Agent Mode"
])

# =====================================
# TAB 0 - VERIFIED SNAPSHOT (zero API calls, can never fail or hit a quota)
# =====================================

with tab0:
    st.markdown(
        "Headline FY25 figures pulled directly from each company's actual annual report text "
        "and hand-verified - no LLM extraction, no live API calls."
    )

    metric_choice = st.radio(
        "Metric", ["Revenue", "Operating Income", "Net Income", "Revenue Growth %"],
        horizontal=True, key="snapshot_metric",
    )

    bar_companies, bar_values = [], []
    for c in companies:
        data = fin.get_year(c)
        v = data.get(metric_choice) if data else None
        if v is not None:
            bar_companies.append(c)
            bar_values.append(v)

    unit_label = "%" if metric_choice == "Revenue Growth %" else "$ Billion"
    bar_df = pd.DataFrame({"Company": bar_companies, "Value": bar_values})
    bar_fig = px.bar(bar_df, x="Company", y="Value", color="Company", text_auto=".3s",
                      title=f"{metric_choice} ({unit_label}) - FY25")
    bar_fig.update_layout(showlegend=False)
    st.plotly_chart(bar_fig, width="stretch")

    st.subheader("📊 Ratio Dashboard")
    st.caption("Margins computed live from the verified figures above - plain arithmetic, no model involved.")
    ratio_df = fin.ratio_dashboard()
    st.dataframe(ratio_df.style.format("{:,.1f}", na_rep="—", subset=ratio_df.columns[1:]), width="stretch")

    st.subheader("🕸️ Relative Profile")
    st.caption("Each axis is normalized against the strongest company on that metric (=100).")
    radar_metrics = ["Revenue", "Net Income", "Revenue Growth %"]
    radar_fig = go.Figure()
    maxes = {
        m: max((fin.get_year(c).get(m) for c in companies if fin.get_year(c) and fin.get_year(c).get(m) is not None), default=1)
        for m in radar_metrics
    }
    for c in companies:
        values = []
        for m in radar_metrics:
            data = fin.get_year(c)
            raw = data.get(m) if data else None
            values.append((raw / maxes[m] * 100) if raw is not None and maxes[m] else 0)
        values.append(values[0])
        radar_fig.add_trace(go.Scatterpolar(r=values, theta=radar_metrics + [radar_metrics[0]], fill="toself", name=c))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
    st.plotly_chart(radar_fig, width="stretch")

    st.subheader("📄 Per-Company Detail")
    detail_cols = st.columns(len(companies))
    for col, c in zip(detail_cols, companies):
        data = fin.get_year(c)
        if not data:
            continue
        with col:
            st.markdown(f"**{c}**")
            st.caption(f"FY ended {data['fiscal_year_end']}")
            for label, val in data.get("extra", {}).items():
                st.metric(label, f"{val:,.1f}" if isinstance(val, float) else f"{val:,}")
            with st.expander("Source excerpt"):
                st.text(data["quote"])
                st.caption(data["source"])

    st.subheader("⬇️ Export this snapshot")
    excel_bytes = exp.export_dataframes_to_excel({"Ratio Dashboard": ratio_df, "Snapshot": bar_df})
    pdf_bytes = exp.export_report_to_pdf("Verified FY25 Snapshot", [
        {"heading": "Ratio Dashboard", "text": None, "table": ratio_df},
        *[{"heading": c, "text": fin.generate_narrative(c), "table": None} for c in companies],
    ])
    dl1, dl2 = st.columns(2)
    dl1.download_button("Download Excel", excel_bytes, "verified_snapshot.xlsx",
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    dl2.download_button("Download PDF report", pdf_bytes, "verified_snapshot.pdf", "application/pdf")

# =====================================
# TAB 1 - DIRECT ANALYSIS (chat-style, mode-aware)
# =====================================

with tab1:
    st.markdown(
        "Ask a question and get a structured analysis with tables and charts. "
        "Follow-up questions keep the conversation in context."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    examples = [
        "Compare revenue and net income across all companies",
        "Compare R&D spending across all three companies",
        "What is NVIDIA's revenue breakdown by segment?",
        "What are the key risk factors for Reliance?",
    ]
    example_cols = st.columns(4)
    for col, example in zip(example_cols, examples):
        if col.button(example, width="stretch", key=f"ex_{example}"):
            st.session_state.pending_query = example

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    typed_query = st.chat_input("Ask a financial question")
    query = st.session_state.pop("pending_query", None) or typed_query

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        try:
            with st.spinner("Analyzing reports..."):
                if verified_only:
                    data, raw_text, retrieved_chunks, companies_used, context, from_cache = analyze_query_verified(query)
                else:
                    try:
                        data, raw_text, retrieved_chunks, companies_used, context, from_cache = analyze_query_llm(query)
                    except Exception as e:
                        render_api_error(e)
                        data, raw_text, retrieved_chunks, companies_used, context, from_cache = analyze_query_verified(query)

            with st.chat_message("assistant"):
                if not retrieved_chunks:
                    st.error("No relevant information found.")
                else:
                    if from_cache:
                        st.caption("📌 Pre-warmed demo answer - served instantly, no live API call made.")

                    if data is None:
                        st.warning("Couldn't parse a structured response, showing the raw answer instead.")
                        st.write(raw_text)
                    else:
                        st.info(data.get("summary", ""))

                        for finding in data.get("key_findings") or []:
                            st.markdown(f"- {finding}")

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
                                pivot = df.pivot_table(index="metric_label", columns="company", values="value", aggfunc="first")
                                st.dataframe(pivot.style.format("{:,.2f}", na_rep="—"), width="stretch")

                                st.subheader("📊 Visual Comparison")
                                n_panels = df["metric_label"].nunique()
                                fig = px.bar(df, x="company", y="value", color="company",
                                             facet_col="metric_label", facet_col_wrap=3, text_auto=".2s")
                                fig.update_yaxes(matches=None)
                                fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                                fig.update_layout(showlegend=False, height=320 * ((n_panels - 1) // 3 + 1))
                                st.plotly_chart(fig, width="stretch")

                                st.download_button(
                                    "⬇️ Download this comparison (Excel)",
                                    exp.export_dataframes_to_excel({"Comparison": df}),
                                    "comparison.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                )

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
                                    pie_col.plotly_chart(pie, width="stretch")

                        risks = data.get("risks") or []
                        if risks:
                            st.subheader("⚠️ Risks Mentioned")
                            for risk in risks:
                                st.warning(risk)

                    with st.expander("📚 Sources & retrieved chunks"):
                        st.write(" • ".join(sorted(companies_used)))
                        for i, chunk in enumerate(retrieved_chunks[:10], start=1):
                            company = chunk.get("company", "Unknown") if isinstance(chunk, dict) else "Unknown"
                            text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                            st.markdown(f"**Source {i} — {company}**")
                            st.text(text[:500])

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": (data.get("summary") if data else raw_text) or "",
                    })

        except Exception as e:
            render_api_error(e)

# =====================================
# TAB RATIO - RATIO & DCF LAB (live market data, deterministic math, no LLM)
# =====================================

with tab_ratio:
    st.markdown(
        "Live comps, ratios, and a two-stage DCF - all pulled from Yahoo Finance via `yfinance` "
        "and computed with plain arithmetic. No LLM is involved anywhere in this tab."
    )

    st.subheader("📋 Comps Table")
    comp_choice = st.multiselect("Companies to include", list(TICKERS) + list(PEER_TICKERS),
                                  default=list(TICKERS))
    selected_tickers = {name: {**TICKERS, **PEER_TICKERS}[name] for name in comp_choice}

    if selected_tickers:
        with st.spinner("Pulling live data from Yahoo Finance..."):
            comps_df = mkt.build_comps_table(selected_tickers)
        st.dataframe(comps_df, width="stretch")
        st.download_button(
            "⬇️ Download comps table (Excel)",
            exp.export_dataframes_to_excel({"Comps": comps_df}),
            "comps_table.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.divider()
    st.subheader("💰 DCF Calculator")
    st.caption(
        "Educational valuation tool with assumptions you control - this is not investment advice. "
        "Intrinsic value is only as good as the WACC/growth assumptions you choose."
    )

    dcf_company = st.selectbox("Company", list(TICKERS), key="dcf_company")
    c1, c2, c3 = st.columns(3)
    wacc = c1.slider("WACC", 0.04, 0.20, 0.10, 0.005, format="%.1f%%")
    terminal_growth = c2.slider("Terminal growth", 0.0, 0.06, 0.03, 0.0025, format="%.2f%%")
    fcf_growth = c3.slider("FCF growth (years 1-5)", -0.10, 0.40, 0.12, 0.01, format="%.0f%%")

    if st.button("Run DCF", type="primary"):
        with st.spinner("Pulling free cash flow data and discounting..."):
            result = mkt.run_dcf(TICKERS[dcf_company], wacc, terminal_growth, fcf_growth)

        if "error" in result:
            st.error(result["error"])
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Intrinsic value / share", f"${result['intrinsic_value_per_share']:,.2f}")
            m2.metric("Current price", f"${result['current_price']:,.2f}" if result.get("current_price") else "—")
            if result.get("current_price"):
                upside = (result["intrinsic_value_per_share"] / result["current_price"] - 1) * 100
                m3.metric("Implied upside/downside", f"{upside:+.1f}%")

            fcf_df = pd.DataFrame(result["projected_fcf"])
            fig = px.bar(fcf_df, x="year", y="fcf", title="Projected free cash flow by year", text_auto=".2s")
            st.plotly_chart(fig, width="stretch")

            st.subheader("🌡️ Sensitivity: intrinsic value per share")
            wacc_range = [wacc - 0.02, wacc - 0.01, wacc, wacc + 0.01, wacc + 0.02]
            growth_range = [terminal_growth - 0.01, terminal_growth - 0.005, terminal_growth, terminal_growth + 0.005, terminal_growth + 0.01]
            wacc_range = [w for w in wacc_range if w > 0]
            growth_range = [g for g in growth_range if g >= 0]
            grid = mkt.dcf_sensitivity_grid(TICKERS[dcf_company], wacc_range, growth_range, fcf_growth)

            if not grid.empty:
                heat_fig = px.imshow(grid, text_auto=".0f", aspect="auto",
                                      labels=dict(x="Terminal growth", y="WACC", color="$/share"),
                                      color_continuous_scale="RdYlGn")
                st.plotly_chart(heat_fig, width="stretch")

    st.divider()
    st.subheader("📈 Price History & Correlation")
    hist_choice = st.multiselect("Companies for price history", list(TICKERS), default=list(TICKERS), key="hist_choice")
    period = st.select_slider("Period", options=["1mo", "3mo", "6mo", "1y", "2y"], value="1y")

    if hist_choice:
        with st.spinner("Pulling price history..."):
            histories = mkt.get_price_history({n: TICKERS[n] for n in hist_choice}, period=period)

        for name, hist in histories.items():
            candle = go.Figure(data=[go.Candlestick(
                x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
            )])
            candle.update_layout(title=f"{name} ({TICKERS[name]})", height=350, xaxis_rangeslider_visible=False)
            st.plotly_chart(candle, width="stretch")

        if len(histories) > 1:
            corr = mkt.compute_correlation_matrix(histories)
            if not corr.empty:
                st.subheader("🔗 Return Correlation Matrix")
                corr_fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                st.plotly_chart(corr_fig, width="stretch")

        st.divider()
        st.subheader("💼 Portfolio Backtest")
        st.caption(
            "Hypothetical only — shows how a fixed-weight basket of these stocks would have moved "
            "over the period above. Not a recommendation."
        )

        weight_cols = st.columns(len(hist_choice))
        raw_weights = {}
        for col, name in zip(weight_cols, hist_choice):
            raw_weights[name] = col.slider(f"{name} weight", 0, 100, 100 // len(hist_choice), key=f"weight_{name}")

        total_weight = sum(raw_weights.values())
        initial_investment = st.number_input("Initial investment ($)", min_value=100, value=10000, step=500)

        if total_weight > 0 and histories:
            normalized_weights = {n: w / total_weight for n, w in raw_weights.items()}

            portfolio_value = None
            for name, hist in histories.items():
                if name not in normalized_weights or "Close" not in hist.columns:
                    continue
                shares_value = hist["Close"] / hist["Close"].iloc[0] * initial_investment * normalized_weights[name]
                portfolio_value = shares_value if portfolio_value is None else portfolio_value.add(shares_value, fill_value=0)

            if portfolio_value is not None and len(portfolio_value) > 1:
                total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
                p1, p2 = st.columns(2)
                p1.metric("Ending value", f"${portfolio_value.iloc[-1]:,.0f}")
                p2.metric("Total return", f"{total_return:+.1f}%")

                port_fig = go.Figure()
                port_fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value.values, fill="tozeroy",
                                               name="Portfolio value"))
                port_fig.update_layout(title="Portfolio value over time", yaxis_title="$")
                st.plotly_chart(port_fig, width="stretch")

# =====================================
# TAB BULL/BEAR - rule-based, zero-LLM analyst-style case generator
# =====================================

with tab_bull:
    st.markdown(
        "A deterministic bull/bear case for each company. Every bullet below is a fixed threshold check "
        "against a real verified or live number — nothing here is written by an LLM, so nothing here can "
        "hallucinate, and every claim can be traced back to a specific figure."
    )

    bull_company = st.selectbox("Company", list(TICKERS), key="bull_company")

    if st.button("Generate case", type="primary", key="bull_button"):
        with st.spinner("Checking thresholds against verified + live data..."):
            case = bb.build_case(bull_company, TICKERS[bull_company])

        if case.get("error"):
            st.warning(f"Live market data unavailable, case is based on verified figures only: {case['error']}")

        col_bull, col_bear = st.columns(2)
        with col_bull:
            st.markdown("### 🐂 Bull Case")
            for point in case["bull"]:
                st.success(point)
        with col_bear:
            st.markdown("### 🐻 Bear Case")
            for point in case["bear"]:
                st.error(point)

        report_pdf = exp.export_report_to_pdf(f"{bull_company} — Bull vs Bear Case", [
            {"heading": "Bull Case", "text": " ".join(case["bull"]), "table": None},
            {"heading": "Bear Case", "text": " ".join(case["bear"]), "table": None},
        ])
        st.download_button("⬇️ Download this case (PDF)", report_pdf, f"{bull_company}_bull_bear.pdf", "application/pdf")

# =====================================
# TAB 2 - AGENT MODE (tool use: report search + live market data + calculator)
# =====================================

with tab2:
    if not GEMINI_AVAILABLE:
        st.warning("Agent Mode needs a Gemini API key (GEMINI_API_KEY in st.secrets) - it isn't available in Verified-Only deployments.")
    else:
        st.markdown(
            "Ask something that might need live market data, multiple lookups, or a calculation - "
            "the agent decides which tools to call and in what order."
        )

        agent_examples = [
            "Is NVIDIA fairly valued compared to Microsoft based on their P/E ratios?",
            "What is Reliance's current market cap vs its FY25 revenue?",
            "How far is NVIDIA's live stock price from its 52-week high?",
        ]
        agent_cols = st.columns(3)
        for col, example in zip(agent_cols, agent_examples):
            if col.button(example, width="stretch", key=f"agent_{example}"):
                st.session_state.agent_query_text = example

        agent_query = st.text_input("Ask the agent", key="agent_query_text")

        if agent_query:
            try:
                with st.spinner("Agent is working..."):
                    answer, trace, from_cache = run_agent(agent_query)

                if from_cache:
                    st.caption("📌 Pre-warmed demo answer - served instantly, no live API call made.")

                if trace:
                    st.subheader("🔧 Agent Trace")
                    for t in trace:
                        with st.expander(f"Step {t['step']}: called `{t['tool']}`"):
                            st.json(t["args"])
                            st.json(t["result"])

                market_calls = [t for t in trace if t["tool"] == "get_live_market_data" and "error" not in t["result"]]
                if market_calls:
                    st.subheader("📡 Live Market Snapshot")
                    snap_cols = st.columns(len(market_calls))
                    for snap_col, t in zip(snap_cols, market_calls):
                        r = t["result"]
                        company_name = t["args"].get("company", r.get("ticker", "?"))
                        price = r.get("current_price")
                        currency = r.get("currency") or ""
                        pe = r.get("trailing_pe")
                        snap_col.metric(
                            f"{company_name} ({r.get('ticker', '')})",
                            f"{currency} {price:,.2f}" if price else "—",
                            f"P/E {pe:.1f}" if pe else None,
                        )

                st.subheader("📋 Answer")
                st.markdown(answer)

            except Exception as e:
                render_api_error(e)
