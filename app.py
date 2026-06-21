import streamlit as st
import faiss
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import re
import json

from collections import Counter
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

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

# =====================================
# QUERY UNDERSTANDING
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
# RETRIEVAL
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

# =====================================
# QUICK EXAMPLES
# =====================================

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

# =====================================
# MAIN PIPELINE
# =====================================

if query:

    try:

        with st.spinner("Analyzing reports..."):

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
                st.error("No relevant information found.")
                st.stop()

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

            response = client.models.generate_content(
                model="gemini-2.5-flash",
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

        # =====================================
        # ANSWER
        # =====================================

        if data is None:
            st.subheader("📊 Financial Analysis")
            st.warning("Couldn't parse a structured response this time, showing the raw answer instead.")
            st.write(response.text)

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

        # =====================================
        # SOURCES
        # =====================================

        st.subheader("📚 Companies Referenced")
        st.write(" • ".join(sorted(companies_used)))

        with st.expander("View retrieved source chunks"):
            for i, chunk in enumerate(retrieved_chunks, start=1):
                company = chunk.get("company", "Unknown") if isinstance(chunk, dict) else "Unknown"
                text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                st.markdown(f"**Source {i} — {company}**")
                st.text(text[:500])

        # =====================================
        # THEMES CHART
        # =====================================

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
        st.error(f"Error: {str(e)}")
