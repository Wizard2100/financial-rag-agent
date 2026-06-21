import streamlit as st
import faiss
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import re
import json
import os
import hashlib
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from google import genai
from google.genai import types
from google.genai import errors

# =====================================
# PRE-WARMED DEMO CACHE (Dynamic populate)
# =====================================
DEMO_CACHE_DIR = "demo_cache"
os.makedirs(DEMO_CACHE_DIR, exist_ok=True)

PRE_WARMED_ANSWERS = {
    "compare revenue and net income across nvidia, microsoft and reliance": {
        "summary": "Microsoft leads in consolidated revenue at $281.7 Billion, but NVIDIA boasts the highest profitability relative to size, achieving a 55.8% net margin ($72.9 Billion Net Income) compared to Microsoft's 36.1% and Reliance's 7.6%.",
        "key_findings": [
            "Microsoft reports the largest absolute revenue of $281.7B, driven by its 15% growth in commercial cloud.",
            "NVIDIA displays hyper-growth of 114% YoY, hitting $130.5B, fueled by global generative AI infrastructure demand.",
            "Reliance Industries dominates Indian markets with $125.3B consolidated revenue (INR 10,71,174 Crore) but operates on lower margins (7.6%) due to heavy asset and infrastructure costs."
        ],
        "comparison_table": [
            {"metric": "Revenue", "company": "Microsoft", "value": 281.7, "unit": "USD Billion", "period": "FY25"},
            {"metric": "Revenue", "company": "NVIDIA", "value": 130.5, "unit": "USD Billion", "period": "FY25"},
            {"metric": "Revenue", "company": "Reliance", "value": 125.3, "unit": "USD Billion", "period": "FY25"},
            {"metric": "Net Income", "company": "Microsoft", "value": 101.8, "unit": "USD Billion", "period": "FY25"},
            {"metric": "Net Income", "company": "NVIDIA", "value": 72.88, "unit": "USD Billion", "period": "FY25"},
            {"metric": "Net Income", "company": "Reliance", "value": 9.5, "unit": "USD Billion", "period": "FY25"}
        ],
        "segment_breakdown": [],
        "risks": [
            "Heavy capital deployment dependency by hyperscalers affects NVIDIA's forward growth pipelines.",
            "Refining margins fluctuations and domestic retail competitive headwinds affect Reliance's traditional energy structures."
        ]
    },
    "compare the ai strategies of microsoft and nvidia": {
        "summary": "NVIDIA and Microsoft occupy complementary poles of the AI hardware-software continuum. NVIDIA dominates hardware compute accelerators (GPU chipsets & CUDA ecosystems), while Microsoft monetizes cognitive applications (Office 365 Copilots & OpenAI services) built on top of Azure cloud infrastructure.",
        "key_findings": [
            "NVIDIA operates as the fundamental 'picks-and-shovels' provider, generating 88% of its sales ($115.2B) from its GPU Data Center segment.",
            "Microsoft acts as the platform orchestrator, driving an Azure Cloud run rate of $75B+ (up 34% YoY) by embedding OpenAI services at scale.",
            "Microsoft represents one of NVIDIA's largest compute customers, while simultaneously designing internal Maia AI silicon to hedge dependency."
        ],
        "comparison_table": [
            {"metric": "AI/Data Center Revenue", "company": "NVIDIA", "value": 115.2, "unit": "USD Billion", "period": "FY25"},
            {"metric": "Azure/Cloud Revenue", "company": "Microsoft", "value": 75.0, "unit": "USD Billion", "period": "FY25"}
        ],
        "segment_breakdown": [
            {"company": "NVIDIA", "segment": "Data Center", "value": 115.2, "unit": "USD Billion"},
            {"company": "NVIDIA", "segment": "Gaming & Others", "value": 15.3, "unit": "USD Billion"},
            {"company": "Microsoft", "segment": "Intelligent Cloud", "value": 105.0, "unit": "USD Billion"},
            {"company": "Microsoft", "segment": "Productivity & More", "value": 176.7, "unit": "USD Billion"}
        ],
        "risks": [
            "Hardware supply bottlenecks and wafer yield issues on advanced TSMC packaging lines for Blackwell GPUs.",
            "Enormous capital expenditures in data centers ($50B+ annually for MSFT) must yield proportional subscription software growth."
        ]
    },
    "what are the key risk factors for reliance industries?": {
        "summary": "Reliance Industries' diversified conglomerate business structure exposes it to sector-specific risks across petrochemicals, retail, and digital services. Heavy capital expenditure commitments are primary factors affecting immediate liquidity margins.",
        "key_findings": [
            "Oil-to-Chemicals (O2C) segment remains highly cyclical, heavily exposed to global demand spreads and crude pricing volatility.",
            "Jio digital services require sustained, capital-intensive investments in spectrum allocations and telecom node hardware.",
            "Aggressive expansions in retail and clean energy have led to a net debt position of INR 1,16,281 Crore ($13.6B), increasing financial leverage."
        ],
        "comparison_table": [
            {"metric": "Net Debt", "company": "Reliance", "value": 13.6, "unit": "USD Billion", "period": "FY25"},
            {"metric": "Net Debt", "company": "NVIDIA", "value": -25.5, "unit": "USD Billion", "period": "FY25"},
            {"metric": "Net Debt", "company": "Microsoft", "value": -35.0, "unit": "USD Billion", "period": "FY25"}
        ],
        "segment_breakdown": [],
        "risks": [
            "Global petrochemical capacity oversupply softening margins.",
            "Aggressive e-commerce competitive battles in Indian retail from Amazon and local quick-commerce rivals."
        ]
    }
}

def populate_local_cache():
    for q, payload in PRE_WARMED_ANSWERS.items():
        key = hashlib.sha256(q.encode()).hexdigest()[:16]
        path = os.path.join(DEMO_CACHE_DIR, f"direct_{key}.json")
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(payload, f)

populate_local_cache()

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="ValuEdge: Global Investment Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# PREMIUM DARK UI CUSTOM STYLE (Bloomberg-Style)
# =====================================
st.markdown("""
<style>
    /* Main layout dark background */
    .stApp {
        background-color: #090d16;
        color: #e6edf3;
    }
    
    /* Elegant Title and Header Gradients */
    .main-title {
        font-size: 38px;
        font-weight: 800;
        background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .subtitle {
        font-size: 16px;
        color: #8b949e;
        margin-bottom: 25px;
    }
    
    /* Glowing custom cards */
    .bloomberg-card {
        background-color: #121824;
        border: 1px solid #212836;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    
    .card-header {
        font-size: 14px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    
    .card-value {
        font-size: 28px;
        font-weight: 700;
        color: #58a6ff;
    }
    
    /* Styling Streamlit sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #212836;
    }
    
    /* Ticker badge */
    .ticker-badge {
        display: inline-block;
        background-color: #1f293d;
        color: #58a6ff;
        padding: 3px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-weight: bold;
        font-size: 13px;
        border: 1px solid #303e5c;
    }
</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD PRE-LOADED VECTOR RESOURCES (Cached)
# =====================================
@st.cache_resource
def load_baseline_resources():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    try:
        index = faiss.read_index("financialIndex.faiss")
        with open("companyChunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        vectors = np.load("financialVectors.npy")
        
        positions_by_company = {}
        for i, chunk in enumerate(chunks):
            company = chunk.get("company", "Unknown") if isinstance(chunk, dict) else "Unknown"
            positions_by_company.setdefault(company, []).append(i)

        vectors_by_company = {
            company: vectors[positions] for company, positions in positions_by_company.items()
        }
        return embedding_model, index, chunks, vectors, positions_by_company, vectors_by_company
    except Exception:
        return embedding_model, None, [], None, {}, {}

embedding_model, index, chunks, vectors, positions_by_company, vectors_by_company = load_baseline_resources()
companies = sorted(positions_by_company.keys()) if positions_by_company else []

TICKERS = {
    "NVIDIA": "NVDA",
    "Microsoft": "MSFT",
    "Reliance": "RELIANCE.NS"
}

# =====================================
# VERIFIED Snapshot & du Pont Financials (FY25)
# =====================================
VERIFIED_FINANCIALS = {
    "NVIDIA": {
        "fiscal_year_end": "January 26, 2025",
        "source": "NVIDIA FY2025 Form 10-K",
        "Revenue": 130.5,
        "Operating Income": 81.45,
        "Net Income": 72.88,
        "Revenue Growth %": 114.0,
        "Assets": 115.0,
        "Equity": 85.0,
        "roe": 85.18,
        "net_margin": 55.85,
        "asset_turnover": 1.13,
        "leverage": 1.35,
        "fcf_margin": 45.0,
        "shares_outstanding": 24.5,
        "cash": 34.0,
        "debt": 8.5,
        "industry": "Semiconductors & AI Hardware",
        "quote": "Revenue more than doubled to $130.5 billion, up 114% year-over-year. Operating income rose 147% to $81.5 billion. Net income $72,880 million, up 145%."
    },
    "Microsoft": {
        "fiscal_year_end": "June 30, 2025",
        "source": "Microsoft FY2025 Annual Report",
        "Revenue": 281.7,
        "Operating Income": 128.5,
        "Net Income": 101.8,
        "Revenue Growth %": 15.0,
        "Assets": 510.0,
        "Equity": 290.0,
        "roe": 35.10,
        "net_margin": 36.14,
        "asset_turnover": 0.55,
        "leverage": 1.76,
        "fcf_margin": 28.0,
        "shares_outstanding": 7.43,
        "cash": 80.0,
        "debt": 45.0,
        "industry": "Cloud Computing & Enterprise Software",
        "quote": "Revenue was $281.7 billion, up 15 percent. Operating income grew 17 percent to $128.5 billion. Azure surpassed $75 billion in revenue for the first time, up 34 percent."
    },
    "Reliance": {
        "fiscal_year_end": "March 31, 2025",
        "source": "Reliance Industries Integrated Annual Report 2024-25",
        "Revenue": 125.3,
        "Operating Income": 14.5, # Adjusted Consolidated EBIT
        "Net Income": 9.5,
        "Revenue Growth %": 7.1,
        "Assets": 222.0,
        "Equity": 99.0,
        "roe": 9.56,
        "net_margin": 7.59,
        "asset_turnover": 0.56,
        "leverage": 2.24,
        "fcf_margin": 6.0,
        "shares_outstanding": 6.77,
        "cash": 4.7,
        "debt": 17.6,
        "industry": "Conglomerate (Energy, Retail, Telecom)",
        "quote": "Consolidated revenue increased by 7.1% to Rs 10,71,174 crore (US$125.3 billion). EBITDA grew 2.9% to Rs 1,83,422 crore (US$21.5 billion). PAT rose 2.9% to Rs 81,309 crore (US$9.5 billion)."
    }
}

# =====================================
# GLOBAL FINANCIAL FETCH ENGINE (Any Ticker Worldwide)
# =====================================
@st.cache_data(ttl=600)
def fetch_global_financials(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        name = info.get("longName") or info.get("shortName") or ticker_symbol
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        shares = info.get("sharesOutstanding")
        currency = info.get("currency", "$")
        
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow
        
        def get_row(df, keys, default=0.0):
            if df is None or df.empty:
                return default
            for k in keys:
                for idx in df.index:
                    if str(idx).strip().lower() == k.strip().lower():
                        val = df.loc[idx].iloc[0] if isinstance(df.loc[idx], pd.Series) else df.loc[idx]
                        if pd.notna(val):
                            return float(val)
            return default
            
        revenue = get_row(financials, ["Total Revenue", "Revenue", "Gross Sales"]) / 1e9
        ebit = get_row(financials, ["Operating Income", "EBIT", "Operating Income Or Loss"]) / 1e9
        net_income = get_row(financials, ["Net Income", "Net Income Common Stockholders"]) / 1e9
        
        assets = get_row(balance_sheet, ["Total Assets"]) / 1e9
        equity = get_row(balance_sheet, ["Stockholders Equity", "Total Stockholders Equity", "Total Equity"]) / 1e9
        cash = get_row(balance_sheet, ["Cash And Cash Equivalents", "Cash", "Cash Cash Equivalents And Short Term Investments"]) / 1e9
        debt = get_row(balance_sheet, ["Total Debt", "Long Term Debt"]) / 1e9
        
        # Estimate FCF
        ocf = get_row(cashflow, ["Operating Cash Flow", "Cash Flow From Operating Activities", "Total Cash From Operating Activities"])
        capex = get_row(cashflow, ["Capital Expenditure", "Net CRF", "Capital Expenditures"])
        fcf = (ocf - abs(capex)) / 1e9 if ocf else (net_income * 0.8)
        
        # Net Margin & Leverage estimates
        net_margin = (net_income / revenue * 100) if revenue else 5.0
        asset_turnover = (revenue / assets) if assets else 0.5
        leverage = (assets / equity) if equity else 1.0
        roe = net_margin * asset_turnover * leverage
        
        return {
            "name": name,
            "price": price,
            "shares": shares / 1e9 if shares else 1.0,
            "currency": currency,
            "revenue": revenue if revenue > 0 else 1.0,
            "ebit": ebit if ebit > 0 else 0.2,
            "net_income": net_income if net_income > 0 else 0.15,
            "assets": assets if assets > 0 else 2.0,
            "equity": equity if equity > 0 else 1.0,
            "cash": cash if cash > 0 else 0.2,
            "debt": debt if debt > 0 else 0.1,
            "fcf_margin": (fcf / revenue * 100) if (revenue and fcf > 0) else 15.0,
            "roe": roe,
            "net_margin": net_margin,
            "asset_turnover": asset_turnover,
            "leverage": leverage
        }
    except Exception:
        return None

# =====================================
# SIDEBAR CONTROLS
# =====================================
st.sidebar.markdown("<h1 style='color:#e6edf3; font-size:20px; font-weight:800;'>🌐 Global Analyzer Settings</h1>", unsafe_allow_html=True)
global_ticker = st.sidebar.text_input("Analyze Any Stock Ticker", value="AAPL", help="Type any ticker symbol (e.g. NVDA, TSLA, AAPL, TCS.NS, BP.L)").upper()

with st.sidebar.spinner("Fetching global market indicators..."):
    global_data = fetch_global_financials(global_ticker)

if global_data:
    st.sidebar.success(f"Loaded: {global_data['name']}")
    st.sidebar.markdown(f"""
    <div style='background-color:#121824; border:1px solid #212836; border-radius:6px; padding:10px 15px; margin-bottom:10px;'>
        <span style='color:#8b949e; font-size:12px;'>Current Live Price</span><br/>
        <span style='font-size:22px; font-weight:700; color:#58a6ff;'>{global_data['currency']} {global_data['price']:,.2f}</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.warning("Could not download financials. Displaying offline estimates.")

st.sidebar.divider()
st.sidebar.markdown("<h2 style='color:#e6edf3; font-size:18px;'>🔑 LLM Configuration</h2>", unsafe_allow_html=True)
user_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Add key for custom RAG prompts.")
api_key = user_api_key or st.secrets.get("GEMINI_API_KEY", "")

if api_key:
    st.sidebar.success("Gemini API Key Loaded!")
else:
    st.sidebar.info("Demo Mode (Cache-only active). Input key for custom prompts.")

# Old cache definitions removed (moved to top of file)

# =====================================
# APP LOGIC HEADER
# =====================================
st.markdown("<h1 class='main-title'>ValuEdge: Global Investment Terminal & RAG Agent</h1>", unsafe_allow_html=True)

# Main Application Tabs
tab_val, tab_dup, tab_port, tab_self_rag, tab_rag = st.tabs([
    "📈 Global DCF Appraiser", 
    "🕸️ du Pont Profitability", 
    "📊 Dynamic Portfolio Backtest",
    "📁 Self-Serve PDF RAG",
    "🔒 Baseline Filed Reports"
])

# Get baseline active company data
active_company = global_ticker
active_data = global_data or {
    "name": "Fallback (Apple)",
    "price": 180.0,
    "shares": 15.4,
    "currency": "$",
    "revenue": 385.0,
    "ebit": 115.0,
    "net_income": 97.0,
    "assets": 350.0,
    "equity": 60.0,
    "fcf_margin": 26.0,
    "roe": 160.0,
    "net_margin": 25.1,
    "asset_turnover": 1.1,
    "leverage": 5.8,
    "cash": 30.0,
    "debt": 100.0
}

# =====================================
# TAB 1: GLOBAL DCF APPRAISER
# =====================================
with tab_val:
    st.markdown(f"### 📈 Interactive DCF Valuation: {active_data['name']}")
    st.caption("Projections calculated dynamically based on real-time corporate parameters.")
    
    col_input, col_chart = st.columns([1, 2])
    
    with col_input:
        st.markdown(f"**Baseline Financial Inputs for {active_company}:**")
        st.caption(f"Revenue: **${active_data['revenue']:.2f}B** | Shares: **{active_data['shares']:.2f}B** | Net Cash/Debt: **${active_data['cash'] - active_data['debt']:.2f}B**")
        
        dcf_rev_growth = st.slider("5-Year Revenue Growth CAGR (%)", 0.0, 100.0, 15.0, step=0.5, key="g_dcf_rg")
        dcf_fcf_margin = st.slider("Target FCF Margin (%)", 1.0, 60.0, active_data["fcf_margin"], step=0.5, key="g_dcf_fcfm")
        dcf_wacc = st.slider("Discount Rate / WACC (%)", 5.0, 20.0, 9.0, step=0.1, key="g_dcf_wacc")
        dcf_terminal = st.slider("Terminal Growth Rate (%)", 0.5, 6.0, 2.5, step=0.1, key="g_dcf_tg")
        
        if dcf_wacc <= dcf_terminal:
            st.error("WACC must be strictly greater than Terminal Growth Rate.")
            dcf_wacc = dcf_terminal + 0.5
            
    # Calculate DCF Projection
    rev_projection = []
    fcf_projection = []
    
    current_rev = active_data["revenue"]
    for yr in range(1, 6):
        current_rev = current_rev * (1 + (dcf_rev_growth / 100))
        current_fcf = current_rev * (dcf_fcf_margin / 100)
        rev_projection.append(current_rev)
        fcf_projection.append(current_fcf)
        
    discount_factors = [1 / ((1 + (dcf_wacc / 100)) ** yr) for yr in range(1, 6)]
    pv_cashflows = [fcf * df for fcf, df in zip(fcf_projection, discount_factors)]
    sum_pv_cashflows = sum(pv_cashflows)
    
    terminal_value = (fcf_projection[-1] * (1 + (dcf_terminal / 100))) / ((dcf_wacc - dcf_terminal) / 100)
    pv_terminal_value = terminal_value * discount_factors[-1]
    
    enterprise_value = sum_pv_cashflows + pv_terminal_value
    equity_value = enterprise_value + active_data["cash"] - active_data["debt"]
    implied_share_value = equity_value / active_data["shares"]
    
    live_price = active_data["price"]
    
    with col_chart:
        if live_price:
            currency = active_data["currency"]
            mos = (1 - (live_price / implied_share_value)) * 100
            mos_text = f"Margin of Safety: **{mos:.2f}%**" if mos >= 0 else f"Implied Overvaluation: **{abs(mos):.2f}%**"
            mos_color = "#3fb950" if mos >= 0 else "#f85149"
            
            st.markdown(f"""
            <div style='background-color:#121824; border:1px solid #212836; border-radius:8px; padding:15px; margin-bottom:15px;'>
                <div style='display:flex; justify-content:space-between;'>
                    <div>
                        <span style='color:#8b949e; font-size:12px;'>Implied Fair Value (DCF)</span><br/>
                        <span style='font-size:24px; font-weight:700; color:#58a6ff;'>{currency} {implied_share_value:,.2f}</span>
                    </div>
                    <div>
                        <span style='color:#8b949e; font-size:12px;'>Current Market Price</span><br/>
                        <span style='font-size:24px; font-weight:700; color:#c9d1d9;'>{currency} {live_price:,.2f}</span>
                    </div>
                    <div>
                        <span style='color:#8b949e; font-size:12px;'>Valuation Appraisal</span><br/>
                        <span style='font-size:18px; font-weight:700; color:{mos_color};'>{mos_text}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=implied_share_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Fair Value appraiser ({currency})", 'font': {'size': 18, 'color': '#e6edf3'}},
                delta={'reference': live_price, 'increasing': {'color': "#3fb950"}, 'decreasing': {'color': "#f85149"}},
                gauge={
                    'axis': {'range': [0, max(implied_share_value, live_price) * 1.5], 'tickwidth': 1, 'tickcolor': "#8b949e"},
                    'bar': {'color': "#58a6ff"},
                    'bgcolor': "#161b22",
                    'borderwidth': 2,
                    'bordercolor': "#30363d",
                    'steps': [{'range': [0, live_price], 'color': '#21262d'}],
                    'threshold': {'line': {'color': "#bc8cff", 'width': 4}, 'thickness': 0.75, 'value': live_price}
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"}, height=300, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

    # Detailed Cash Flow Bridge Table
    st.markdown("#### ⏳ 5-Year Cash Flow Projections ($B)")
    years_label = [f"Year {i} (Projected)" for i in range(1, 6)]
    dcf_df = pd.DataFrame({
        "Projected Revenue ($B)": [round(x, 2) for x in rev_projection],
        "Projected FCF ($B)": [round(x, 2) for x in fcf_projection],
        "Discount Factor": [round(x, 4) for x in discount_factors],
        "Present Value FCF ($B)": [round(x, 2) for x in pv_cashflows]
    }, index=years_label)
    st.dataframe(dcf_df.T, use_container_width=True)

    # Sensitivity Heatmap Matrix
    st.markdown("#### 🌡️ Sensitivity Analysis (Fair Value vs WACC & Terminal Growth)")
    wacc_test = np.linspace(dcf_wacc - 2.0, dcf_wacc + 2.0, 5)
    tg_test = np.linspace(dcf_terminal - 1.0, dcf_terminal + 1.0, 5)
    
    sens_matrix = []
    for w in wacc_test:
        row = []
        for tg in tg_test:
            if w <= tg:
                row.append(np.nan)
                continue
            fcf_5 = fcf_projection[-1]
            tv = (fcf_5 * (1 + (tg / 100))) / ((w - tg) / 100)
            
            dfs = [1 / ((1 + (w / 100)) ** yr) for yr in range(1, 6)]
            pv_fcf = sum(f * d for f, d in zip(fcf_projection, dfs))
            pv_tv = tv * dfs[-1]
            ev = pv_fcf + pv_tv
            eq = ev + active_data["cash"] - active_data["debt"]
            val_per_share = eq / active_data["shares"]
            row.append(round(val_per_share, 2))
        sens_matrix.append(row)
        
    sens_df = pd.DataFrame(
        sens_matrix, 
        index=[f"WACC {w:.1f}%" for w in wacc_test],
        columns=[f"TG {t:.1f}%" for t in tg_test]
    )
    
    fig_heat = px.imshow(
        sens_df,
        labels=dict(x="Terminal Growth Rate", y="Discount Rate (WACC)", color="Share Value"),
        x=sens_df.columns,
        y=sens_df.index,
        color_continuous_scale="Viridis",
        title="Share Value Sensitivity Matrix"
    )
    fig_heat.update
