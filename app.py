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
from google import genai
from google.genai import types
from google.genai import errors

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="ValuEdge: Equity Research & Valuation Platform",
    page_icon="📈",
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
# LOAD VECTOR RESOURCES (Cached)
# =====================================
@st.cache_resource
def load_vector_resources():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("financialIndex.faiss")

    with open("companyChunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    vectors = np.load("financialVectors.npy")

    # Map chunks by company name
    positions_by_company = {}
    for i, chunk in enumerate(chunks):
        company = chunk.get("company", "Unknown") if isinstance(chunk, dict) else "Unknown"
        positions_by_company.setdefault(company, []).append(i)

    vectors_by_company = {
        company: vectors[positions] for company, positions in positions_by_company.items()
    }

    return embedding_model, index, chunks, vectors, positions_by_company, vectors_by_company

embedding_model, index, chunks, vectors, positions_by_company, vectors_by_company = load_vector_resources()
companies = sorted(positions_by_company.keys())

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
# LIVE MARKET TICKER CARDS (Sidebar)
# =====================================
@st.cache_data(ttl=300)
def fetch_sidebar_market_data():
    prices = {}
    for company, ticker_symbol in TICKERS.items():
        try:
            ticker = yf.Ticker(ticker_symbol)
            fast = ticker.fast_info
            current_price = fast.get("last_price")
            prev_close = fast.get("previous_close")
            
            if current_price is None or prev_close is None:
                # Fallback to info
                info = ticker.info
                current_price = info.get("currentPrice") or info.get("regularMarketPrice")
                prev_close = info.get("regularMarketPreviousClose")
            
            change = 0.0
            pct_change = 0.0
            if current_price and prev_close:
                change = current_price - prev_close
                pct_change = (change / prev_close) * 100
                
            prices[company] = {
                "price": current_price,
                "change": change,
                "pct_change": pct_change,
                "ticker": ticker_symbol
            }
        except Exception:
            prices[company] = {"price": None, "change": 0, "pct_change": 0, "ticker": ticker_symbol}
    return prices

sidebar_prices = fetch_sidebar_market_data()

st.sidebar.markdown("<h2 style='color:#e6edf3; font-size:18px;'>📡 Live Market Feeds</h2>", unsafe_allow_html=True)
for company, data in sidebar_prices.items():
    if data["price"]:
        color = "#3fb950" if data["change"] >= 0 else "#f85149"
        sign = "+" if data["change"] >= 0 else ""
        currency = "₹" if "NS" in data["ticker"] else "$"
        st.sidebar.markdown(f"""
        <div style='background-color:#121824; border:1px solid #212836; border-radius:6px; padding:10px 15px; margin-bottom:10px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span style='font-weight:bold; font-size:14px; color:#c9d1d9;'>{company}</span>
                <span class='ticker-badge'>{data["ticker"]}</span>
            </div>
            <div style='margin-top:5px;'>
                <span style='font-size:18px; font-weight:700; color:#58a6ff;'>{currency}{data["price"]:,.2f}</span>
                <span style='font-size:12px; font-weight:600; color:{color}; margin-left:10px;'>
                    {sign}{data["change"]:,.2f} ({sign}{data["pct_change"]:.2f}%)
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.sidebar.caption("No live ticker data loaded.")

# Sidebar API settings
st.sidebar.divider()
st.sidebar.markdown("<h2 style='color:#e6edf3; font-size:18px;'>🔑 LLM Configuration</h2>", unsafe_allow_html=True)
user_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Add key for custom queries.")
api_key = user_api_key or st.secrets.get("GEMINI_API_KEY", "")

if api_key:
    st.sidebar.success("Gemini API Key Loaded!")
else:
    st.sidebar.info("Demo Mode (Cache-only active). Input key for custom prompts.")

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
# APP LOGIC HEADER
# =====================================
st.markdown("<h1 class='main-title'>ValuEdge: Quantitative Equity Research & Valuation Platform</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyze NVIDIA, Microsoft, and Reliance Industries through interactive DCF Valuation models, du Pont Decomposition, Portfolio Backtesting, and Vector RAG search.</p>", unsafe_allow_html=True)

# Main Application Tabs
tab_val, tab_dup, tab_port, tab_rag = st.tabs([
    "📈 Dynamic DCF Valuer", 
    "🕸️ du Pont Profitability", 
    "📊 Portfolio Optimizer & Backtest",
    "🧠 Hybrid RAG AI Analyst"
])

# =====================================
# TAB 1: DYNAMIC DCF VALUER
# =====================================
with tab_val:
    st.markdown("### 📈 Interactive Discounted Cash Flow (DCF) Modeller")
    st.caption("Customize valuation parameters using actual FY25 annual report baselines and live ticker pricing.")
    
    val_company = st.selectbox("Select Target Company", ["NVIDIA", "Microsoft", "Reliance"])
    base_data = VERIFIED_FINANCIALS[val_company]
    
    col_input, col_chart = st.columns([1, 2])
    
    with col_input:
        st.markdown(f"**Baseline FY25 Metrics for {val_company}:**")
        st.caption(f"Revenue: **${base_data['Revenue']}B** | FCF Margin: **{base_data['fcf_margin']}%** | Outstanding Shares: **{base_data['shares_outstanding']}B**")
        
        # User dynamic overrides
        dcf_rev_growth = st.slider("5-Year Revenue CAGR (%)", 0.0, 150.0, base_data["Revenue Growth %"] if val_company == "NVIDIA" else 20.0, step=0.5, key="dcf_rg")
        dcf_fcf_margin = st.slider("Target FCF Margin (%)", 1.0, 60.0, base_data["fcf_margin"], step=0.5, key="dcf_fcfm")
        dcf_wacc = st.slider("Discount Rate / WACC (%)", 5.0, 20.0, 9.5 if val_company != "Reliance" else 11.5, step=0.1, key="dcf_wacc")
        dcf_terminal = st.slider("Terminal Growth Rate (%)", 0.5, 6.0, 2.5 if val_company != "Reliance" else 3.5, step=0.1, key="dcf_tg")
        
        # Safe constraint check
        if dcf_wacc <= dcf_terminal:
            st.error("WACC must be strictly greater than Terminal Growth Rate.")
            dcf_wacc = dcf_terminal + 0.5
            
    # Calculate DCF Projection
    rev_projection = []
    fcf_projection = []
    
    current_rev = base_data["Revenue"]
    for yr in range(1, 6):
        current_rev = current_rev * (1 + (dcf_rev_growth / 100))
        current_fcf = current_rev * (dcf_fcf_margin / 100)
        rev_projection.append(current_rev)
        fcf_projection.append(current_fcf)
        
    # Discount calculations
    discount_factors = [1 / ((1 + (dcf_wacc / 100)) ** yr) for yr in range(1, 6)]
    pv_cashflows = [fcf * df for fcf, df in zip(fcf_projection, discount_factors)]
    sum_pv_cashflows = sum(pv_cashflows)
    
    # Terminal value Gordon model
    terminal_value = (fcf_projection[-1] * (1 + (dcf_terminal / 100))) / ((dcf_wacc - dcf_terminal) / 100)
    pv_terminal_value = terminal_value * discount_factors[-1]
    
    enterprise_value = sum_pv_cashflows + pv_terminal_value
    equity_value = enterprise_value + base_data["cash"] - base_data["debt"]
    implied_share_value = equity_value / base_data["shares_outstanding"]
    
    # Get current live stock price
    live_price = sidebar_prices[val_company]["price"]
    
    with col_chart:
        # Gauge Chart Comparison
        if live_price:
            currency = "₹" if "NS" in base_data.get("ticker", TICKERS[val_company]) else "$"
            
            # Show valuation details card
            mos = (1 - (live_price / implied_share_value)) * 100
            mos_text = f"Margin of Safety: **{mos:.2f}%**" if mos >= 0 else f"Implied Overvaluation: **{abs(mos):.2f}%**"
            mos_color = "#3fb950" if mos >= 0 else "#f85149"
            
            st.markdown(f"""
            <div style='background-color:#121824; border:1px solid #212836; border-radius:8px; padding:15px; margin-bottom:15px;'>
                <div style='display:flex; justify-content:space-between;'>
                    <div>
                        <span style='color:#8b949e; font-size:12px;'>Implied Fair Value (DCF)</span><br/>
                        <span style='font-size:24px; font-weight:700; color:#58a6ff;'>{currency}{implied_share_value:,.2f}</span>
                    </div>
                    <div>
                        <span style='color:#8b949e; font-size:12px;'>Current Market Price</span><br/>
                        <span style='font-size:24px; font-weight:700; color:#c9d1d9;'>{currency}{live_price:,.2f}</span>
                    </div>
                    <div>
                        <span style='color:#8b949e; font-size:12px;'>Valuation Appraisal</span><br/>
                        <span style='font-size:18px; font-weight:700; color:{mos_color};'>{mos_text}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=implied_share_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Fair Value Appraiser ({currency})", 'font': {'size': 18, 'color': '#e6edf3'}},
                delta={'reference': live_price, 'increasing': {'color': "#3fb950"}, 'decreasing': {'color': "#f85149"}},
                gauge={
                    'axis': {'range': [0, max(implied_share_value, live_price) * 1.5], 'tickwidth': 1, 'tickcolor': "#8b949e"},
                    'bar': {'color': "#58a6ff"},
                    'bgcolor': "#161b22",
                    'borderwidth': 2,
                    'bordercolor': "#30363d",
                    'steps': [
                        {'range': [0, live_price], 'color': '#21262d'}
                    ],
                    'threshold': {
                        'line': {'color': "#bc8cff", 'width': 4},
                        'thickness': 0.75,
                        'value': live_price
                    }
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"}, height=300, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.warning("Could not fetch current live price to compare.")

    # Detailed Cash Flow Bridge Table
    st.markdown("#### ⏳ 5-Year Cash Flow Projection details")
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
            # project final year cash flows
            fcf_5 = fcf_projection[-1]
            tv = (fcf_5 * (1 + (tg / 100))) / ((w - tg) / 100)
            
            # discount factors
            dfs = [1 / ((1 + (w / 100)) ** yr) for yr in range(1, 6)]
            pv_fcf = sum(f * d for f, d in zip(fcf_projection, dfs))
            pv_tv = tv * dfs[-1]
            ev = pv_fcf + pv_tv
            eq = ev + base_data["cash"] - base_data["debt"]
            val_per_share = eq / base_data["shares_outstanding"]
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
    fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"}, height=350)
    st.plotly_chart(fig_heat, use_container_width=True)

# =====================================
# TAB 2: DU PONT PROFITABILITY
# =====================================
with tab_dup:
    st.markdown("### 🕸️ du Pont Profitability Decomposition")
    st.caption("Breaks down Return on Equity (ROE) into operational efficiency, asset utilization, and financial leverage ratios.")
    
    dup_choice = st.selectbox("Select Target Company", ["NVIDIA", "Microsoft", "Reliance"], key="dup_choice")
    d_data = VERIFIED_FINANCIALS[dup_choice]
    
    # 3-step DuPont Tree representation
    st.markdown(f"""
    <div style='background-color:#121824; border:1px solid #212836; border-radius:8px; padding:20px; margin-bottom:20px;'>
        <div style='text-align:center; margin-bottom:20px;'>
            <span style='color:#8b949e; font-size:14px; text-transform:uppercase;'>Return on Equity (ROE)</span><br/>
            <span style='font-size:42px; font-weight:800; color:#bc8cff;'>{d_data['roe']:.2f}%</span>
        </div>
        <div style='display:flex; justify-content:space-around; text-align:center;'>
            <div style='flex:1; border-right:1px solid #212836;'>
                <span style='color:#8b949e; font-size:12px;'>Net Profit Margin</span><br/>
                <span style='font-size:24px; font-weight:700; color:#58a6ff;'>{d_data['net_margin']:.2f}%</span><br/>
                <span style='color:#8b949e; font-size:11px;'>Net Income / Revenue</span>
            </div>
            <div style='flex:1; border-right:1px solid #212836;'>
                <span style='color:#8b949e; font-size:12px;'>Asset Turnover</span><br/>
                <span style='font-size:24px; font-weight:700; color:#58a6ff;'>{d_data['asset_turnover']:.2f}x</span><br/>
                <span style='color:#8b949e; font-size:11px;'>Revenue / Total Assets</span>
            </div>
            <div style='flex:1;'>
                <span style='color:#8b949e; font-size:12px;'>Equity Multiplier (Leverage)</span><br/>
                <span style='font-size:24px; font-weight:700; color:#58a6ff;'>{d_data['leverage']:.2f}x</span><br/>
                <span style='color:#8b949e; font-size:11px;'>Total Assets / Equity</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Benchmarking DuPont components side-by-side
    st.markdown("#### 📊 du Pont Component Benchmarking (FY25)")
    
    dup_bench = []
    for c, val in VERIFIED_FINANCIALS.items():
        dup_bench.append({"Company": c, "Net Profit Margin (%)": val["net_margin"], "Asset Turnover (x)": val["asset_turnover"], "Financial Leverage (x)": val["leverage"], "ROE (%)": val["roe"]})
    df_dup = pd.DataFrame(dup_bench)
    
    # Plotly bar charts comparing ratios side-by-side
    fig_dup_margin = px.bar(df_dup, x="Company", y="Net Profit Margin (%)", color="Company", title="Net profit margins (%)", text_auto=".2f")
    fig_dup_margin.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"})
    
    fig_dup_turn = px.bar(df_dup, x="Company", y="Asset Turnover (x)", color="Company", title="Asset Turnover Ratios (x)", text_auto=".2f")
    fig_dup_turn.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"})
    
    fig_dup_lev = px.bar(df_dup, x="Company", y="Financial Leverage (x)", color="Company", title="Financial Leverage / Equity Multipliers (x)", text_auto=".2f")
    fig_dup_lev.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"})
    
    col_m, col_t, col_l = st.columns(3)
    col_m.plotly_chart(fig_dup_margin, use_container_width=True)
    col_t.plotly_chart(fig_dup_turn, use_container_width=True)
    col_l.plotly_chart(fig_dup_lev, use_container_width=True)
    
    st.markdown("""
    > [!NOTE]
    > **Strategic Insights:**
    > * **NVIDIA** drives its industry-leading ROE (**85.18%**) through massive operational margins (**55.85%** Net Profit Margin), indicating huge pricing power.
    > * **Microsoft** relies on balanced margins (**36.14%**) and software scale, with moderate asset turnover and leverage.
    > * **Reliance** operates as a capital-intensive conglomerate, using financial leverage (**2.24x**) to magnify lower operating margins (**7.59%**), which is typical for logistics, telecom, and oil refining industries.
    """)

# =====================================
# TAB 3: PORTFOLIO OPTIMIZER & BACKTEST
# =====================================
with tab_port:
    st.markdown("### 📊 Portfolio Optimizer & Historical Backtester")
    st.caption("Simulate portfolio performance using daily closing prices for NVIDIA, Microsoft, and Reliance Industries.")
    
    col_alloc, col_perf = st.columns([1, 2])
    
    with col_alloc:
        st.markdown("**1. Configure Asset Allocation**")
        w_nvda = st.slider("NVIDIA Allocation (%)", 0, 100, 40)
        w_msft = st.slider("Microsoft Allocation (%)", 0, 100, 40)
        w_rel = st.slider("Reliance Allocation (%)", 0, 100, 20)
        
        total_w = w_nvda + w_msft + w_rel
        
        if total_w != 100:
            st.warning(f"Allocations sum to **{total_w}%**. We will auto-normalize them to **100%**.")
            n_nvda = w_nvda / total_w
            n_msft = w_msft / total_w
            n_rel = w_rel / total_w
        else:
            n_nvda = w_nvda / 100.0
            n_msft = w_msft / 100.0
            n_rel = w_rel / 100.0
            
        st.caption(f"Normalized Weights: **NVDA: {n_nvda*100:.1f}%** | **MSFT: {n_msft*100:.1f}%** | **RELIANCE: {n_rel*100:.1f}%**")
        
        # Duration choice
        duration_yrs = st.selectbox("Backtest History Period", [1, 2, 5], index=1)
        st.markdown("**Benchmarked Index**: S&P 500 (`^GSPC`)")
        
        run_backtest = st.button("🚀 Run Backtest Engine", use_container_width=True)
        
    with col_perf:
        if run_backtest:
            with st.spinner("Downloading historical pricing feeds..."):
                end_d = datetime.today()
                start_d = end_d - timedelta(days=duration_yrs * 365)
                
                tickers = ["NVDA", "MSFT", "RELIANCE.NS"]
                
                try:
                    df_prices = yf.download(tickers, start=start_d, end=end_d)["Close"]
                    df_benchmark = yf.download("^GSPC", start=start_d, end=end_d)["Close"]
                    
                    if df_prices.empty or df_benchmark.empty:
                        st.error("Failed to download historical prices. Try again.")
                    else:
                        # Clean pricing data
                        df_prices = df_prices.ffill().bfill()
                        df_benchmark = df_benchmark.ffill().bfill()
                        
                        # Align indexes
                        df_prices, df_benchmark = df_prices.align(df_benchmark, join='inner', axis=0)
                        
                        # Calculate daily returns
                        returns = df_prices.pct_change().dropna()
                        bench_returns = df_benchmark.pct_change().dropna()
                        
                        # Portfolio weights vector (Sorted alphabetically: MSFT, NVDA, RELIANCE.NS)
                        w_vector = np.array([n_msft, n_nvda, n_rel])
                        
                        portfolio_daily = returns.dot(w_vector)
                        
                        # Resolve single-column dataframes to 1D series
                        if isinstance(portfolio_daily, pd.DataFrame):
                            portfolio_daily = portfolio_daily.squeeze()
                        if isinstance(bench_returns, pd.DataFrame):
                            bench_daily = bench_returns.squeeze()
                        else:
                            bench_daily = bench_returns
                            
                        if isinstance(portfolio_daily, pd.DataFrame):
                            portfolio_daily = portfolio_daily.iloc[:, 0]
                        if isinstance(bench_daily, pd.DataFrame):
                            bench_daily = bench_daily.iloc[:, 0]
                        
                        # Cumulative returns
                        cum_portfolio = (1 + portfolio_daily).cumprod() - 1
                        cum_benchmark = (1 + bench_daily).cumprod() - 1
                        
                        # Compute performance statistics
                        ann_return_p = float(portfolio_daily.mean() * 252 * 100)
                        ann_vol_p = float(portfolio_daily.std() * np.sqrt(252) * 100)
                        sharpe_p = float((ann_return_p - 4.0) / ann_vol_p) if ann_vol_p > 0 else 0.0
                        
                        cum_returns_plus_one = (1 + portfolio_daily).cumprod()
                        running_max = cum_returns_plus_one.cummax()
                        drawdowns = (cum_returns_plus_one - running_max) / running_max
                        max_drawdown = float(drawdowns.min() * 100)
                        
                        # Bench statistics
                        ann_return_b = float(bench_daily.mean() * 252 * 100)
                        ann_vol_b = float(bench_daily.std() * np.sqrt(252) * 100)
                        sharpe_b = float((ann_return_b - 4.0) / ann_vol_b) if ann_vol_b > 0 else 0.0
                        
                        # Display metrics side-by-side
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        col_m1.metric("Annualized Return", f"{ann_return_p:.1f}%", f"{ann_return_p - ann_return_b:+.1f}% vs Index")
                        col_m2.metric("Annualized Volatility", f"{ann_vol_p:.1f}%", f"{ann_vol_p - ann_vol_b:+.1f}% vs Index", delta_color="inverse")
                        col_m3.metric("Sharpe Ratio (Rf=4%)", f"{sharpe_p:.2f}", f"{sharpe_p - sharpe_b:+.2f} vs Index")
                        col_m4.metric("Max Drawdown", f"{max_drawdown:.1f}%", help="Peak to valley drawdown")
                        
                        # Plot cumulative performance chart
                        fig_df = pd.DataFrame({
                            "Portfolio": cum_portfolio * 100,
                            "S&P 500 Index": cum_benchmark * 100
                        }, index=returns.index)
                        
                        fig_perf = px.line(fig_df, y=["Portfolio", "S&P 500 Index"], title=f"Cumulative Portfolio Return vs S&P 500 ({duration_yrs} Year Backtest)")
                        fig_perf.update_layout(
                            yaxis_title="Cumulative Return (%)",
                            xaxis_title="Date",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': "#e6edf3"},
                            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                        )
                        st.plotly_chart(fig_perf, use_container_width=True)
                except Exception as e:
                    st.error(f"Backtesting error: {str(e)}")
        else:
            # Show initial empty state instructions
            st.info("Click 'Run Backtest Engine' on the left panel to execute portfolio optimization simulation.")
            
            # Renders basic asset allocation pie chart
            pie_df = pd.DataFrame({
                "Asset": ["NVIDIA", "Microsoft", "Reliance"],
                "Weight": [n_nvda, n_msft, n_rel]
            })
            fig_pie = px.pie(pie_df, names="Asset", values="Weight", hole=0.35, title="Configured Asset Allocation Mix")
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"})
            st.plotly_chart(fig_pie, use_container_width=True)

# =====================================
# TAB 4: HYBRID RAG AI ANALYST
# =====================================
# Direct Analysis logic RAG
def retrieve_context_chunks(query_vector, target_companies_list=None):
    if target_companies_list:
        k = max(8, 24 // len(target_companies_list))
        retrieved = []
        for company in target_companies_list:
            if company in vectors_by_company:
                company_vectors = vectors_by_company[company]
                dists = np.linalg.norm(company_vectors - query_vector, axis=1)
                top_k = np.argsort(dists)[:k]
                for idx in top_k:
                    global_pos = positions_by_company[company][idx]
                    retrieved.append((chunks[global_pos], dists[idx]))
        # sort by distance
        retrieved.sort(key=lambda x: x[1])
        return [item[0] for item in retrieved], [item[1] for item in retrieved]
    else:
        D, I = index.search(query_vector.astype("float32"), 15)
        matched_chunks = [chunks[int(i)] for i in I[0] if 0 <= int(i) < len(chunks)]
        return matched_chunks, list(D[0])

def generate_rag_content(query, context):
    system_prompt = f"""
You are a senior equity research analyst answering a client's query based on company annual report text.

CONTEXT EXCERPTS:
{context}

QUERY:
{query}

Respond in clean, valid JSON format, matching exactly this structure:
{{
  "summary": "2-3 sentence executive summary answering the question",
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
1. comparison_table: only include actual numerical figures backed by the context.
2. segment_breakdown: only populate if the question asks for segment/divisional splits. Otherwise, leave empty.
3. currencies: keep units exactly as reported (USD for NVDA/MSFT, INR Crore for Reliance).
"""
    client = genai.Client(api_key=api_key)
    
    # Try gemini-2.5-flash with a lite fallback
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=system_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        return response.text
    except errors.ClientError as e:
        if e.code == 429: # try fallback
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=system_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            return response.text
        raise e

# Helpers to detect target companies in a query
COMPANY_ALIASES = {
    "NVIDIA": ["nvidia", "nvda"],
    "Microsoft": ["microsoft", "msft"],
    "Reliance": ["reliance", "ril", "jio"]
}

def parse_target_companies(query):
    q = query.lower()
    targets = [c for c, aliases in COMPANY_ALIASES.items() if any(a in q for a in aliases)]
    return targets if targets else None

with tab_rag:
    st.markdown("### 🧠 Hybrid RAG AI Analyst")
    st.caption("Ask questions about NVIDIA, Microsoft, and Reliance Industries. If no API key is set, common demo questions will be served via the pre-warmed cache.")
    
    # Standard pre-warmed question templates
    st.markdown("**Try a standard comparison:**")
    eq_cols = st.columns(3)
    examples = [
        "Compare revenue and net income across NVIDIA, Microsoft and Reliance",
        "Compare the AI strategies of Microsoft and NVIDIA",
        "What are the key risk factors for Reliance Industries?"
    ]
    for col, ex in zip(eq_cols, examples):
        if col.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.rag_query = ex
            
    rag_query = st.text_input("Ask a financial question", key="rag_query")
    
    if rag_query:
        # Check cache first
        query_hash = hashlib.sha256(rag_query.strip().lower().encode()).hexdigest()[:16]
        cache_path = os.path.join(DEMO_CACHE_DIR, f"direct_{query_hash}.json")
        
        data = None
        retrieved_chunks = []
        distances = []
        is_cached = False
        
        # Load embedding and search FAISS
        query_vector = embedding_model.encode([rag_query]).astype("float32")
        targets_detected = parse_target_companies(rag_query)
        retrieved_chunks, distances = retrieve_context_chunks(query_vector, targets_detected)
        
        # Build context block
        context_parts = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            comp = chunk.get("company", "Unknown") if isinstance(chunk, dict) else "Unknown"
            txt = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            context_parts.append(f"[Source {idx} | {comp}]\n{txt}")
        context = "\n\n".join(context_parts)
        
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                data = json.load(f)
            is_cached = True
            st.caption("📌 Cached response loaded. Zero API quota used.")
        else:
            if not api_key:
                st.error("No Gemini API Key loaded. This query is not in the demo cache. Please enter your API Key in the sidebar to run custom prompts.")
                st.stop()
                
            with st.spinner("Agent is retrieving & parsing filings..."):
                try:
                    raw_resp = generate_rag_content(rag_query, context)
                    raw_clean = re.sub(r"^```(json)?|```$", "", raw_resp.strip()).strip()
                    data = json.loads(raw_clean)
                except Exception as e:
                    st.error(f"Error executing LLM parser: {str(e)}")
                    
        # Renders the parsed findings
        if data:
            st.markdown(f"#### 🔍 Executive Summary")
            st.info(data.get("summary", ""))
            
            findings = data.get("key_findings") or []
            if findings:
                st.markdown("**Key Findings:**")
                for f in findings:
                    st.markdown(f"- {f}")
                    
            # Pivot comparison table
            table = data.get("comparison_table") or []
            if table:
                df = pd.DataFrame(table)
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.dropna(subset=["value", "metric", "company"])
                
                if not df.empty:
                    df["unit"] = df["unit"].fillna("")
                    df["metric_label"] = df["metric"] + df["unit"].apply(lambda u: f" ({u})" if u else "")
                    
                    st.markdown("#### 📋 Parsed Comparison Data")
                    pivot = df.pivot_table(index="metric_label", columns="company", values="value", aggfunc="first")
                    st.dataframe(pivot.style.format("{:,.2f}", na_rep="—"), use_container_width=True)
                    
                    # Render plotly bar chart panel
                    fig_cmp = px.bar(df, x="company", y="value", color="company", facet_col="metric_label", text_auto=".2s")
                    fig_cmp.update_yaxes(matches=None)
                    fig_cmp.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                    fig_cmp.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"})
                    st.plotly_chart(fig_cmp, use_container_width=True)
                    
            # Segment Pie charts
            segments = data.get("segment_breakdown") or []
            if segments:
                df_seg = pd.DataFrame(segments)
                df_seg["value"] = pd.to_numeric(df_seg["value"], errors="coerce")
                df_seg = df_seg.dropna(subset=["value", "segment", "company"])
                
                if not df_seg.empty:
                    st.markdown("#### 🥧 Divisional Revenue Split")
                    seg_companies = df_seg["company"].unique()
                    pie_cols = st.columns(len(seg_companies))
                    
                    for p_col, comp in zip(pie_cols, seg_companies):
                        g = df_seg[df_seg["company"] == comp]
                        fig_pie = px.pie(g, names="segment", values="value", title=f"{comp} Segments", hole=0.3)
                        fig_pie.update_traces(textinfo="percent+label", showlegend=False)
                        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"}, height=280)
                        p_col.plotly_chart(fig_pie, use_container_width=True)
                        
            # Risks
            risks = data.get("risks") or []
            if risks:
                st.markdown("#### ⚠️ Highlighted Corporate Risks")
                for r in risks:
                    st.warning(r)
                    
        # RAG Inspector
        st.markdown("#### 📚 Interactive Source Inspector")
        with st.expander("Inspect Retrieved Report Chunks & Vector Distance Metrics"):
            for idx, (chunk, dist) in enumerate(zip(retrieved_chunks, distances), start=1):
                comp = chunk.get("company", "Unknown") if isinstance(chunk, dict) else "Unknown"
                txt = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                st.markdown(f"**Source {idx} — {comp}** (Vector L2 Distance: `{dist:.4f}`)")
                st.text(txt[:600] + "...")
                st.divider()
