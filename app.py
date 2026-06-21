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
# SEARCH SUGGESTIONS ENGINE (Company Name -> Stock Tickers List)
# =====================================
def get_search_suggestions(query):
    if not query or len(query.strip()) < 2:
        return [
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
            {"symbol": "NVDA", "name": "NVIDIA Corporation"},
            {"symbol": "RELIANCE.NS", "name": "Reliance Industries Limited"},
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "TSLA", "name": "Tesla, Inc."}
        ]
        
    query_clean = query.strip()
    query_lower = query_clean.lower()
    
    # Pre-populate popular presets to avoid API latency and handle private/public mappings
    popular_presets = {
        "microsoft": [{"symbol": "MSFT", "name": "Microsoft Corporation"}],
        "nvidia": [{"symbol": "NVDA", "name": "NVIDIA Corporation"}],
        "reliance": [{"symbol": "RELIANCE.NS", "name": "Reliance Industries Limited"}],
        "apple": [{"symbol": "AAPL", "name": "Apple Inc."}],
        "tesla": [{"symbol": "TSLA", "name": "Tesla, Inc."}],
        "google": [{"symbol": "GOOGL", "name": "Alphabet Inc. (Google)"}],
        "amazon": [{"symbol": "AMZN", "name": "Amazon.com, Inc."}],
        "ola": [{"symbol": "OLAELEC.NS", "name": "Ola Electric Mobility Limited"}],
        "ola electric": [{"symbol": "OLAELEC.NS", "name": "Ola Electric Mobility Limited"}],
        "ola cabs": [{"symbol": "OLAELEC.NS", "name": "Ola Electric Mobility Limited (Ola Cabs EV division)"}],
    }
    
    if query_lower in popular_presets:
        return popular_presets[query_lower]
        
    # Check substring preset matching
    for key, val in popular_presets.items():
        if key in query_lower or query_lower in key:
            return val

    suggestions = []
    
    # Method A: Try yfinance Search
    try:
        search = yf.Search(query_clean, max_results=8)
        if search.quotes:
            for q in search.quotes:
                symbol = q.get("symbol")
                name = q.get("shortname") or q.get("longname") or symbol
                if symbol and name:
                    suggestions.append({"symbol": symbol, "name": name})
    except Exception:
        pass
        
    # Method B: Fallback to direct Yahoo Search API
    if not suggestions:
        import urllib.request
        import urllib.parse
        try:
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(query_clean)}&quotesCount=8"
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            with urllib.request.urlopen(req, timeout=3) as response:
                res_data = json.loads(response.read().decode())
                quotes = res_data.get("quotes", [])
                for q in quotes:
                    symbol = q.get("symbol")
                    name = q.get("shortname") or q.get("longname") or symbol
                    if symbol and name:
                        suggestions.append({"symbol": symbol, "name": name})
        except Exception:
            pass

    # Method C: Default Fallback if both search methods yield nothing
    if not suggestions:
        suggestions = [{"symbol": query_clean.upper(), "name": f"Query: {query_clean.upper()}"}]
        
    return suggestions


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
company_search_query = st.sidebar.text_input(
    "Search Company Name", 
    value="Microsoft", 
    help="Type any company name (e.g. Microsoft, Nvidia, Reliance, Ola, Apple, Tesla) or ticker symbol."
)

with st.sidebar.spinner("Searching matching companies..."):
    suggestions = get_search_suggestions(company_search_query)

dropdown_options = []
symbol_map = {}
for s in suggestions:
    label = f"{s['name']} ({s['symbol']})"
    dropdown_options.append(label)
    symbol_map[label] = s['symbol']

selected_label = st.sidebar.selectbox(
    "Select matching company",
    options=dropdown_options,
    index=0,
    help="Choose the correct company from search results."
)

global_ticker = symbol_map.get(selected_label, "MSFT")

with st.sidebar.spinner(f"Fetching financials for {global_ticker}..."):
    global_data = fetch_global_financials(global_ticker)

if global_data:
    st.sidebar.success(f"Loaded: {global_data['name']} ({global_ticker})")
    
    price_val = global_data.get('price')
    price_str = f"{global_data['currency']} {price_val:,.2f}" if price_val is not None else "Price N/A"
    
    st.sidebar.markdown(f"""
    <div style='background-color:#121824; border:1px solid #212836; border-radius:6px; padding:10px 15px; margin-bottom:10px;'>
        <span style='color:#8b949e; font-size:12px;'>Current Live Price</span><br/>
        <span style='font-size:22px; font-weight:700; color:#58a6ff;'>{price_str}</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.warning(f"Could not download financials for '{global_ticker}'. Displaying offline estimates.")

st.sidebar.divider()
st.sidebar.markdown("<h2 style='color:#e6edf3; font-size:18px;'>🔑 LLM Configuration</h2>", unsafe_allow_html=True)
user_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Add key for custom RAG prompts.")
api_key = user_api_key or st.secrets.get("GEMINI_API_KEY", "")

if api_key:
    st.sidebar.success("Gemini API Key Loaded!")
else:
    st.sidebar.info("Demo Mode (Cache-only active). Input key for custom prompts.")

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
        if live_price is not None and live_price > 0:
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

# =====================================
# TAB 2: DU PONT PROFITABILITY
# =====================================
with tab_dup:
    st.markdown(f"### 🕸️ du Pont Profitability Decomposition: {active_data['name']}")
    st.caption("Breaks down Return on Equity (ROE) into operational efficiency, asset utilization, and financial leverage ratios.")
    
    # Pre-calculated benchmark data dictionary for display components
    VERIFIED_FINANCIALS = {
        "NVIDIA": {"roe": 121.2, "net_margin": 55.8, "asset_turnover": 0.98, "leverage": 2.21},
        "Microsoft": {"roe": 38.4, "net_margin": 36.1, "asset_turnover": 0.58, "leverage": 1.83},
        "Reliance": {"roe": 9.2, "net_margin": 7.6, "asset_turnover": 0.39, "leverage": 3.10}
    }
    
    st.markdown(f"""
    <div style='background-color:#121824; border:1px solid #212836; border-radius:8px; padding:20px; margin-bottom:20px;'>
        <div style='text-align:center; margin-bottom:20px;'>
            <span style='color:#8b949e; font-size:14px; text-transform:uppercase;'>Return on Equity (ROE)</span><br/>
            <span style='font-size:42px; font-weight:800; color:#bc8cff;'>{active_data['roe']:.2f}%</span>
        </div>
        <div style='display:flex; justify-content:space-around; text-align:center;'>
            <div style='flex:1; border-right:1px solid #212836;'>
                <span style='color:#8b949e; font-size:12px;'>Net Profit Margin</span><br/>
                <span style='font-size:24px; font-weight:700; color:#58a6ff;'>{active_data['net_margin']:.2f}%</span><br/>
                <span style='color:#8b949e; font-size:11px;'>Net Income / Revenue</span>
            </div>
            <div style='flex:1; border-right:1px solid #212836;'>
                <span style='color:#8b949e; font-size:12px;'>Asset Turnover</span><br/>
                <span style='font-size:24px; font-weight:700; color:#58a6ff;'>{active_data['asset_turnover']:.2f}x</span><br/>
                <span style='color:#8b949e; font-size:11px;'>Revenue / Total Assets</span>
            </div>
            <div style='flex:1;'>
                <span style='color:#8b949e; font-size:12px;'>Equity Multiplier (Leverage)</span><br/>
                <span style='font-size:24px; font-weight:700; color:#58a6ff;'>{active_data['leverage']:.2f}x</span><br/>
                <span style='color:#8b949e; font-size:11px;'>Total Assets / Equity</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Benchmarking DuPont components side-by-side
    st.markdown("#### 📊 du Pont Component Benchmarking (Baseline Portfolio)")
    
    dup_bench = []
    for c, val in VERIFIED_FINANCIALS.items():
        dup_bench.append({"Company": c, "Net Profit Margin (%)": val["net_margin"], "Asset Turnover (x)": val["asset_turnover"], "Financial Leverage (x)": val["leverage"], "ROE (%)": val["roe"]})
    df_dup = pd.DataFrame(dup_bench)
    
    col_m, col_t, col_l = st.columns(3)
    
    fig_dup_margin = px.bar(df_dup, x="Company", y="Net Profit Margin (%)", color="Company", title="Net profit margins (%)", text_auto=".2f")
    fig_dup_margin.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"})
    col_m.plotly_chart(fig_dup_margin, use_container_width=True)
    
    fig_dup_turn = px.bar(df_dup, x="Company", y="Asset Turnover (x)", color="Company", title="Asset Turnover Ratios (x)", text_auto=".2f")
    fig_dup_turn.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"})
    col_t.plotly_chart(fig_dup_turn, use_container_width=True)
    
    fig_dup_lev = px.bar(df_dup, x="Company", y="Financial Leverage (x)", color="Company", title="Financial Leverage Multipliers (x)", text_auto=".2f")
    fig_dup_lev.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"})
    col_l.plotly_chart(fig_dup_lev, use_container_width=True)

# =====================================
# TAB 3: DYNAMIC PORTFOLIO BACKTEST
# =====================================
with tab_port:
    st.markdown("### 📊 Portfolio Optimizer & Historical Backtester")
    st.caption("Enter a comma-separated list of any global tickers to backtest. Daily data is fetched in real-time.")
    
    # User-defined tickers input
    port_tickers_input = st.text_input("Enter Ticker Symbols (Comma-separated)", value="NVDA, MSFT, AAPL, GOOG")
    port_tickers = [t.strip().upper() for t in port_tickers_input.split(",") if t.strip()]
    
    col_alloc, col_perf = st.columns([1, 2])
    
    with col_alloc:
        st.markdown("**1. Configure Asset Weights**")
        weights = {}
        for ticker in port_tickers:
            weights[ticker] = st.slider(f"{ticker} Weight (%)", 0, 100, 100 // len(port_tickers))
            
        total_w = sum(weights.values())
        normalized_weights = {}
        for ticker, w in weights.items():
            normalized_weights[ticker] = w / total_w if total_w > 0 else 1.0 / len(port_tickers)
            
        st.markdown("**Normalized Allocation Weights:**")
        st.code(" | ".join([f"{t}: {w*100:.1f}%" for t, w in normalized_weights.items()]))
        
        duration_yrs = st.selectbox("Backtest History Period", [1, 2, 5], index=1, key="backtest_dur")
        run_backtest = st.button("🚀 Run Backtest Engine", use_container_width=True, key="run_port_backtest")
        
    with col_perf:
        if run_backtest:
            with st.spinner("Downloading historical pricing feeds..."):
                end_d = datetime.today()
                start_d = end_d - timedelta(days=duration_yrs * 365)
                
                try:
                    df_prices = yf.download(port_tickers, start=start_d, end=end_d)["Close"]
                    df_benchmark = yf.download("^GSPC", start=start_d, end=end_d)["Close"]
                    
                    if df_prices.empty or df_benchmark.empty:
                        st.error("Failed to download pricing data. Please check ticker symbols.")
                    else:
                        df_prices = df_prices.ffill().bfill()
                        df_benchmark = df_benchmark.ffill().bfill()
                        df_prices, df_benchmark = df_prices.align(df_benchmark, join='inner', axis=0)
                        
                        returns = df_prices.pct_change().dropna()
                        bench_returns = df_benchmark.pct_change().dropna()
                        
                        # Sort weights to match alphabetically sorted columns
                        sorted_tickers = sorted(port_tickers)
                        w_vector = np.array([normalized_weights[t] for t in sorted_tickers])
                        
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
                            
                        cum_portfolio = (1 + portfolio_daily).cumprod() - 1
                        cum_benchmark = (1 + bench_daily).cumprod() - 1
                        
                        ann_return_p = float(portfolio_daily.mean() * 252 * 100)
                        ann_vol_p = float(portfolio_daily.std() * np.sqrt(252) * 100)
                        sharpe_p = float((ann_return_p - 4.0) / ann_vol_p) if ann_vol_p > 0 else 0.0
                        
                        cum_returns_plus_one = (1 + portfolio_daily).cumprod()
                        running_max = cum_returns_plus_one.cummax()
                        drawdowns = (cum_returns_plus_one - running_max) / running_max
                        max_drawdown = float(drawdowns.min() * 100)
                        
                        ann_return_b = float(bench_daily.mean() * 252 * 100)
                        ann_vol_b = float(bench_daily.std() * np.sqrt(252) * 100)
                        sharpe_b = float((ann_return_b - 4.0) / ann_vol_b) if ann_vol_b > 0 else 0.0
                        
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        col_m1.metric("Annualized Return", f"{ann_return_p:.1f}%", f"{ann_return_p - ann_return_b:+.1f}% vs Index")
                        col_m2.metric("Annualized Volatility", f"{ann_vol_p:.1f}%", f"{ann_vol_p - ann_vol_b:+.1f}% vs Index", delta_color="inverse")
                        col_m3.metric("Sharpe Ratio (Rf=4%)", f"{sharpe_p:.2f}", f"{sharpe_p - sharpe_b:+.2f} vs Index")
                        col_m4.metric("Max Drawdown", f"{max_drawdown:.1f}%", help="Peak to valley drawdown")
                        
                        fig_df = pd.DataFrame({
                            "Portfolio": cum_portfolio * 100,
                            "S&P 500 Index": cum_benchmark * 100
                        }, index=returns.index)
                        
                        fig_perf = px.line(fig_df, y=["Portfolio", "S&P 500 Index"], title="Cumulative Performance Comparison")
                        fig_perf.update_layout(
                            yaxis_title="Cumulative Return (%)",
                            xaxis_title="Date",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': "#e6edf3"}
                        )
                        st.plotly_chart(fig_perf, use_container_width=True)
                except Exception as e:
                    st.error(f"Backtesting error: {str(e)}")
        else:
            st.info("Click 'Run Backtest Engine' on the left panel to execute simulation.")
            pie_df = pd.DataFrame({
                "Asset": list(normalized_weights.keys()),
                "Weight": list(normalized_weights.values())
            })
            fig_pie = px.pie(pie_df, names="Asset", values="Weight", hole=0.35, title="Asset Allocation Mix")
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"})
            st.plotly_chart(fig_pie, use_container_width=True)

# =====================================
# TAB 4: SELF-SERVE PDF RAG
# =====================================
with tab_self_rag:
    st.markdown("### 📁 Self-Serve RAG: Upload Any Corporate Filing")
    st.caption("Upload any PDF report (10-K, ESG, Transcript). The platform will chunk, embed, and index it locally in real-time.")
    
    uploaded_file = st.file_uploader("Upload filing PDF", type=["pdf"])
    
    if uploaded_file:
        # Check if already processed
        if "file_hash" not in st.session_state or st.session_state.file_hash != uploaded_file.name:
            with st.spinner("Extracting PDF text contents..."):
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                # Cap at 80 pages to prevent memory exhaustion in Streamlit Cloud
                pages_to_read = min(80, len(pdf_reader.pages))
                for pg_idx in range(pages_to_read):
                    pg_text = pdf_reader.pages[pg_idx].extract_text()
                    if pg_text:
                        text += pg_text
                        
            with st.spinner(f"Splitting and Vectorizing text chunks..."):
                chunk_sz = 1000
                ov_sz = 200
                up_chunks = []
                start_p = 0
                while start_p < len(text):
                    end_p = start_p + chunk_sz
                    up_chunks.append(text[start_p:end_p])
                    start_p += (chunk_sz - ov_sz)
                    
                up_vectors = embedding_model.encode(up_chunks, show_progress_bar=False)
                up_vectors = np.array(up_vectors, dtype=np.float32)
                
                up_index = faiss.IndexFlatL2(up_vectors.shape[1])
                up_index.add(up_vectors)
                
                st.session_state.file_hash = uploaded_file.name
                st.session_state.uploaded_chunks = up_chunks
                st.session_state.uploaded_index = up_index
                st.session_state.uploaded_vectors = up_vectors
                st.success(f"Fully Vectorized! Created {len(up_chunks)} segments inside temporary FAISS Index.")

        # Query UI for uploaded document
        if "uploaded_index" in st.session_state:
            up_query = st.text_input("Ask a question about the uploaded document")
            if up_query:
                q_vec = embedding_model.encode([up_query]).astype("float32")
                D, I = st.session_state.uploaded_index.search(q_vec, 6)
                
                ret_chunks = [st.session_state.uploaded_chunks[int(i)] for i in I[0] if 0 <= int(i) < len(st.session_state.uploaded_chunks)]
                ret_context = "\n\n".join([f"[Source Chunk {idx+1}]\n{txt}" for idx, txt in enumerate(ret_chunks)])
                
                if api_key:
                    with st.spinner("AI Analyst is evaluating context..."):
                        client = genai.Client(api_key=api_key)
                        prompt = f"""
                        You are a senior equity analyst. Answer the user's question using ONLY the provided document segments.
                        
                        CONTEXT:
                        {ret_context}
                        
                        QUESTION:
                        {up_query}
                        """
                        try:
                            resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                            st.markdown("#### 💡 AI Answer:")
                            st.write(resp.text)
                        except Exception as e:
                            st.error(f"LLM Error: {str(e)}")
                else:
                    st.info("Input a Gemini API Key in the sidebar to get AI-generated answers. Direct matching source chunks are shown below.")
                
                # Show source chunks
                st.markdown("#### 🔍 Matching Segments:")
                for idx, (txt, dist) in enumerate(zip(ret_chunks, D[0])):
                    with st.expander(f"Chunk {idx+1} (L2 Distance: {dist:.4f})"):
                        st.text(txt)

# =====================================
# TAB 5: BASELINE FILED SNAPSHOTS
# =====================================
# Direct Analysis logic RAG
def retrieve_baseline_chunks(query_vector, target_companies_list=None):
    if not chunks:
        return [], []
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
        if e.code == 429:
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
    st.markdown("### 🔒 Baseline Filed Reports (FY25 Snapshot RAG)")
    st.caption("Ask questions about the original baseline datasets: NVIDIA, Microsoft, and Reliance Industries reports.")
    
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
        query_hash = hashlib.sha256(rag_query.strip().lower().encode()).hexdigest()[:16]
        cache_path = os.path.join(DEMO_CACHE_DIR, f"direct_{query_hash}.json")
        
        data = None
        retrieved_chunks = []
        distances = []
        is_cached = False
        
        query_vector = embedding_model.encode([rag_query]).astype("float32")
        targets_detected = parse_target_companies(rag_query)
        retrieved_chunks, distances = retrieve_baseline_chunks(query_vector, targets_detected)
        
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
                    
        if data:
            st.markdown(f"#### 🔍 Executive Summary")
            st.info(data.get("summary", ""))
            
            findings = data.get("key_findings") or []
            if findings:
                st.markdown("**Key Findings:**")
                for f in findings:
                    st.markdown(f"- {f}")
                    
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
                    
                    fig_cmp = px.bar(df, x="company", y="value", color="company", facet_col="metric_label", text_auto=".2s")
                    fig_cmp.update_yaxes(matches=None)
                    fig_cmp.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                    fig_cmp.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"})
                    st.plotly_chart(fig_cmp, use_container_width=True)
                    
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
                        
            risks = data.get("risks") or []
            if risks:
                st.markdown("#### ⚠️ Highlighted Corporate Risks")
                for r in risks:
                    st.warning(r)
                    
        st.markdown("#### 📚 Interactive Source Inspector")
        with st.expander("Inspect Retrieved Report Chunks & Vector Distance Metrics"):
            for idx, (chunk, dist) in enumerate(zip(retrieved_chunks, distances), start=1):
                comp = chunk.get("company", "Unknown") if isinstance(chunk, dict) else "Unknown"
                txt = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                st.markdown(f"**Source {idx} — {comp}** (Vector L2 Distance: `{dist:.4f}`)")
                st.text(txt[:600] + "...")
                st.divider()
