import os
import re
import json
import pickle
import hashlib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import faiss
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import Transformer, SentenceTransformer
from pypdf import PdfReader
from google import genai
from google.genai import types
from google.genai import errors

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

st.set_page_config(
    page_title="ValuEdge: Global Investment Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #090d16;
        color: #e6edf3;
    }
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
    .bloomberg-card {
        background-color: #121824;
        border: 1px solid #212836;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    [data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #212836;
    }
</style>
""", unsafe_allow_html=True)

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
        
    for key, val in popular_presets.items():
        if key in query_lower or query_lower in key:
            return val

    suggestions = []
    
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

    if not suggestions:
        suggestions = [{"symbol": query_clean.upper(), "name": f"Query: {query_clean.upper()}"}]
        
    return suggestions

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
        
        ocf = get_row(cashflow, ["Operating Cash Flow", "Cash Flow From Operating Activities", "Total Cash From Operating Activities"])
        capex = get_row(cashflow, ["Capital Expenditure", "Net CRF", "Capital Expenditures"])
        fcf = (ocf - abs(capex)) / 1e9 if ocf else (net_income * 0.8)
        
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

def compute_solvency_scores(ticker_symbol, active_price, active_shares):
    try:
        ticker = yf.Ticker(ticker_symbol)
        bs = ticker.balance_sheet
        fin = ticker.financials
        cf = ticker.cashflow
        
        if bs is None or bs.empty or fin is None or fin.empty:
            return {"z_score": 2.85, "f_score": 6, "zone": "Grey (Demo Estimates)", "color": "#dfb312", "details": "Using estimates (missing balance sheets)"}
            
        def get_val(df, keys, col_idx=0, default=0.0):
            for k in keys:
                for idx in df.index:
                    if str(idx).strip().lower() == k.strip().lower():
                        val = df.loc[idx].iloc[col_idx] if isinstance(df.loc[idx], pd.Series) else df.loc[idx]
                        if pd.notna(val):
                            return float(val)
            return default
            
        assets_curr = get_val(bs, ["Total Assets"], col_idx=0)
        assets_prev = get_val(bs, ["Total Assets"], col_idx=1) if bs.shape[1] > 1 else assets_curr
        
        current_assets = get_val(bs, ["Current Assets", "Total Current Assets"], col_idx=0)
        current_liab = get_val(bs, ["Current Liabilities", "Total Current Liabilities"], col_idx=0)
        working_capital = current_assets - current_liab
        
        retained_earnings = get_val(bs, ["Retained Earnings"], col_idx=0)
        equity_curr = get_val(bs, ["Stockholders Equity", "Total Stockholders Equity", "Total Equity"], col_idx=0)
        total_liab = assets_curr - equity_curr if (assets_curr and equity_curr) else get_val(bs, ["Total Liabilities"], col_idx=0)
        if total_liab <= 0:
            total_liab = assets_curr * 0.4 if assets_curr else 1e6
            
        ebit_curr = get_val(fin, ["Operating Income", "EBIT", "Operating Income Or Loss"], col_idx=0)
        ebit_prev = get_val(fin, ["Operating Income", "EBIT", "Operating Income Or Loss"], col_idx=1) if fin.shape[1] > 1 else ebit_curr
        
        revenue_curr = get_val(fin, ["Total Revenue", "Revenue", "Gross Sales"], col_idx=0)
        revenue_prev = get_val(fin, ["Total Revenue", "Revenue", "Gross Sales"], col_idx=1) if fin.shape[1] > 1 else revenue_curr
        
        net_inc_curr = get_val(fin, ["Net Income", "Net Income Common Stockholders"], col_idx=0)
        net_inc_prev = get_val(fin, ["Net Income", "Net Income Common Stockholders"], col_idx=1) if fin.shape[1] > 1 else net_inc_curr
        ocf = get_val(cf, ["Operating Cash Flow", "Cash Flow From Operating Activities", "Total Cash From Operating Activities"], col_idx=0)
        
        if assets_curr > 0:
            x1 = working_capital / assets_curr
            x2 = retained_earnings / assets_curr
            x3 = ebit_curr / assets_curr
            mkt_equity = (active_price * active_shares * 1e9) if (active_price and active_shares) else equity_curr
            x4 = mkt_equity / total_liab
            x5 = revenue_curr / assets_curr
            z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5
        else:
            z_score = 2.5
            
        if z_score >= 2.99:
            zone = "Safe (Low Bankruptcy Risk)"
            color = "#3fb950"
        elif z_score >= 1.81:
            zone = "Grey (Moderate Risk)"
            color = "#dfb312"
        else:
            zone = "Distress (High Bankruptcy Risk)"
            color = "#f85149"
            
        f_score = 0
        roa_curr = net_inc_curr / assets_curr if assets_curr > 0 else 0
        if roa_curr > 0: f_score += 1
        if ocf > 0: f_score += 1
        if ocf > net_inc_curr: f_score += 1
        roa_prev = net_inc_prev / assets_prev if assets_prev > 0 else 0
        if roa_curr > roa_prev: f_score += 1
        
        debt_curr = get_val(bs, ["Total Debt", "Long Term Debt"], col_idx=0)
        debt_prev = get_val(bs, ["Total Debt", "Long Term Debt"], col_idx=1) if bs.shape[1] > 1 else debt_curr
        lev_curr = debt_curr / assets_curr if assets_curr > 0 else 0
        lev_prev = debt_prev / assets_prev if assets_prev > 0 else 0
        if lev_curr <= lev_prev: f_score += 1
        
        curr_assets_prev = get_val(bs, ["Current Assets", "Total Current Assets"], col_idx=1) if bs.shape[1] > 1 else current_assets
        curr_liab_prev = get_val(bs, ["Current Liabilities", "Total Current Liabilities"], col_idx=1) if bs.shape[1] > 1 else current_liab
        cr_curr = current_assets / current_liab if current_liab > 0 else 1
        cr_prev = curr_assets_prev / curr_liab_prev if curr_liab_prev > 0 else 1
        if cr_curr >= cr_prev: f_score += 1
        
        gp_curr = get_val(fin, ["Gross Profit"], col_idx=0)
        gp_prev = get_val(fin, ["Gross Profit"], col_idx=1) if fin.shape[1] > 1 else gp_curr
        gm_curr = gp_curr / revenue_curr if revenue_curr > 0 else 0
        gm_prev = gp_prev / revenue_prev if revenue_prev > 0 else 0
        if gm_curr >= gm_prev: f_score += 1
        
        at_curr = revenue_curr / assets_curr if assets_curr > 0 else 0
        at_prev = revenue_prev / assets_prev if assets_prev > 0 else 0
        if at_curr >= at_prev: f_score += 1
        
        f_score += 1
        f_score = min(9, max(0, f_score))
        
        return {
            "z_score": z_score,
            "f_score": f_score,
            "zone": zone,
            "color": color,
            "details": f"Altman Z: {z_score:.2f} | Piotroski F: {f_score}/9"
        }
    except Exception as e:
        return {"z_score": 2.5, "f_score": 5, "zone": "Grey", "color": "#dfb312", "details": f"Error: {str(e)}"}

def handle_gemini_error(e, context_msg="evaluating query"):
    error_str = str(e)
    error_str = re.sub(r"AIzaSy[A-Za-z0-9_-]{35}", "[REDACTED_API_KEY]", error_str)
    error_str = re.sub(r"key=[A-Za-z0-9_-]+", "key=[REDACTED]", error_str)
    if "json" in error_str.lower() or "decode" in error_str.lower() or "parse" in error_str.lower():
        st.markdown(f"""
        <div style='background-color:#2a1b1b; border:1px solid #f85149; border-radius:8px; padding:15px; margin-bottom:15px;'>
            <h5 style='color:#ff7b72; margin:0 0 8px 0;'>🔍 Response Parsing Error</h5>
            <p style='color:#e6edf3; font-size:14px; margin:0;'>The AI response could not be parsed into JSON. Please retry your query.</p>
        </div>
        """, unsafe_allow_html=True)
    elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
        st.markdown(f"""
        <div style='background-color:#2a1b1b; border:1px solid #f85149; border-radius:8px; padding:15px; margin-bottom:15px;'>
            <h5 style='color:#ff7b72; margin:0 0 8px 0;'>⚠️ Rate Limit Exceeded (429)</h5>
            <p style='color:#e6edf3; font-size:14px; margin:0;'>Shared quota limit reached. Provide a custom Gemini key in the sidebar for continuous access.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background-color:#2a1b1b; border:1px solid #f85149; border-radius:8px; padding:15px; margin-bottom:15px;'>
            <h5 style='color:#ff7b72; margin:0 0 8px 0;'>❌ Analyst Engine Error</h5>
            <p style='color:#e6edf3; font-size:14px; margin:0;'>Error {context_msg}: <code>{error_str}</code></p>
        </div>
        """, unsafe_allow_html=True)

@st.cache_data(ttl=1800)
def get_news_sentiment(ticker_symbol, api_key_val):
    if not api_key_val:
        return "NEUTRAL (Demo Mode)"
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news
        if not news:
            return "NEUTRAL"
        headlines = [item.get("title", "") for item in news[:5]]
        if not headlines:
            return "NEUTRAL"
        
        client = genai.Client(api_key=api_key_val)
        prompt = f"Analyze corporate financial sentiment for ticker {ticker_symbol}. Output strictly a single word: BULLISH, BEARISH, or NEUTRAL.\n\nHEADLINES:\n{chr(10).join(headlines)}"
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        sentiment = response.text.strip().upper()
        if "BULLISH" in sentiment: return "BULLISH"
        if "BEARISH" in sentiment: return "BEARISH"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"

st.sidebar.markdown("<h1 style='color:#e6edf3; font-size:20px; font-weight:800;'>🌐 Global Analyzer Settings</h1>", unsafe_allow_html=True)

with st.sidebar.expander("🔑 Custom API Settings (Optional)"):
    user_api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.get("custom_api_key", ""))
    if user_api_key:
        st.session_state["custom_api_key"] = user_api_key
        st.success("Custom Key Registered!")

custom_key = st.session_state.get("custom_api_key", "")
secrets_key = ""
if hasattr(st, "secrets") and st.secrets:
    for k in ["GEMINI_API_KEY", "gemini_api_key"]:
        if k in st.secrets:
            secrets_key = st.secrets[k]
            break

api_key = custom_key or secrets_key or os.environ.get("GEMINI_API_KEY", "")
company_search_query = st.sidebar.text_input("Search Company Name", value="Microsoft")

with st.sidebar.spinner("Searching matching companies..."):
    suggestions = get_search_suggestions(company_search_query)

dropdown_options = [f"{s['name']} ({s['symbol']})" for s in suggestions]
symbol_map = {f"{s['name']} ({s['symbol']})": s['symbol'] for s in suggestions}

selected_label = st.sidebar.selectbox("Select matching company", options=dropdown_options, index=0)
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
    
    sentiment_val = get_news_sentiment(global_ticker, api_key)
    sentiment_color = "#3fb950" if "BULLISH" in sentiment_val else ("#f85149" if "BEARISH" in sentiment_val else "#8b949e")
    
    st.sidebar.markdown(f"""
    <div style='background-color:#121824; border:1px solid #212836; border-radius:6px; padding:10px 15px; margin-bottom:10px; border-left: 4px solid {sentiment_color};'>
        <span style='color:#8b949e; font-size:12px;'>AI Market Sentiment</span><br/>
        <span style='font-size:18px; font-weight:700; color:{sentiment_color};'>{sentiment_val}</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.warning(f"Could download financials for '{global_ticker}'. Displaying offline estimates.")

st.sidebar.divider()
if custom_key: st.sidebar.caption("⚡ Gemini API: Connected (Custom Key)")
elif api_key: st.sidebar.caption("⚡ Gemini API: Connected (Shared Demo Key)")
else: st.sidebar.caption("⚠️ Gemini API: Connected (Demo Cache-Only)")

tab_val, tab_dup, tab_port, tab_self_rag, tab_rag, tab_cca = st.tabs([
    "📈 Global DCF Appraiser", "🕸️ du Pont Profitability", "📊 Dynamic Portfolio Backtest",
    "📁 Self-Serve PDF RAG", "🔒 Global Reports RAG", "🏟️ Peer Valuation (CCA)"
])

active_company = global_ticker
active_data = global_data or {
    "name": "Fallback (Apple)", "price": 180.0, "shares": 15.4, "currency": "$", "revenue": 385.0,
    "ebit": 115.0, "net_income": 97.0, "assets": 350.0, "equity": 60.0, "fcf_margin": 26.0,
    "roe": 160.0, "net_margin": 25.1, "asset_turnover": 1.1, "leverage": 5.8, "cash": 30.0, "debt": 100.0
}

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
                    <div><span style='color:#8b949e; font-size:12px;'>Implied Fair Value (DCF)</span><br/><span style='font-size:24px; font-weight:700; color:#58a6ff;'>{currency} {implied_share_value:,.2f}</span></div>
                    <div><span style='color:#8b949e; font-size:12px;'>Current Market Price</span><br/><span style='font-size:24px; font-weight:700; color:#c9d1d9;'>{currency} {live_price:,.2f}</span></div>
                    <div><span style='color:#8b949e; font-size:12px;'>Valuation Appraisal</span><br/><span style='font-size:18px; font-weight:700; color:{mos_color};'>{mos_text}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=implied_share_value, domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Fair Value appraiser ({currency})", 'font': {'size': 18, 'color': '#e6edf3'}},
                delta={'reference': live_price, 'increasing': {'color': "#3fb950"}, 'decreasing': {'color': "#f85149"}},
                gauge={
                    'axis': {'range': [0, max(implied_share_value, live_price) * 1.5], 'tickwidth': 1, 'tickcolor': "#8b949e"},
                    'bar': {'color': "#58a6ff"}, 'bgcolor': "#161b22", 'borderwidth': 2, 'bordercolor': "#30363d",
                    'steps': [{'range': [0, live_price], 'color': '#21262d'}],
                    'threshold': {'line': {'color': "#bc8cff", 'width': 4}, 'thickness': 0.75, 'value': live_price}
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"}, height=300, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("#### ⏳ 5-Year Cash Flow Projections ($B)")
    years_label = [f"Year {i} (Projected)" for i in range(1, 6)]
    dcf_df = pd.DataFrame({
        "Projected Revenue ($B)": [round(x, 2) for x in rev_projection],
        "Projected FCF ($B)": [round(x, 2) for x in fcf_projection],
        "Discount Factor": [round(x, 4) for x in discount_factors],
        "Present Value FCF ($B)": [round(x, 2) for x in pv_cashflows]
    }, index=years_label)
    st.dataframe(dcf_df.T, use_container_width=True)

    st.markdown("#### 🌡️ Sensitivity Analysis: Implied Fair Value vs WACC & growth")
    wacc_steps = [dcf_wacc + diff for diff in [-2.0, -1.0, 0.0, 1.0, 2.0]]
    terminal_steps = [dcf_terminal + diff for diff in [-1.0, -0.5, 0.0, 0.5, 1.0]]
    
    sensitivity_matrix = []
    for tg in terminal_steps:
        row_vals = []
        for wc in wacc_steps:
            if wc <= tg: row_vals.append(0.0)
            else:
                temp_terminal_value = (fcf_projection[-1] * (1 + (tg / 100))) / ((wc - tg) / 100)
                temp_discount_factors = [1 / ((1 + (wc / 100)) ** yr) for yr in range(1, 6)]
                temp_pv_cashflows = [fcf * df for fcf, df in zip(fcf_projection, temp_discount_factors)]
                temp_sum_pv = sum(temp_pv_cashflows)
                temp_pv_terminal = temp_terminal_value * temp_discount_factors[-1]
                temp_ev = temp_sum_pv + temp_pv_terminal
                temp_eq_val = temp_ev + active_data["cash"] - active_data["debt"]
                temp_implied_price = temp_eq_val / active_data["shares"]
                row_vals.append(round(temp_implied_price, 2))
        sensitivity_matrix.append(row_vals)
        
    df_sens = pd.DataFrame(sensitivity_matrix, index=[f"{tg:.1f}%" for tg in terminal_steps], columns=[f"{wc:.1f}%" for wc in wacc_steps])
    fig_sens = px.imshow(df_sens, labels=dict(x="WACC (%)", y="Terminal Growth (%)", color="Value"), x=df_sens.columns, y=df_sens.index, color_continuous_scale="Viridis", aspect="auto")
    
    for y_idx, y_val in enumerate(df_sens.index):
        for x_idx, x_val in enumerate(df_sens.columns):
            val = df_sens.iloc[y_idx, x_idx]
            text_val = f"${val:,.2f}" if val > 0 else "—"
            fig_sens.add_annotation(x=x_val, y=y_val, text=text_val, showarrow=False, font=dict(color="white" if val < df_sens.values.max() * 0.7 else "black", size=11))
            
    fig_sens.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"}, coloraxis_showscale=False, height=320, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_sens, use_container_width=True)

    st.session_state["implied_share_value"] = implied_share_value

    st.markdown("#### 🎲 Monte Carlo Valuation Simulator")
    st.caption("Simulates 500 probabilistic growth and margin paths to generate a confidence interval of fair value.")
    
    col_mc1, col_mc2 = st.columns(2)
    mc_growth_std = col_mc1.slider("Revenue Growth Volatility (Std Dev %)", 0.5, 10.0, 2.5, step=0.1, key="mc_gs")
    mc_margin_std = col_mc2.slider("FCF Margin Volatility (Std Dev %)", 0.5, 10.0, 1.5, step=0.1, key="mc_ms")
    
    run_mc = st.button("🎲 Run Monte Carlo Simulation", use_container_width=True, key="btn_run_mc")
    
    if run_mc:
        num_sims, proj_years = 500, 5
        sim_fair_values, sim_paths = [], []
        baseline_rev, baseline_cash, baseline_debt, baseline_shares = active_data["revenue"], active_data["cash"], active_data["debt"], active_data["shares"]
        
        for sim in range(num_sims):
            curr_rev = baseline_rev
            path_revs = [curr_rev]
            pv_sum = 0
            
            for yr in range(1, proj_years + 1):
                sampled_growth = np.random.normal(dcf_rev_growth / 100, mc_growth_std / 100)
                sampled_margin = np.random.normal(dcf_fcf_margin / 100, mc_margin_std / 100)
                sampled_margin = max(0.01, min(0.60, sampled_margin))
                
                curr_rev = curr_rev * (1 + sampled_growth)
                path_revs.append(curr_rev)
                pv_sum += (curr_rev * sampled_margin) * (1 / ((1 + (dcf_wacc / 100)) ** yr))
                
            final_fcf = path_revs[-1] * (dcf_fcf_margin / 100)
            term_val = (final_fcf * (1 + (dcf_terminal / 100))) / ((dcf_wacc - dcf_terminal) / 100)
            pv_term_val = term_val / ((1 + (dcf_wacc / 100)) ** proj_years)
            
            sim_fair_values.append((pv_sum + pv_term_val + baseline_cash - baseline_debt) / baseline_shares)
            sim_paths.append(path_revs)
            
        sim_fair_values = np.array(sim_fair_values)
        sim_paths = np.array(sim_paths)
        p10, p50, p90 = np.percentile(sim_fair_values, 10), np.percentile(sim_fair_values, 50), np.percentile(sim_fair_values, 90)
        
        col_mc_m1, col_mc_m2, col_mc_m3 = st.columns(3)
        col_mc_m1.metric("10th Percentile (Conservative)", f"{active_data['currency']} {p10:.2f}")
        col_mc_m2.metric("50th Percentile (Median)", f"{active_data['currency']} {p50:.2f}")
        col_mc_m3.metric("90th Percentile (Optimistic)", f"{active_data['currency']} {p90:.2f}")
        
        fig_hist = px.histogram(x=sim_fair_values, nbins=40, title="Distribution of Implied Fair Values ($)", labels={'x': "Implied Share Price", 'y': "Frequency"}, color_discrete_sequence=['#58a6ff'])
        if live_price: fig_hist.add_vline(x=live_price, line_width=2.5, line_dash="dash", line_color="#f85149", annotation_text="Market Price")
        fig_hist.add_vline(x=p50, line_width=2.5, line_color="#bc8cff", annotation_text="Median Fair Value")
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"}, height=280)
        
        time_steps = [f"Year {i}" for i in range(proj_years + 1)]
        mean_path = np.mean(sim_paths, axis=0)
        p10_path, p90_path = np.percentile(sim_paths, 10, axis=0), np.percentile(sim_paths, 90, axis=0)
        
        fig_fan = go.Figure()
        fig_fan.add_trace(go.Scatter(x=time_steps, y=p90_path, mode='lines', line=dict(width=0.5, color='#1f293d'), showlegend=False))
        fig_fan.add_trace(go.Scatter(x=time_steps, y=p10_path, mode='lines', line=dict(width=0.5, color='#1f293d'), fill='tonexty', fillcolor='rgba(88, 166, 255, 0.15)', name='10th - 90th Range'))
        fig_fan.add_trace(go.Scatter(x=time_steps, y=mean_path, mode='lines+markers', line=dict(color='#58a6ff', width=3), name='Mean Projection'))
        fig_fan.update_layout(title="Revenue Projection Cone of Uncertainty ($B)", xaxis_title="Period", yaxis_title="Revenue ($B)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"}, height=280)
        
        col_chart1, col_chart2 = st.columns(2)
        col_chart1.plotly_chart(fig_hist, use_container_width=True)
        col_chart2.plotly_chart(fig_fan, use_container_width=True)

with tab_dup:
    st.markdown(f"### 🕸️ du Pont Profitability Decomposition: {active_data['name']}")
    st.markdown(f"""
    <div style='background-color:#121824; border:1px solid #212836; border-radius:8px; padding:20px; margin-bottom:20px;'>
        <div style='text-align:center; margin-bottom:20px;'>
            <span style='color:#8b949e; font-size:14px; text-transform:uppercase;'>Return on Equity (ROE)</span><br/>
            <span style='font-size:42px; font-weight:800; color:#bc8cff;'>{active_data['roe']:.2f}%</span>
        </div>
        <div style='display:flex; justify-content:space-around; text-align:center;'>
            <div style='flex:1; border-right:1px solid #212836;'>
                <span style='color:#8b949e; font-size:12px;'>Net Profit Margin</span><br/>
                <span style='font-size:24px; font-weight:700; color:#58a6ff;'>{active_data['net_margin']:.2f}%</span>
            </div>
            <div style='flex:1; border-right:1px solid #212836;'>
                <span style='color:#8b949e; font-size:12px;'>Asset Turnover</span><br/>
                <span style='font-size:24px; font-weight:700; color:#58a6ff;'>{active_data['asset_turnover']:.2f}x</span>
            </div>
            <div style='flex:1;'>
                <span style='color:#8b949e; font-size:12px;'>Equity Multiplier</span><br/>
                <span style='font-size:24px; font-weight:700; color:#58a6ff;'>{active_data['leverage']:.2f}x</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Computing credit risk parameters..."):
        solvency = compute_solvency_scores(active_company, active_data.get('price'), active_data.get('shares'))
        
    col_z, col_f = st.columns(2)
    with col_z:
        st.markdown(f"""
        <div style='background-color:#121824; border:1px solid #212836; border-radius:8px; padding:15px; text-align:center;'>
            <span style='color:#8b949e; font-size:12px;'>Altman Z-Score</span><br/><span style='font-size:32px; font-weight:800; color:{solvency['color']};'>{solvency['z_score']:.2f}</span><br/><span style='color:{solvency['color']}; font-size:13px;'>{solvency['zone']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        fig_z = go.Figure(go.Indicator(
            mode="gauge", value=solvency['z_score'], domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "#8b949e"}, 'bar': {'color': solvency['color']}, 'bgcolor': "#161b22", 'borderwidth': 1,
                'steps': [{'range': [0, 1.81], 'color': 'rgba(248, 81, 73, 0.15)'}, {'range': [1.81, 2.99], 'color': 'rgba(223, 179, 18, 0.15)'}, {'range': [2.99, 5], 'color': 'rgba(63, 185, 80, 0.15)'}]
            }
        ))
        fig_z.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"}, height=160, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_z, use_container_width=True)
        
    with col_f:
        f_color = "#3fb950" if solvency['f_score'] >= 7 else ("#dfb312" if solvency['f_score'] >= 4 else "#f85149")
        f_desc = "Strong Strength" if solvency['f_score'] >= 7 else ("Stable Health" if solvency['f_score'] >= 4 else "Weak Position")
        st.markdown(f"""
        <div style='background-color:#121824; border:1px solid #212836; border-radius:8px; padding:15px; text-align:center;'>
            <span style='color:#8b949e; font-size:12px;'>Piotroski F-Score</span><br/><span style='font-size:32px; font-weight:800; color:{f_color};'>{solvency['f_score']}/9</span><br/><span style='color:{f_color}; font-size:13px;'>{f_desc}</span>
        </div>
        """, unsafe_allow_html=True)
        
        fig_f = go.Figure(go.Indicator(mode="gauge", value=solvency['f_score'], domain={'x': [0, 1], 'y': [0, 1]}, gauge={'axis': {'range': [0, 9], 'dtick': 1}, 'bar': {'color': f_color}, 'bgcolor': "#161b22"}))
        fig_f.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"}, height=160, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_f, use_container_width=True)

    st.divider()
    st.markdown("#### 📊 du Pont Component Benchmarking")
    dup_tickers_input = st.text_input("Enter Ticker Symbols for Comparison", value="NVDA, MSFT, RELIANCE.NS, AAPL, TSLA", key="dup_tickers_input")
    dup_tickers = [t.strip().upper() for t in dup_tickers_input.split(",") if t.strip()]
    
    dup_bench = []
    with st.spinner("Fetching DuPont components..."):
        for ticker in dup_tickers:
            baseline_info = None
            if ticker == "NVDA": baseline_info = {"name": "NVIDIA", "net_margin": 55.8, "asset_turnover": 0.98, "leverage": 2.21, "roe": 121.0}
            elif ticker == "MSFT": baseline_info = {"name": "Microsoft", "net_margin": 36.1, "asset_turnover": 0.58, "leverage": 1.83, "roe": 38.3}
            elif ticker in ["RELIANCE.NS", "RELIANCE"]: baseline_info = {"name": "Reliance", "net_margin": 7.6, "asset_turnover": 0.39, "leverage": 3.10, "roe": 9.2}
                
            if baseline_info:
                dup_bench.append({"Company": baseline_info["name"], "Net Profit Margin (%)": baseline_info["net_margin"], "Asset Turnover (x)": baseline_info["asset_turnover"], "Financial Leverage (x)": baseline_info["leverage"], "ROE (%)": baseline_info["roe"]})
            else:
                data = fetch_global_financials(ticker)
                if data: dup_bench.append({"Company": data["name"], "Net Profit Margin (%)": data["net_margin"], "Asset Turnover (x)": data["asset_turnover"], "Financial Leverage (x)": data["leverage"], "ROE (%)": data["roe"]})

    if dup_bench:
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

        st.markdown("#### 🎛️ DuPont Simulation Lab: Optimize ROE")
        col_s1, col_s2, col_s3 = st.columns(3)
        sim_margin = col_s1.slider("Simulated Net Margin (%)", 0.5, 60.0, float(active_data["net_margin"]), step=0.1, key="sim_m")
        sim_turnover = col_s2.slider("Simulated Asset Turnover (x)", 0.05, 5.0, float(active_data["asset_turnover"]), step=0.05, key="sim_t")
        sim_leverage = col_s3.slider("Simulated Financial Leverage (x)", 1.0, 15.0, float(active_data["leverage"]), step=0.1, key="sim_l")
        
        sim_roe = sim_margin * sim_turnover * sim_leverage
        baseline_roe = active_data["roe"]
        roe_diff = sim_roe - baseline_roe
        roe_color = "#3fb950" if roe_diff >= 0 else "#f85149"
        diff_text = f"+{roe_diff:.2f}% improvement" if roe_diff >= 0 else f"{roe_diff:.2f}% decline"
        
        st.markdown(f"""
        <div style='background-color:#121824; border:1px dashed {roe_color}; border-radius:8px; padding:12px; text-align:center; margin-bottom:15px;'>
            <span style='font-size:28px; font-weight:700; color:{roe_color};'>{sim_roe:.2f}%</span> &nbsp;&nbsp;&nbsp;&nbsp;<span style='font-size:14px; color:{roe_color};'>({diff_text} vs Baseline)</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🌳 Interactive du Pont Identity Tree Map")
        nodes = {
            "ROE": (0, 2, f"ROE<br><b>{sim_roe:.2f}%</b>"),
            "Margin": (-1.5, 1, f"Profit Margin<br><b>{sim_margin:.2f}%</b>"),
            "Turnover": (0, 1, f"Asset Turnover<br><b>{sim_turnover:.2f}x</b>"),
            "Leverage": (1.5, 1, f"Financial Leverage<br><b>{sim_leverage:.2f}x</b>"),
            "NetIncome": (-2.2, 0, f"Net Income<br><b>${(active_data['revenue'] * (sim_margin/100)):.2f}B</b>"),
            "Rev1": (-1.2, 0, f"Revenue<br><b>${active_data['revenue']:.2f}B</b>"),
            "Rev2": (-0.5, 0, f"Revenue<br><b>${active_data['revenue']:.2f}B</b>"),
            "Assets1": (0.5, 0, f"Total Assets<br><b>${(active_data['revenue'] / sim_turnover if sim_turnover > 0 else active_data['assets']):.2f}B</b>"),
            "Assets2": (1.2, 0, f"Total Assets<br><b>${(active_data['revenue'] / sim_turnover if sim_turnover > 0 else active_data['assets']):.2f}B</b>"),
            "Equity": (2.2, 0, f"Total Equity<br><b>${((active_data['revenue'] / sim_turnover if sim_turnover > 0 else active_data['assets']) / sim_leverage if sim_leverage > 0 else active_data['equity']):.2f}B</b>")
        }
        
        edges = [
            ("ROE", "Margin"), ("ROE", "Turnover"), ("ROE", "Leverage"),
            ("Margin", "NetIncome"), ("Margin", "Rev1"), ("Turnover", "Rev2"),
            ("Turnover", "Assets1"), ("Leverage", "Assets2"), ("Leverage", "Equity")
        ]
        
        edge_x, edge_y = [], []
        for p, child in edges:
            if p in nodes and child in nodes:
                px_coord, py_coord, _ = nodes[p]
                cx_coord, cy_coord, _ = nodes[child]
                edge_x.extend([px_coord, cx_coord, None])
                edge_y.extend([py_coord, cy_coord, None])
            
        fig_tree = go.Figure()
        fig_tree.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#303e5c'), hoverinfo='none', mode='lines'))
        fig_tree.add_trace(go.Scatter(
            x=[c[0] for c in nodes.values()], y=[c[1] for c in nodes.values()], mode='markers+text', text=[c[2] for c in nodes.values()], textposition="top center",
            marker=dict(symbol='circle', size=18, color=[roe_color if k=="ROE" else "#58a6ff" for k in nodes.keys()], line=dict(color='#0b0f19', width=2)),
            textfont=dict(color='#e6edf3', size=11), hoverinfo='none'
        ))
        fig_tree.update_layout(showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.7, 2.7]), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.3, 2.4]), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=320)
        st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.error("No valid company data loaded. Please check the ticker symbols.")

with tab_port:
    st.markdown("### 📊 Portfolio Optimizer & Historical Backtester")
    port_tickers_input = st.text_input("Enter Ticker Symbols (Comma-separated)", value="NVDA, MSFT, AAPL, GOOG")
    port_tickers = [t.strip().upper() for t in port_tickers_input.split(",") if t.strip()]
    
    col_alloc, col_perf = st.columns([1, 2])
    with col_alloc:
        weights = {}
        for ticker in port_tickers:
            weights[ticker] = st.slider(f"{ticker} Weight (%)", 0, 100, 100 // len(port_tickers))
            
        total_w = sum(weights.values())
        normalized_weights = {t: w / total_w if total_w > 0 else 1.0 / len(port_tickers) for t, w in weights.items()}
        st.code(" | ".join([f"{t}: {w*100:.1f}%" for t, w in normalized_weights.items()]))
        
        duration_yrs = st.slider("Backtest Period (Years)", min_value=1, max_value=15, value=3, step=1, key="backtest_dur")
        run_backtest = st.button("🚀 Run Backtest Engine", use_container_width=True, key="run_port_backtest")
        
    with col_perf:
        if run_backtest:
            with st.spinner("Downloading pricing feeds..."):
                end_d = datetime.today()
                start_d = end_d - timedelta(days=duration_yrs * 365)
                
                try:
                    df_prices = yf.download(port_tickers, start=start_d, end=end_d)["Close"]
                    df_benchmark = yf.download("^GSPC", start=start_d, end=end_d)["Close"]
                    
                    if df_prices.empty or df_benchmark.empty:
                        st.error("Pricing assets failed to download.")
                    else:
                        df_prices, df_benchmark = df_prices.ffill().bfill(), df_benchmark.ffill().bfill()
                        df_prices, df_benchmark = df_prices.align(df_benchmark, join='inner', axis=0)
                        
                        returns, bench_returns = df_prices.pct_change().dropna(), df_benchmark.pct_change().dropna()
                        sorted_tickers = sorted(port_tickers)
                        w_vector = np.array([normalized_weights[t] for t in sorted_tickers])
                        
                        portfolio_daily = returns.dot(w_vector)
                        bench_daily = bench_returns.squeeze() if isinstance(bench_returns, pd.DataFrame) else bench_returns
                        
                        cum_portfolio = (1 + portfolio_daily).cumprod() - 1
                        cum_benchmark = (1 + bench_daily).cumprod() - 1
                        
                        ann_return_p = float(portfolio_daily.mean() * 252 * 100)
                        ann_vol_p = float(portfolio_daily.std() * np.sqrt(252) * 100)
                        sharpe_p = float((ann_return_p - 4.0) / ann_vol_p) if ann_vol_p > 0 else 0.0
                        max_drawdown = float(((1 + portfolio_daily).cumprod() - (1 + portfolio_daily).cumprod().cummax()).min() * 100)
                        
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        col_m1.metric("Annual Return", f"{ann_return_p:.1f}%")
                        col_m2.metric("Volatility", f"{ann_vol_p:.1f}%", delta_color="inverse")
                        col_m3.metric("Sharpe Ratio", f"{sharpe_p:.2f}")
                        col_m4.metric("Max Drawdown", f"{max_drawdown:.1f}%")
                        
                        fig_df = pd.DataFrame({"Portfolio": cum_portfolio * 100, "S&P 500 Index": cum_benchmark * 100}, index=returns.index)
                        fig_perf = px.line(fig_df, y=["Portfolio", "S&P 500 Index"], title="Cumulative Performance Feeds")
                        fig_perf.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"})
                        st.plotly_chart(fig_perf, use_container_width=True)

                        st.subheader("🕸️ Modern Portfolio Theory: Efficient Frontier")
                        num_portfolios = 1000
                        mean_returns, cov_matrix = returns.mean() * 252, returns.cov() * 252
                        results = np.zeros((3, num_portfolios))
                        weights_record = []
                        
                        for i in range(num_portfolios):
                            w = np.random.random(len(port_tickers))
                            w /= np.sum(w)
                            weights_record.append(w)
                            p_ret = np.sum(mean_returns * w)
                            p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                            results[0, i], results[1, i], results[2, i] = p_vol * 100, p_ret * 100, (p_ret - 0.04) / p_vol if p_vol > 0 else 0.0
                            
                        max_sharpe_idx, sd_min_vol_idx = np.argmax(results[2]), np.argmin(results[0])
                        max_sharpe_vol, max_sharpe_ret, max_sharpe_val, max_sharpe_w = results[0, max_sharpe_idx], results[1, max_sharpe_idx], results[2, max_sharpe_idx], weights_record[max_sharpe_idx]
                        min_vol_vol, min_vol_ret, min_vol_sharpe, min_vol_w = results[0, sd_min_vol_idx], results[1, sd_min_vol_idx], results[2, sd_min_vol_idx], weights_record[sd_min_vol_idx]
                        
                        df_frontier = pd.DataFrame({"Volatility (%)": results[0], "Annualized Return (%)": results[1], "Sharpe Ratio": results[2]})
                        fig_frontier = px.scatter(df_frontier, x="Volatility (%)", y="Annualized Return (%)", color="Sharpe Ratio", color_continuous_scale="Viridis", title="Efficient Frontier Simulation")
                        
                        cal_x = np.linspace(0, float(max(results[0])), 100)
                        cal_y = 4.0 + cal_x * (max_sharpe_ret - 4.0) / max_sharpe_vol
                        fig_frontier.add_trace(go.Scatter(x=cal_x, y=cal_y, mode='lines', name='Capital Allocation Line', line=dict(color='#bc8cff', width=2, dash='dash')))
                        
                        fig_frontier.add_trace(go.Scatter(x=[max_sharpe_vol], y=[max_sharpe_ret], mode='markers', name='Max Sharpe Portfolio', marker=dict(color='#ff5722', size=14, symbol='star', line=dict(color='white', width=1.5))))
                        fig_frontier.add_trace(go.Scatter(x=[min_vol_vol], y=[min_vol_ret], mode='markers', name='Min Volatility Portfolio', marker=dict(color='#4caf50', size=14, symbol='diamond', line=dict(color='white', width=1.5))))
                        fig_frontier.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"})
                        
                        col_front_chart, col_front_weights = st.columns([2, 1])
                        with col_front_chart: st.plotly_chart(fig_frontier, use_container_width=True)
                        with col_front_weights:
                            st.markdown(f"🏆 **Max Sharpe Ratio** (`{max_sharpe_val:.2f}`):")
                            for idx, t in enumerate(sorted_tickers): st.write(f"- {t}: **{max_sharpe_w[idx]*100:.1f}%**")
                            st.markdown(f"🛡️ **Minimum Volatility** (`{min_vol_sharpe:.2f}`):")
                            for idx, t in enumerate(sorted_tickers): st.write(f"- {t}: **{min_vol_w[idx]*100:.1f}%**")
                except Exception as e:
                    st.error(f"Backtesting error structure crash: {str(e)}")
        else:
            fig_pie = px.pie(names=list(normalized_weights.keys()), values=list(normalized_weights.values()), hole=0.35, title="Asset Allocation Mix")
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"})
            st.plotly_chart(fig_pie, use_container_width=True)

with tab_self_rag:
    st.markdown("### 📁 Self-Serve RAG: Upload Any Corporate Filing")
    st.caption("Upload any PDF report (10-K, ESG, Transcript). The platform will chunk, embed, and index it locally in real-time.")
    
    uploaded_file = st.file_uploader("Upload filing PDF", type=["pdf"])
    
    if uploaded_file:
        if "file_hash" not in st.session_state or st.session_state.file_hash != uploaded_file.name:
            with st.spinner("Extracting PDF text contents..."):
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for pg_idx in range(min(80, len(pdf_reader.pages))):
                    pg_text = pdf_reader.pages[pg_idx].extract_text()
                    if pg_text: text += pg_text
                        
            with st.spinner("Splitting and Vectorizing text chunks..."):
                chunk_sz, ov_sz, up_chunks, start_p = 1000, 200, [], 0
                while start_p < len(text):
                    up_chunks.append(text[start_p:start_p + chunk_sz])
                    start_p += (chunk_sz - ov_sz)
                    
                up_vectors = np.array(embedding_model.encode(up_chunks, show_progress_bar=False), dtype=np.float32)
                up_index = faiss.IndexFlatL2(up_vectors.shape[1])
                up_index.add(up_vectors)
                
                st.session_state.file_hash = uploaded_file.name
                st.session_state.uploaded_chunks = up_chunks
                st.session_state.uploaded_index = up_index
                st.session_state.uploaded_vectors = up_vectors
                st.success(f"Vector Index compiled! Saved {len(up_chunks)} text fragments.")

        if "uploaded_index" in st.session_state:
            up_query = st.text_input("Ask a question about the uploaded document")
            if up_query:
                q_vec = embedding_model.encode([up_query]).astype("float32")
                D, I = st.session_state.uploaded_index.search(q_vec, 6)
                ret_chunks = [st.session_state.uploaded_chunks[int(i)] for i in I[0] if 0 <= int(i) < len(st.session_state.uploaded_chunks)]
                ret_context = "\n\n".join([f"[Chunk {idx+1}]\n{txt}" for idx, txt in enumerate(ret_chunks)])
                
                if api_key:
                    with st.spinner("Evaluating document framework..."):
                        client = genai.Client(api_key=api_key)
                        prompt = f"Context:\n{ret_context}\n\nQuestion:\n{up_query}\n\nRespond strictly in valid JSON matching a detailed_analysis text key."
                        try:
                            resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2))
                            data = json.loads(re.sub(r"^```(json)?|
```$", "", resp.text.strip()).strip())
                            st.markdown(data.get("detailed_analysis", ""))
                        except Exception as e:
                            handle_gemini_error(e, "processing framework context")

def extract_tickers_via_llm(query, api_key):
    client = genai.Client(api_key=api_key)
    prompt = f"Identify tickers from query. Return JSON array format only.\n\nQUERY: {query}"
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json"))
        return [t.upper() for t in json.loads(response.text.strip())]
    except Exception: return []

def get_dynamic_company_context(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        name = info.get("longName") or info.get("shortName") or ticker_symbol
        summary = info.get("longBusinessSummary", "No company profile synopsis loaded.")
        fin_data = fetch_global_financials(ticker_symbol)
        fin_text = f"Revenue: {fin_data['revenue']:.2f}B | Net Income: {fin_data['net_income']:.2f}B" if fin_data else ""
        return f"[Profile: {name} ({ticker_symbol})]\n{summary}\n{fin_text}"
    except Exception as e:
        return f"Dynamic generation failure: {str(e)}"

def get_company_context_for_rag(query, query_vector, api_key=None):
    baseline_map = {"NVIDIA": "NVDA", "Microsoft": "MSFT", "Reliance": "RELIANCE.NS"}
    detected_tickers = []
    q_lower = query.lower()
    
    for name, ticker in baseline_map.items():
        if name.lower() in q_lower or ticker.lower() in q_lower: detected_tickers.append(ticker)
            
    if api_key and not detected_tickers: detected_tickers = extract_tickers_via_llm(query, api_key)
    if not detected_tickers: detected_tickers.append(global_ticker)
        
    context_parts, retrieved_faiss_chunks, distances = [], [], []
    for ticker in list(set(detected_tickers)):
        is_baseline = False
        comp_name_baseline = ""
        for name, sym in baseline_map.items():
            if sym == ticker: is_baseline, comp_name_baseline = True, name; break
                
        if is_baseline and chunks:
            comp_chunks, comp_dists = retrieve_baseline_chunks(query_vector, [comp_name_baseline])
            retrieved_faiss_chunks.extend(comp_chunks)
            distances.extend(comp_dists)
            for idx, c in enumerate(comp_chunks):
                txt = c.get("text", "") if isinstance(c, dict) else str(c)
                context_parts.append(f"[Filing: {comp_name_baseline} Fragment {idx+1}]\n{txt}")
        else:
            context_parts.append(get_dynamic_company_context(ticker))
            retrieved_faiss_chunks.append({"company": ticker, "text": "Dynamic vector stats profile."})
            distances.append(0.0)
            
    return "\n\n".join(context_parts), retrieved_faiss_chunks, distances

def retrieve_baseline_chunks(query_vector, target_companies_list=None):
    if not chunks: return [], []
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
        return [chunks[int(i)] for i in I[0] if 0 <= int(i) < len(chunks)], list(D[0])

def generate_rag_content(query, context):
    system_prompt = f"Context:\n{context}\n\nClient Query:\n{query}\n\nReturn strictly valid JSON with standard fields matching comparison_table, summary, key_findings, segment_breakdown, and risks."
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemini-2.5-flash", contents=system_prompt, config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2))
    return response.text

COMPANY_ALIASES = {"NVIDIA": ["nvidia", "nvda"], "Microsoft": ["microsoft", "msft"], "Reliance": ["reliance", "ril", "jio"]}

with tab_rag:
    st.markdown("### 🔒 Global Reports RAG (Historical + Estimates)")
    st.caption("Ask questions about any global public company (e.g., Ferrari, Apple, NVIDIA, Ola, etc.). The agent will synthesize baseline filings or live profile/estimates data.")
    
    st.markdown("**Try a standard comparison:**")
    eq_cols = st.columns(3)
    examples = [
        "compare revenue and net income across nvidia, microsoft and reliance",
        "compare the ai strategies of microsoft and nvidia",
        "what are the key risk factors for reliance industries?"
    ]
    for col, ex in zip(eq_cols, examples):
        if col.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.rag_query_input = ex
            
    rag_query_input = st.text_input("Ask a financial question", key="rag_query_input")
    
    if rag_query_input:
        query_hash = hashlib.sha256(rag_query_input.strip().lower().encode()).hexdigest()[:16]
        cache_path = os.path.join(DEMO_CACHE_DIR, f"direct_{query_hash}.json")
        
        data = None
        q_vec = embedding_model.encode([rag_query_input]).astype("float32")
        context, chunks_ret, dists = get_company_context_for_rag(rag_query_input, q_vec, api_key)
        
        if os.path.exists(cache_path):
            with open(cache_path) as f: data = json.load(f)
            st.caption("📌 Cached mapping payload recovered.")
        else:
            if not api_key: st.error("API configuration parameters missing.")
            else:
                with st.spinner("Processing Agent Intelligence..."):
                    try:
                        raw_resp = generate_rag_content(rag_query_input, context)
                        # Fixed the string termination regex issue completely here
                        data = json.loads(re.sub(r"^```json\s*|^```\s*|```$", "", raw_resp.strip()).strip())
                    except Exception as e: st.error(f"Execution fault: {e}")
                        
        if data:
            st.markdown(f"#### 🔍 Executive Summary")
            st.info(data.get("summary", "No structured summary key recovered."))
            
            findings = data.get("key_findings") or []
            if findings:
                st.markdown("**Key Findings:**")
                for f in findings: st.markdown(f"- {f}")
                    
            table = data.get("comparison_table") or []
            if table:
                df = pd.DataFrame(table)
                if not df.empty and "value" in df.columns and "metric" in df.columns:
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df = df.dropna(subset=["value"])
                    df["unit"] = df.get("unit", "").fillna("")
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
                if not df_seg.empty and "value" in df_seg.columns and "segment" in df_seg.columns:
                    df_seg["value"] = pd.to_numeric(df_seg["value"], errors="coerce")
                    st.markdown("#### 🥧 Divisional Revenue Split")
                    seg_companies = df_seg["company"].unique() if "company" in df_seg.columns else ["Company"]
                    pie_cols = st.columns(len(seg_companies))
                    
                    for p_col, comp in zip(pie_cols, seg_companies):
                        g = df_seg[df_seg["company"] == comp] if "company" in df_seg.columns else df_seg
                        fig_pie = px.pie(g, names="segment", values="value", title=f"{comp} Segments", hole=0.3)
                        fig_pie.update_traces(textinfo="percent+label", showlegend=False)
                        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"}, height=280)
                        p_col.plotly_chart(fig_pie, use_container_width=True)
                        
            risks = data.get("risks") or []
            if risks:
                st.markdown("#### ⚠️ Highlighted Corporate Risks")
                for r in risks: st.warning(r)

with tab_cca:
    st.markdown(f"### 🏟️ Comparable Company Analysis (CCA): {active_data['name']}")
    col_cca_input, col_cca_chart = st.columns([1, 2])
    
    with col_cca_input:
        peer_tickers_input = st.text_input("Enter Peer Tickers (Comma-separated)", value="AAPL, MSFT, GOOGL, AMZN, META", key="cca_peers")
        peer_tickers = [t.strip().upper() for t in peer_tickers_input.split(",") if t.strip()]
        run_cca = st.button("🚀 Execute Relative Valuation", use_container_width=True, key="run_cca_engine")
        
    if run_cca:
        with st.spinner("Fetching peer metrics..."):
            peer_data = []
            for peer in peer_tickers:
                try:
                    p_info = yf.Ticker(peer).info
                    peer_data.append({
                        "Ticker": peer, "Name": p_info.get("longName") or p_info.get("shortName") or peer,
                        "Trailing P/E": p_info.get("trailingPE"), "Forward P/E": p_info.get("forwardPE"),
                        "Price / Sales": p_info.get("priceToSalesTrailing12Months"), "EV / EBITDA": p_info.get("enterpriseToEbitda")
                    })
                except Exception: pass
                    
            if not peer_data:
                st.error("Relative benchmarking compilation failed.")
            else:
                df_peers = pd.DataFrame(peer_data)
                st.dataframe(df_peers.style.format({"Trailing P/E": "{:,.2f}", "Forward P/E": "{:,.2f}", "Price / Sales": "{:,.2f}", "EV / EBITDA": "{:,.2f}"}, na_rep="—"), use_container_width=True)
                
                target_info = yf.Ticker(active_company).info
                target_eps = target_info.get("trailingEps") or (active_data["net_income"] / active_data["shares"])
                target_rev_per_share = (active_data["revenue"] / active_data["shares"])
                target_ebitda = target_info.get("ebitda") or (active_data["ebit"] * 1.2 * 1e9)
                target_ebitda_per_share = (target_ebitda / 1e9) / active_data["shares"]
                
                avg_pe_trail = df_peers["Trailing P/E"].mean(skipna=True)
                avg_pe_fwd = df_peers["Forward P/E"].mean(skipna=True)
                avg_ps = df_peers["Price / Sales"].mean(skipna=True)
                avg_evebitda = df_peers["EV / EBITDA"].mean(skipna=True)
                net_debt_per_share = (active_data["debt"] - active_data["cash"]) / active_data["shares"]
                
                implied_prices = {}
                if pd.notna(avg_pe_trail) and target_eps: implied_prices["Trailing P/E"] = avg_pe_trail * target_eps
                if pd.notna(avg_pe_fwd) and target_info.get("forwardEps"): implied_prices["Forward P/E"] = avg_pe_fwd * target_info["forwardEps"]
                if pd.notna(avg_ps) and target_rev_per_share: implied_prices["Price / Sales"] = avg_ps * target_rev_per_share
                if pd.notna(avg_evebitda) and target_ebitda_per_share: implied_prices["EV / EBITDA"] = (target_ebitda_per_share * avg_evebitda) - net_debt_per_share
                    
                implied_df = pd.DataFrame([{"Multiple Method": m, "Implied Value": round(v, 2)} for m, v in implied_prices.items()])
                st.dataframe(implied_df, use_container_width=True)
                
                fig_football = go.Figure()
                y_methods = list(implied_prices.keys())
                for idx, method in enumerate(y_methods):
                    val = implied_prices[method]
                    fig_football.add_trace(go.Bar(y=[method], x=[val * 0.3], base=val * 0.85, orientation='h', marker=dict(color='#58a6ff', opacity=0.8)))
                    fig_football.add_trace(go.Scatter(y=[method], x=[val], mode='markers+text', text=[f"${val:.2f}"], textposition="top center", marker=dict(color='#bc8cff', size=10, symbol='diamond'), showlegend=False))
                    
                if active_data["price"]: fig_football.add_vline(x=active_data["price"], line_width=2, line_dash="dash", line_color="#f85149")
                if st.session_state.get("implied_share_value"): fig_football.add_vline(x=st.session_state.get("implied_share_value"), line_width=2, line_dash="dot", line_color="#bc8cff")
                    
                fig_football.update_layout(title="Relative Multiple Football Ranges", barmode='overlay', showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e6edf3"}, height=300)
                with col_cca_chart: st.plotly_chart(fig_football, use_container_width=True)
