"""
market_tools.py

Everything here is "live, real, no API key needed" - it all goes through
yfinance, which scrapes/queries Yahoo Finance's public endpoints for free.
This is a different kind of "without API" than financials.py: financials.py
is static and verified, this module is live and market-sourced. Neither one
calls an LLM.

IMPORTANT: the DCF calculator is an educational valuation exercise with
user-adjustable assumptions (WACC, growth rate). It is not investment
advice, and the app should say so wherever it's displayed - intrinsic value
is only as good as the assumptions you put in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def get_live_snapshot(ticker_symbol: str) -> dict:
    """Pulls the fields needed for the ratio dashboard and comps table.
    Falls back to fast_info if the full .info call fails (Yahoo occasionally
    rate-limits or changes its response shape), matching the resilience
    pattern already used in the agent's get_live_market_data tool."""
    try:
        info = yf.Ticker(ticker_symbol).info
        return {
            "ticker": ticker_symbol,
            "currency": info.get("currency"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "enterprise_value": info.get("enterpriseValue"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "ev_to_revenue": info.get("enterpriseToRevenue"),
            "return_on_equity": info.get("returnOnEquity"),
            "return_on_assets": info.get("returnOnAssets"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "free_cashflow": info.get("freeCashflow"),
            "total_debt": info.get("totalDebt"),
            "total_cash": info.get("totalCash"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        try:
            fast = yf.Ticker(ticker_symbol).fast_info
            return {
                "ticker": ticker_symbol,
                "currency": fast.get("currency"),
                "current_price": fast.get("last_price"),
                "market_cap": fast.get("market_cap"),
                "fifty_two_week_high": fast.get("year_high"),
                "fifty_two_week_low": fast.get("year_low"),
            }
        except Exception as e:
            return {"error": f"Live market data unavailable right now: {e}"}


def build_comps_table(tickers: dict[str, str]) -> pd.DataFrame:
    """tickers: {display_name: ticker_symbol}. Returns one row per company
    with the standard comps-screen multiples, all pulled live."""
    rows = []
    for name, symbol in tickers.items():
        snap = get_live_snapshot(symbol)
        if "error" in snap:
            rows.append({"Company": name, "Ticker": symbol, "Error": snap["error"]})
            continue
        rows.append({
            "Company": name,
            "Ticker": symbol,
            "Price": snap.get("current_price"),
            "Market Cap": snap.get("market_cap"),
            "Trailing P/E": snap.get("trailing_pe"),
            "Forward P/E": snap.get("forward_pe"),
            "P/B": snap.get("price_to_book"),
            "EV/EBITDA": snap.get("ev_to_ebitda"),
            "EV/Revenue": snap.get("ev_to_revenue"),
            "ROE %": (snap.get("return_on_equity") or 0) * 100 if snap.get("return_on_equity") is not None else None,
            "Net Margin %": (snap.get("profit_margin") or 0) * 100 if snap.get("profit_margin") is not None else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DCF - simple two-stage (explicit projection + Gordon-growth terminal
# value), driven by the last reported free cash flow from yfinance.
# ---------------------------------------------------------------------------

def run_dcf(ticker_symbol: str, wacc: float, terminal_growth: float,
            fcf_growth_rate: float, projection_years: int = 5) -> dict:
    """All rates are plain decimals (0.10 = 10%). Returns the projected FCF
    path, enterprise value, equity value, and per-share intrinsic value -
    or an 'error' key if the underlying data isn't available for this ticker."""

    if wacc <= terminal_growth:
        return {"error": "WACC must be greater than the terminal growth rate, or the terminal value diverges."}

    snap = get_live_snapshot(ticker_symbol)
    if "error" in snap:
        return snap

    base_fcf = snap.get("free_cashflow")
    if base_fcf is None or base_fcf <= 0:
        return {"error": f"No usable free cash flow figure from Yahoo Finance for {ticker_symbol}."}

    shares = snap.get("shares_outstanding")
    if not shares:
        return {"error": f"No shares-outstanding figure from Yahoo Finance for {ticker_symbol}."}

    net_debt = (snap.get("total_debt") or 0) - (snap.get("total_cash") or 0)

    projected = []
    fcf = base_fcf
    for year in range(1, projection_years + 1):
        fcf = fcf * (1 + fcf_growth_rate)
        discount_factor = (1 + wacc) ** year
        projected.append({
            "year": year,
            "fcf": fcf,
            "discount_factor": discount_factor,
            "pv_fcf": fcf / discount_factor,
        })

    terminal_value = projected[-1]["fcf"] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / projected[-1]["discount_factor"]

    enterprise_value = sum(p["pv_fcf"] for p in projected) + pv_terminal
    equity_value = enterprise_value - net_debt
    intrinsic_value_per_share = equity_value / shares

    return {
        "ticker": ticker_symbol,
        "base_fcf": base_fcf,
        "assumptions": {
            "wacc": wacc,
            "terminal_growth": terminal_growth,
            "fcf_growth_rate": fcf_growth_rate,
            "projection_years": projection_years,
        },
        "projected_fcf": projected,
        "terminal_value": terminal_value,
        "pv_terminal_value": pv_terminal,
        "enterprise_value": enterprise_value,
        "net_debt": net_debt,
        "equity_value": equity_value,
        "intrinsic_value_per_share": intrinsic_value_per_share,
        "current_price": snap.get("current_price"),
    }


def dcf_sensitivity_grid(ticker_symbol: str, wacc_range: list[float],
                          growth_range: list[float], fcf_growth_rate: float,
                          projection_years: int = 5) -> pd.DataFrame:
    """Intrinsic value per share for every (WACC, terminal growth) pair -
    feed this straight into a heatmap."""
    grid = pd.DataFrame(index=[f"{w:.1%}" for w in wacc_range],
                         columns=[f"{g:.1%}" for g in growth_range], dtype=float)

    for w in wacc_range:
        for g in growth_range:
            if w <= g:
                continue
            result = run_dcf(ticker_symbol, w, g, fcf_growth_rate, projection_years)
            if "error" not in result:
                grid.loc[f"{w:.1%}", f"{g:.1%}"] = result["intrinsic_value_per_share"]

    return grid


# ---------------------------------------------------------------------------
# Price history - candlestick + cross-company return correlation
# ---------------------------------------------------------------------------

def get_price_history(tickers: dict[str, str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """tickers: {display_name: ticker_symbol}. Returns {display_name: OHLCV df}."""
    histories = {}
    for name, symbol in tickers.items():
        try:
            hist = yf.Ticker(symbol).history(period=period)
            if not hist.empty:
                histories[name] = hist
        except Exception:
            continue
    return histories


def compute_correlation_matrix(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily-return correlation across companies, from the same history
    dict returned by get_price_history."""
    returns = {}
    for name, hist in histories.items():
        if "Close" in hist.columns and len(hist) > 1:
            returns[name] = hist["Close"].pct_change().dropna()

    if len(returns) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(returns).dropna()
    return df.corr()
