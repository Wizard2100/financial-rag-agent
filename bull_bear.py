"""
bull_bear.py

A rule-based "Bull Case / Bear Case" generator. This is deliberately built
to *look* like the kind of synthesis an LLM would produce, while actually
being a fixed set of threshold checks against real numbers - every bullet
point traces back to one verified or live figure, so there's nothing here
that can hallucinate or drift between runs.

Design note for write-ups/interviews: this is the "explainable AI" angle on
the project - the insight layer is auditable by construction, which is a
deliberately different trade-off than asking an LLM to free-write an
opinion.
"""

from __future__ import annotations

import financials as fin
import market_tools as mkt


def _fmt_pct(x):
    return f"{x:.1f}%" if x is not None else "n/a"


def build_case(company: str, ticker_symbol: str) -> dict:
    """Returns {"bull": [str, ...], "bear": [str, ...], "live": dict, "error": str|None}.
    Live-data points are skipped gracefully (not fabricated) if Yahoo Finance
    doesn't have a field for this ticker."""

    verified = fin.get_year(company)
    live = mkt.get_live_snapshot(ticker_symbol)

    bull, bear = [], []

    growth = fin.revenue_growth(company)
    if growth is not None:
        if growth >= 15:
            bull.append(f"Revenue grew {growth:.1f}% YoY in {fin.LATEST_YEAR} — well above typical large-cap growth.")
        elif growth < 5:
            bear.append(f"Revenue growth has slowed to {growth:.1f}% YoY in {fin.LATEST_YEAR}.")

    op_margin = fin.operating_margin(company)
    if op_margin is not None:
        if op_margin >= 30:
            bull.append(f"Operating margin of {op_margin:.1f}% indicates strong pricing power / cost discipline.")
        elif op_margin < 10:
            bear.append(f"Operating margin of only {op_margin:.1f}% leaves little cushion against cost shocks.")
    elif verified:
        bear.append(
            f"{company}'s report doesn't break out a directly comparable operating-income line, "
            "which makes margin comparisons against peers harder."
        )

    net_margin = fin.net_margin(company)
    if net_margin is not None and net_margin < 10:
        bear.append(f"Net margin of {net_margin:.1f}% is thin relative to the large-cap peer set.")

    if "error" not in live:
        pe = live.get("trailing_pe")
        if pe is not None:
            if pe > 50:
                bear.append(f"Trailing P/E of {pe:.1f}x is rich, leaving the stock more exposed to multiple compression "
                            "if growth disappoints.")
            elif pe < 20:
                bull.append(f"Trailing P/E of {pe:.1f}x is modest for a company of this profile.")

        d2e = live.get("debt_to_equity")
        if d2e is not None:
            if d2e > 100:
                bear.append(f"Debt-to-equity of {d2e:.0f}% signals meaningfully leveraged balance sheet.")
            elif d2e < 30:
                bull.append(f"Debt-to-equity of {d2e:.0f}% indicates a conservative balance sheet.")

        total_debt = live.get("total_debt") or 0
        total_cash = live.get("total_cash") or 0
        if total_cash > total_debt and (total_debt or total_cash):
            net_cash = total_cash - total_debt
            bull.append(f"Net cash position of roughly ${net_cash / 1e9:,.1f}B reduces balance-sheet risk.")

        fcf = live.get("free_cashflow")
        if fcf is not None and fcf > 0:
            bull.append(f"Free cash flow of roughly ${fcf / 1e9:,.1f}B funds buybacks/dividends/reinvestment without "
                        "needing external financing.")

        roe = live.get("return_on_equity")
        if roe is not None and roe > 0.30:
            bull.append(f"Return on equity of {_fmt_pct(roe * 100)} is well above typical large-cap returns.")

        price = live.get("current_price")
        high_52w = live.get("fifty_two_week_high")
        if price and high_52w and high_52w > 0:
            pct_off_high = (1 - price / high_52w) * 100
            if pct_off_high > 20:
                bear.append(f"Trading {pct_off_high:.0f}% below its 52-week high.")

    if not bull:
        bull.append("No verified figure cleared the bull-case thresholds used by this generator.")
    if not bear:
        bear.append("No verified figure cleared the bear-case thresholds used by this generator.")

    return {"bull": bull, "bear": bear, "live": live if "error" not in live else None, "error": live.get("error")}
