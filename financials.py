"""
financials.py

Everything in this module is deterministic: plain arithmetic on hand-verified
annual-report figures, plus rule-based sentence templates. Nothing in here
calls an LLM, so nothing in here can hallucinate a number or fail on a quota.

This is the "Verified-Only" analysis mode - use it as the default/fallback
path so the app degrades gracefully instead of erroring out when the Gemini
quota is exhausted, and use it as the primary path whenever you specifically
want to demo something that cannot fail.

VERIFIED_FINANCIALS values are hand-checked against the actual uploaded
annual-report text (see the `source` field on each company). When you add a
new fiscal year, add a new entry keyed by year so YoY trend charts work -
see `get_trend` below.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Verified financials, keyed by company -> fiscal year -> figures.
# FY25 figures are carried over from the existing app. FY23/FY24 are left
# as None on purpose - filling them in from the actual FY23/FY24 annual
# reports is the single highest-value next step (see the roadmap doc,
# Priority 1). Do not estimate these numbers; leave them None until you have
# the real figure, the dashboard below already handles missing years.
# ---------------------------------------------------------------------------

VERIFIED_FINANCIALS = {
    "NVIDIA": {
        "FY25": {
            "fiscal_year_end": "January 26, 2025",
            "source": "NVIDIA FY2025 Form 10-K (filed Feb 26, 2025)",
            "Revenue": 130.5,
            "Operating Income": 81.45,
            "Net Income": 72.88,
            "Revenue Growth %": 114.0,
            "extra": {"R&D Expense ($B)": 12.91, "Gross Margin (%)": 75.0},
            "quote": "Revenue more than doubled to $130.5 billion, up 114% year-over-year. "
                     "Operating income rose 147% to $81.5 billion. Net income $72,880 million, up 145%.",
        },
        "FY24": None,
        "FY23": None,
    },
    "Microsoft": {
        "FY25": {
            "fiscal_year_end": "June 30, 2025",
            "source": "Microsoft FY2025 Annual Report",
            "Revenue": 281.7,
            "Operating Income": 128.5,
            "Net Income": 101.8,
            "Revenue Growth %": 15.0,
            "extra": {"Azure Revenue ($B, min)": 75.0, "Microsoft Cloud Revenue ($B)": 168.9},
            "quote": "Revenue was $281.7 billion, up 15 percent. Operating income grew 17 percent to "
                     "$128.5 billion. Azure surpassed $75 billion in revenue for the first time, up 34 percent.",
        },
        "FY24": None,
        "FY23": None,
    },
    "Reliance": {
        "FY25": {
            "fiscal_year_end": "March 31, 2025",
            "source": "Reliance Industries Integrated Annual Report 2024-25",
            "Revenue": 125.3,
            "Operating Income": None,
            "Net Income": 9.5,
            "Revenue Growth %": 7.1,
            "extra": {"EBITDA ($B)": 21.5, "Revenue (INR Crore)": 1071174, "Net Income / PAT (INR Crore)": 81309},
            "quote": "Consolidated revenue increased by 7.1% to Rs 10,71,174 crore (US$125.3 billion). "
                     "EBITDA grew 2.9% to Rs 1,83,422 crore (US$21.5 billion). PAT rose 2.9% to "
                     "Rs 81,309 crore (US$9.5 billion).",
            "currency_note": "Reliance reports in INR Crore; USD figures above are as converted in the source report.",
        },
        "FY24": None,
        "FY23": None,
    },
}

LATEST_YEAR = "FY25"


def companies() -> list[str]:
    return sorted(VERIFIED_FINANCIALS.keys())


def get_year(company: str, year: str = LATEST_YEAR) -> dict | None:
    return VERIFIED_FINANCIALS.get(company, {}).get(year)


def get_trend(company: str, metric: str) -> dict[str, float]:
    """Returns {year: value} for whichever years actually have data for this
    metric - used to draw a trend line once FY23/FY24 are filled in. With
    only FY25 populated, this correctly returns a single point rather than
    inventing the missing years."""
    out = {}
    for year, data in VERIFIED_FINANCIALS.get(company, {}).items():
        if data and data.get(metric) is not None:
            out[year] = data[metric]
    return out


# ---------------------------------------------------------------------------
# Ratio engine - plain arithmetic on the verified figures above. Returns None
# (never a fabricated number) whenever an input is missing.
# ---------------------------------------------------------------------------

def operating_margin(company: str, year: str = LATEST_YEAR) -> float | None:
    data = get_year(company, year)
    if not data or not data.get("Revenue") or data.get("Operating Income") is None:
        return None
    return round(100 * data["Operating Income"] / data["Revenue"], 1)


def net_margin(company: str, year: str = LATEST_YEAR) -> float | None:
    data = get_year(company, year)
    if not data or not data.get("Revenue") or data.get("Net Income") is None:
        return None
    return round(100 * data["Net Income"] / data["Revenue"], 1)


def gross_margin(company: str, year: str = LATEST_YEAR) -> float | None:
    data = get_year(company, year)
    if not data:
        return None
    return data.get("extra", {}).get("Gross Margin (%)")


def revenue_growth(company: str, year: str = LATEST_YEAR) -> float | None:
    data = get_year(company, year)
    return data.get("Revenue Growth %") if data else None


def ratio_dashboard(year: str = LATEST_YEAR) -> "pd.DataFrame":
    """One row per company: every verified-data ratio in one table. Live,
    market-based ratios (P/E, ROE, D/E, etc.) live in market_tools.py instead,
    since those come from Yahoo Finance, not the annual report."""
    import pandas as pd

    rows = []
    for c in companies():
        data = get_year(c, year)
        rows.append({
            "Company": c,
            "Revenue ($B)": data.get("Revenue") if data else None,
            "Revenue Growth %": revenue_growth(c, year),
            "Operating Margin %": operating_margin(c, year),
            "Net Margin %": net_margin(c, year),
            "Gross Margin %": gross_margin(c, year),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Template-based narrative generator. This is the "looks like AI, isn't an
# LLM call" piece: every sentence is filled in from a verified number, with
# the wording chosen by simple if/else rules. No model, no API, no quota.
# ---------------------------------------------------------------------------

def _trend_word(value: float | None, positive_is_good: bool = True) -> str:
    if value is None:
        return "changed"
    if value > 0:
        return "grew" if positive_is_good else "worsened"
    if value < 0:
        return "declined" if positive_is_good else "improved"
    return "held steady"


def generate_narrative(company: str, year: str = LATEST_YEAR) -> str:
    data = get_year(company, year)
    if not data:
        return f"No verified {year} figures are loaded for {company} yet."

    growth = data.get("Revenue Growth %")
    op_margin = operating_margin(company, year)
    net_marg = net_margin(company, year)

    sentences = []

    if data.get("Revenue") is not None:
        growth_clause = f", {_trend_word(growth)} {abs(growth):.1f}% year-over-year" if growth is not None else ""
        sentences.append(f"{company} reported revenue of ${data['Revenue']:.1f}B in {year}{growth_clause}.")

    if op_margin is not None:
        sentences.append(f"Operating margin came in at {op_margin:.1f}% of revenue.")
    elif data.get("extra", {}).get("EBITDA ($B)") is not None:
        sentences.append(
            f"{company} doesn't break out a directly comparable operating-income line; "
            f"EBITDA was ${data['extra']['EBITDA ($B)']:.1f}B."
        )

    if net_marg is not None:
        sentences.append(f"Net income margin was {net_marg:.1f}%.")

    if data.get("currency_note"):
        sentences.append(data["currency_note"])

    return " ".join(sentences)


def generate_comparison_narrative(metric: str = "Revenue", year: str = LATEST_YEAR) -> str:
    """Ranks every loaded company on one verified metric and writes a plain
    comparison sentence - the no-API substitute for the LLM comparison tab."""
    rows = [(c, get_year(c, year) and get_year(c, year).get(metric)) for c in companies()]
    rows = [(c, v) for c, v in rows if v is not None]
    if not rows:
        return f"No verified {year} data is available for '{metric}'."

    rows.sort(key=lambda r: r[1], reverse=True)
    leader, leader_val = rows[0]

    parts = [f"On {metric} ({year}), {leader} leads at {leader_val:,.1f}."]
    for c, v in rows[1:]:
        gap = leader_val - v
        parts.append(f"{c} follows at {v:,.1f}, {gap:,.1f} behind {leader}.")

    return " ".join(parts)
