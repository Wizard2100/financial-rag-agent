import math

import financials as fin


def test_companies_list_matches_loaded_data():
    assert set(fin.companies()) == {"NVIDIA", "Microsoft", "Reliance"}


def test_operating_margin_is_correct_for_nvidia():
    # 81.45 / 130.5 = 62.4%
    assert fin.operating_margin("NVIDIA") == 62.4


def test_operating_margin_is_none_when_source_doesnt_report_it():
    # Reliance's report doesn't break out a comparable operating income line
    assert fin.operating_margin("Reliance") is None


def test_net_margin_is_correct_for_microsoft():
    # 101.8 / 281.7 = 36.1%
    assert fin.net_margin("Microsoft") == 36.1


def test_ratio_dashboard_has_one_row_per_company():
    df = fin.ratio_dashboard()
    assert len(df) == len(fin.companies())
    assert set(df["Company"]) == set(fin.companies())


def test_get_trend_returns_only_populated_years():
    # FY24/FY23 are intentionally None until real data is added - the
    # trend function must never invent a value for them
    trend = fin.get_trend("NVIDIA", "Revenue")
    assert trend == {"FY25": 130.5}


def test_narrative_mentions_the_actual_revenue_figure():
    text = fin.generate_narrative("NVIDIA")
    assert "130.5" in text
    assert "NVIDIA" in text


def test_narrative_handles_missing_operating_income_gracefully():
    text = fin.generate_narrative("Reliance")
    assert "EBITDA" in text  # falls back to EBITDA since op income is None
    assert "None" not in text  # never leak a raw None into user-facing text


def test_comparison_narrative_ranks_correctly():
    text = fin.generate_comparison_narrative("Revenue")
    # Microsoft has the highest FY25 revenue of the three loaded companies
    assert text.startswith("On Revenue (FY25), Microsoft leads")


def test_comparison_narrative_handles_unknown_metric():
    text = fin.generate_comparison_narrative("Nonexistent Metric")
    assert "No verified" in text
