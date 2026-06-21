"""
export_utils.py

Turns whatever's currently on screen into a file the user can actually keep -
a tangible export matters more to a recruiter clicking around than one more
chart, since a live Streamlit demo link can be asleep or down by the time
someone looks at it.

Both functions return raw bytes, ready for st.download_button(data=...).
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def export_dataframes_to_excel(sheets: dict[str, pd.DataFrame]) -> bytes:
    """sheets: {sheet_name: dataframe}. One sheet per entry, sheet names
    truncated to Excel's 31-character limit."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if df is None or df.empty:
                continue
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return buffer.getvalue()


def export_report_to_pdf(title: str, sections: list[dict]) -> bytes:
    """sections: list of {"heading": str, "text": str | None, "table": pd.DataFrame | None}.
    Renders a simple, readable one-column report - headings, paragraphs, and
    tables in order, no styling rabbit holes."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    flow = [Paragraph(title, styles["Title"]), Spacer(1, 16)]

    for section in sections:
        if section.get("heading"):
            flow.append(Paragraph(section["heading"], styles["Heading2"]))
            flow.append(Spacer(1, 6))

        if section.get("text"):
            flow.append(Paragraph(section["text"], styles["BodyText"]))
            flow.append(Spacer(1, 10))

        table_df = section.get("table")
        if table_df is not None and not table_df.empty:
            data = [list(table_df.columns)] + table_df.astype(str).values.tolist()
            table = Table(data, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
            ]))
            flow.append(table)
            flow.append(Spacer(1, 14))

    doc.build(flow)
    return buffer.getvalue()
