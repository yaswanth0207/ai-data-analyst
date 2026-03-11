"""
PDF report generation: dataset summary, Q&A history, charts, insights. Uses reportlab.
"""
import io
import os
import tempfile
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Image, PageBreak


def _add_section(doc_parts: list, title: str, content_flow: list) -> None:
    doc_parts.append(Paragraph(f"<b>{title}</b>", ParagraphStyle(name="Heading", fontSize=14, spaceAfter=6)))
    for item in content_flow:
        doc_parts.append(item)
    doc_parts.append(Spacer(1, 0.2 * inch))


def generate_report_pdf(session_id: str, session: dict[str, Any], charts: dict[str, Any]) -> str:
    """
    Generate a PDF report for the session. Returns path to the saved PDF file.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("AI Data Analyst - Session Report", ParagraphStyle(name="Title", fontSize=18, spaceAfter=12)))
    story.append(Paragraph(f"Session ID: {session_id}", styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    # Dataset summary
    schema = session.get("schema") or {}
    summary_data = [
        ["Rows", str(schema.get("row_count", "—"))],
        ["Columns", str(schema.get("column_count", "—"))],
        ["Column names", ", ".join(schema.get("columns", [])[:15]) + ("..." if len(schema.get("columns", [])) > 15 else "")],
    ]
    t = Table(summary_data, colWidths=[1.5 * inch, 4 * inch])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story.append(Paragraph("<b>Dataset Summary</b>", ParagraphStyle(name="H2", fontSize=12, spaceAfter=6)))
    story.append(t)
    story.append(Spacer(1, 0.25 * inch))

    # Key insights
    insights = session.get("insights") or []
    if insights:
        story.append(Paragraph("<b>Key Insights</b>", ParagraphStyle(name="H2", fontSize=12, spaceAfter=6)))
        for i, ins in enumerate(insights[:10], 1):
            story.append(Paragraph(f"{i}. {ins[:500]}", styles["Normal"]))
        story.append(Spacer(1, 0.25 * inch))

    # Q&A history
    history = session.get("history") or []
    story.append(Paragraph("<b>Questions & Answers</b>", ParagraphStyle(name="H2", fontSize=12, spaceAfter=6)))
    for i, h in enumerate(history, 1):
        story.append(Paragraph(f"<b>Q{i}:</b> {str(h.get('question', ''))[:400]}", styles["Normal"]))
        story.append(Paragraph(f"<b>A:</b> {str(h.get('answer', ''))[:800]}", styles["Normal"]))
        if h.get("code"):
            story.append(Paragraph(f"<i>Code:</i> <font size=7>{str(h.get('code', ''))[:600]}...</font>", styles["Normal"]))
        story.append(Spacer(1, 0.15 * inch))
    story.append(Spacer(1, 0.3 * inch))

    # Charts (placeholders or note – reportlab can't render plotly directly; we add a note)
    chart_ids = [h.get("chart_id") for h in history if h.get("chart_id")]
    if chart_ids:
        story.append(Paragraph("<b>Charts Generated</b>", ParagraphStyle(name="H2", fontSize=12, spaceAfter=6)))
        story.append(Paragraph(f"This session generated {len(chart_ids)} chart(s). View them in the app.", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    path = os.path.join(tempfile.gettempdir(), f"analyst_report_{session_id[:8]}.pdf")
    with open(path, "wb") as f:
        f.write(buffer.getvalue())
    return path
