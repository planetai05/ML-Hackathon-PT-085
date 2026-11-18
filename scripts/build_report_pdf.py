"""Generate consolidated PathSnap Technical Report PDF from reports/report_skeleton.md.

Produces: reports/PathSnap_Technical_Report.pdf
"""
from __future__ import annotations

import os
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "PathSnap: Distributed Map-Matching & Sequential Route Reconstruction — Team PT-085")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 750, 558, 750)

        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, footer_text)
        self.drawString(54, 36, "Confidential — Evaluator Technical Report — November 2026")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)

        self.restoreState()


def clean_markdown_inline(text: str) -> str:
    """Format markdown bold, code, and math symbols for ReportLab XML."""
    # Escape XML entities first
    text = text.replace("&", "&amp;")
    # Convert bold **text** -> <b>text</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Convert inline code `text` -> <font name="Courier" color="#0369a1">\1</font>
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" color="#0369a1">\1</font>', text)
    # Clean latex math $...$ -> text
    text = re.sub(r"\$([^\$]+)\$", r"<i>\1</i>", text)
    return text


def build_pdf():
    md_path = "reports/report_skeleton.md"
    pdf_path = "reports/PathSnap_Technical_Report.pdf"

    if not os.path.exists(md_path):
        raise FileNotFoundError(f"{md_path} not found")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom typography
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0369a1"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )
    h3_style = ParagraphStyle(
        "Heading3_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3,
    )
    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
        backColor=colors.HexColor("#f1f5f9"),
        borderColor=colors.HexColor("#cbd5e1"),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=8,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0f172a"),
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
    )

    story = []

    lines = md_text.split("\n")
    i = 0
    in_code = False
    code_buffer = []

    while i < len(lines):
        line = lines[i].rstrip()

        # Handle code blocks
        if line.startswith("```"):
            if in_code:
                in_code = False
                story.append(Paragraph("<br/>".join(code_buffer), code_style))
                code_buffer = []
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_buffer.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;"))
            i += 1
            continue

        # Skip horizontal rules
        if line.strip() == "---":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=8))
            i += 1
            continue

        # Skip empty lines
        if not line.strip():
            i += 1
            continue

        # Markdown tables
        if line.strip().startswith("|") and line.strip().endswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_lines.append(lines[i].strip())
                i += 1

            if len(table_lines) >= 3:
                header_raw = [c.strip() for c in table_lines[0].split("|")[1:-1]]
                data_rows = []
                for row_str in table_lines[2:]:  # skip separator row
                    cells = [c.strip() for c in row_str.split("|")[1:-1]]
                    data_rows.append(cells)

                # Convert to Paragraphs
                table_data = []
                table_data.append([Paragraph(clean_markdown_inline(h), table_header) for h in header_raw])
                for r in data_rows:
                    table_data.append([Paragraph(clean_markdown_inline(c), table_cell) for c in r])

                num_cols = len(header_raw)
                total_w = 504
                col_w = total_w / num_cols

                t = Table(table_data, colWidths=[col_w] * num_cols, repeatRows=1)
                t.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ])
                )
                story.append(Spacer(1, 4))
                story.append(t)
                story.append(Spacer(1, 6))
            continue

        # Document Title
        if line.startswith("# "):
            story.append(Paragraph(clean_markdown_inline(line[2:]), title_style))
            i += 1
            continue

        # Document Subtitle
        if line.startswith("**") and line.endswith("**") and i < 5:
            story.append(Paragraph(clean_markdown_inline(line[2:-2]), subtitle_style))
            i += 1
            continue

        # Headings
        if line.startswith("## "):
            story.append(Paragraph(clean_markdown_inline(line[3:]), h1_style))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(clean_markdown_inline(line[4:]), h2_style))
            i += 1
            continue
        if line.startswith("#### "):
            story.append(Paragraph(clean_markdown_inline(line[5:]), h3_style))
            i += 1
            continue

        # Bullets
        if line.startswith("- ") or line.startswith("* "):
            story.append(Paragraph("• " + clean_markdown_inline(line[2:]), bullet_style))
            i += 1
            continue

        # Numbered lists (e.g. 1. text)
        m_num = re.match(r"^(\d+)\.\s+(.+)$", line)
        if m_num:
            story.append(Paragraph(f"<b>{m_num.group(1)}.</b> {clean_markdown_inline(m_num.group(2))}", bullet_style))
            i += 1
            continue

        # Normal text paragraph
        story.append(Paragraph(clean_markdown_inline(line), body_style))
        i += 1

    # Append key figures at the end of the report
    story.append(PageBreak())
    story.append(Paragraph("10. Supplementary Report Figures & Visualizations", h1_style))
    story.append(Paragraph("The publication figures below visualize model comparisons, HPO trials, HMM continuity impact, and Streamlit user interface operations:", body_style))

    figs = [
        ("Figure 1: Cross-Validation & Student-Testing Accuracy Across 7 Models", "results/figures/report/model_comparison_accuracy.png", 460, 230),
        ("Figure 2: Log Loss Probability Quality Comparison", "results/figures/report/model_comparison_logloss.png", 460, 230),
        ("Figure 3: Computational Efficiency (Training Time vs Inference Latency)", "results/figures/report/model_efficiency.png", 460, 240),
        ("Figure 4: Bayesian HPO Optimization History & Parameter Importance", "results/figures/hpo/optimization_history.png", 460, 230),
        ("Figure 5: HMM Sequence Impact on Top-1 Accuracy and Route Continuity", "results/figures/report/hmm_route_impact.png", 460, 280),
        ("Figure 6: Streamlit Evaluator Interface Connected to Real XGBoost Model", "docs/screenshots/01_ui_connected_real_model.jpg", 460, 240),
        ("Figure 7: Reconstructed Route and Folium Map for Trip 00000056", "docs/screenshots/04_ui_result_metrics_map_trip56.jpg", 460, 240),
        ("Figure 8: Detailed Pipeline Latency and Stage Timings", "docs/screenshots/06_ui_details_and_timings.jpg", 460, 240),
    ]

    for title, fpath, w, h in figs:
        if os.path.exists(fpath):
            story.append(KeepTogether([
                Spacer(1, 10),
                Paragraph(f"<b>{title}</b>", h3_style),
                Image(fpath, width=w, height=h),
                Spacer(1, 8),
            ]))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report PDF built successfully: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")


if __name__ == "__main__":
    build_pdf()
