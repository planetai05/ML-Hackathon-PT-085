"""Build the PathSnap technical report PDF from reports/PathSnap_Report.md.

Produces: reports/PathSnap_Report_PT-085.pdf
Images are written in the markdown as ![caption](path) and placed inline, keeping their aspect ratio.
"""
from __future__ import annotations

import os
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
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
            self.drawString(54, 755, "PathSnap Technical Report — Team PT-085")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 750, 558, 750)

        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, footer_text)
        self.drawString(54, 36, "PathSnap — Team PT-085 — November 2026")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)

        self.restoreState()


def clean_markdown_inline(text: str) -> str:
    """Format markdown bold, code, and math symbols for ReportLab XML."""
    # Escape XML entities first, and map glyphs missing from the base PDF fonts
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("→", "-&gt;").replace("≈", "~").replace("≤", "&lt;=").replace("≥", "&gt;=")
    # Inline code first, so * and _ inside code are left alone
    codes = []

    def _code(m):
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", _code, text)
    # Convert bold **text** -> <b>text</b>, italic *text* -> <i>text</i>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: f'<font name="Courier" color="#0369a1">{codes[int(m.group(1))]}</font>', text)
    return text


def build_pdf():
    md_path = "reports/PathSnap_Report.md"
    pdf_path = "reports/PathSnap_Report_PT-085.pdf"

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

    caption_style = ParagraphStyle(
        "Caption_Custom", parent=body_style, fontSize=8, leading=10.5,
        textColor=colors.HexColor("#475569"), spaceBefore=3, spaceAfter=10,
    )
    sub_bullet_style = ParagraphStyle(
        "SubBullet_Custom", parent=bullet_style, leftIndent=28, firstLineIndent=-8,
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
            code_buffer.append(line.strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;"))
            i += 1
            continue

        # Skip horizontal rules
        if line.strip() == "---":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=8))
            i += 1
            continue

        # Inline images: ![caption](path)
        m_img = re.match(r"^!\[(.+)\]\((.+)\)$", line.strip())
        if m_img:
            caption, fpath = m_img.group(1), m_img.group(2)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"figure not found: {fpath}")
            iw, ih = ImageReader(fpath).getSize()
            scale = min(470 / iw, 330 / ih)
            story.append(KeepTogether([
                Spacer(1, 6),
                Image(fpath, width=iw * scale, height=ih * scale),
                Paragraph(f"<i>{clean_markdown_inline(caption)}</i>", caption_style),
            ]))
            i += 1
            continue

        # Indented sub-bullets / continuation lines
        if line.startswith("  ") and line.strip():
            sub = line.strip()
            if sub.startswith("- "):
                story.append(Paragraph("– " + clean_markdown_inline(sub[2:]), sub_bullet_style))
            else:
                story.append(Paragraph(clean_markdown_inline(sub), sub_bullet_style))
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
                # Width of each column proportional to its longest text (clamped), so long text gets room
                weights = []
                for c in range(num_cols):
                    longest = max(len(re.sub(r"[`*]", "", row[c] if c < len(row) else "")) for row in [header_raw] + data_rows)
                    weights.append(min(max(longest, 7), 45))
                col_widths = [total_w * w / sum(weights) for w in weights]

                t = Table(table_data, colWidths=col_widths, repeatRows=1)
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

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report PDF built successfully: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")


if __name__ == "__main__":
    build_pdf()
