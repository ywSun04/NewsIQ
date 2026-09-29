"""Render the trade-off markdown into a PDF. Body text only; no extra claims."""

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "NewsIQ_tradeoff_analysis.md"
DEST = ROOT / "NewsIQ_tradeoff_analysis.pdf"


def main():
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)
    pdf.add_page()
    pdf.set_title("NewsIQ trade-off analysis")
    pdf.set_author("Sun Yawen")
    for raw in SOURCE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            pdf.ln(3)
            continue
        if line.startswith("# "):
            pdf.set_font("Times", "B", 16)
            pdf.multi_cell(0, 8, line[2:])
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Times", "B", 13)
            pdf.multi_cell(0, 7, line[3:])
            pdf.ln(1)
        else:
            pdf.set_font("Times", size=11)
            pdf.multi_cell(0, 6, line)
    pdf.output(DEST)
    print(DEST, "pages", pdf.pages_count)


if __name__ == "__main__":
    main()
