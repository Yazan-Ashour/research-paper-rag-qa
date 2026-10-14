import json
from pathlib import Path

import pymupdf
from fpdf import FPDF

from data_prep import ARABIC_DIR, ARABIC_PDF_DIR
from processing.prepare_arabic import ARTICLES_FILE, load_rows

ARABIC_SCAN_DIR = ARABIC_DIR / "scans"
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
FONT_NAME = "arabic"
TITLE_SIZE = 20
BODY_SIZE = 14
LINE_HEIGHT = 9
PARAGRAPH_GAP = 4
SCAN_DPI = 150


def find_font():
    """Return the first installed font that has Arabic glyphs."""
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("No Arabic font found; add a .ttf path to FONT_CANDIDATES")


def article_paragraphs(rows, titles):
    """Return the unique ARCD paragraphs of each title in order of first appearance."""
    paragraphs = {title: [] for title in titles}
    for row in rows:
        title, context = row["title"], row["context"].strip()
        if title in paragraphs and context not in paragraphs[title]:
            paragraphs[title].append(context)
    return paragraphs


def render_pdf(title, paragraphs, font_path, destination):
    """Write an article as an Arabic PDF with a real text layer."""
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font(FONT_NAME, fname=font_path)
    pdf.set_text_shaping(use_shaping_engine=True, direction="rtl", script="arab", language="ara")
    pdf.set_font(FONT_NAME, size=TITLE_SIZE)
    pdf.multi_cell(0, LINE_HEIGHT + 3, title, align="R")
    pdf.ln(PARAGRAPH_GAP)
    pdf.set_font(FONT_NAME, size=BODY_SIZE)
    for paragraph in paragraphs:
        pdf.multi_cell(0, LINE_HEIGHT, paragraph, align="R")
        pdf.ln(PARAGRAPH_GAP)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(destination))


def rasterize(source, destination):
    """Turn a PDF into an image-only copy, like a scanned document with no text layer."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(source) as document, pymupdf.open() as scanned:
        for page in document:
            pixmap = page.get_pixmap(dpi=SCAN_DPI, colorspace=pymupdf.csGRAY)
            scanned_page = scanned.new_page(width=page.rect.width, height=page.rect.height)
            scanned_page.insert_image(scanned_page.rect, pixmap=pixmap)
        scanned.save(str(destination), deflate=True)


def page_count(path):
    """Return the number of pages in a PDF."""
    with pymupdf.open(path) as document:
        return document.page_count


def main():
    """Render every selected Arabic article as a digital PDF and as an image-only scan."""
    with open(ARABIC_DIR / ARTICLES_FILE, encoding="utf-8") as file:
        articles = json.load(file)
    paragraphs = article_paragraphs(load_rows(), [article["title"] for article in articles])
    font_path = find_font()
    print(f"Font: {font_path}")

    for article in articles:
        digital = ARABIC_PDF_DIR / f"{article['id']}.pdf"
        scan = ARABIC_SCAN_DIR / f"{article['id']}.pdf"
        render_pdf(article["title"], paragraphs[article["title"]], font_path, digital)
        rasterize(digital, scan)
        print(f"{article['id']}: {len(paragraphs[article['title']])} paragraphs, {page_count(digital)} pages")
    print(f"\nDigital PDFs: {ARABIC_PDF_DIR}/   Image-only scans: {ARABIC_SCAN_DIR}/")


if __name__ == "__main__":
    main()