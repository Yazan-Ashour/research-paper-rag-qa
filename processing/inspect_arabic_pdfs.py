import re
from statistics import mean, median

import pymupdf

from arabic_text import arabic_ratio, normalize_arabic
from data_prep import ARABIC_DIR
from pdf_extraction import ARABIC_OCR_LANGUAGES, extract_ocr, extract_pymupdf
from reporting import RESULTS_DIR, save_report

PDF_DIR = ARABIC_DIR / "humyn"
SAMPLE_COUNT = 8
MIN_TEXT_CHARS = 50
SNIPPET_CHARS = 300
REPORT_FILE = "arabic_pdf_inspection.json"
SAMPLES_FILE = "arabic_pdf_samples.txt"
NUMBER_PATTERN = re.compile(r"(\d+)")


def natural_key(path):
    """Sort file names so that page-2 comes before page-10."""
    return [int(part) if part.isdigit() else part for part in NUMBER_PATTERN.split(path.name)]


def inspect_pdf(path):
    """Return the page count, text-layer length, Arabic share and image count of one PDF."""
    with pymupdf.open(path) as document:
        text = " ".join(page.get_text() for page in document)
        images = sum(len(page.get_images()) for page in document)
        return {
            "file": path.name,
            "pages": document.page_count,
            "text_chars": len(text.strip()),
            "arabic_share": round(arabic_ratio(text), 3),
            "images": images,
        }


def evenly_spaced(items, count):
    """Return count items spread evenly across the list."""
    if len(items) <= count:
        return list(items)
    step = (len(items) - 1) / (count - 1)
    return [items[round(index * step)] for index in range(count)]


def word_overlap(source, target):
    """Return the share of distinct source words that also occur in the target."""
    source_words, target_words = set(source.split()), set(target.split())
    return len(source_words & target_words) / len(source_words) if source_words else 0.0


def compare_layers(path):
    """Read one PDF with its text layer and with OCR, and measure how much the two agree."""
    layer = " ".join(extract_pymupdf(path))
    ocr = " ".join(extract_ocr(path, ARABIC_OCR_LANGUAGES))
    layer_words, ocr_words = normalize_arabic(layer), normalize_arabic(ocr)
    return {
        "file": path.name,
        "layer_chars": len(layer.strip()),
        "ocr_chars": len(ocr.strip()),
        "layer_words_found_in_ocr": round(word_overlap(layer_words, ocr_words), 3),
        "ocr_words_found_in_layer": round(word_overlap(ocr_words, layer_words), 3),
        "layer_text": layer.strip()[:SNIPPET_CHARS],
        "ocr_text": ocr.strip()[:SNIPPET_CHARS],
    }


def write_samples(comparisons):
    """Save the text-layer and OCR snippets for reading in an editor that shapes Arabic correctly."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / SAMPLES_FILE
    with open(path, "w", encoding="utf-8") as file:
        for item in comparisons:
            file.write(f"=== {item['file']}\n--- text layer\n{item['layer_text']}\n--- OCR\n{item['ocr_text']}\n\n")
    return path


def main():
    """Inspect a folder of Arabic PDFs: how many have text layers, and how the layer compares with OCR."""
    files = sorted(PDF_DIR.glob("*.pdf"), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No PDF files found in {PDF_DIR}")
    inspected = [inspect_pdf(path) for path in files]
    with_text = [item for item in inspected if item["text_chars"] >= MIN_TEXT_CHARS]

    print(f"PDF files: {len(files)}  pages: {sum(item['pages'] for item in inspected)}")
    print(f"With a text layer (>= {MIN_TEXT_CHARS} characters): {len(with_text)}")
    print(f"Image-only (no usable text layer): {len(files) - len(with_text)}")
    print(f"Median text-layer length: {median(item['text_chars'] for item in inspected):.0f} characters")
    print(f"Average Arabic share of text-layer letters: {mean(item['arabic_share'] for item in inspected):.1%}")

    sample = evenly_spaced(files, SAMPLE_COUNT)
    print(f"\nComparing text layer and OCR on {len(sample)} sample pages...", flush=True)
    comparisons = [compare_layers(path) for path in sample]
    for item in comparisons:
        print(
            f"{item['file'][-22:]:<24}layer {item['layer_chars']:>5} chars  OCR {item['ocr_chars']:>5} chars  "
            f"layer-in-OCR {item['layer_words_found_in_ocr']:.0%}  OCR-in-layer {item['ocr_words_found_in_layer']:.0%}"
        )

    report = {
        "settings": {
            "folder": str(PDF_DIR),
            "sample_count": SAMPLE_COUNT,
            "ocr_languages": ARABIC_OCR_LANGUAGES,
            "agreement": "Share of distinct normalized words of one reading that also appear in the other; no ground truth is involved",
        },
        "files": inspected,
        "comparisons": [{key: value for key, value in item.items() if not key.endswith("_text")} for item in comparisons],
    }
    path = save_report(report, REPORT_FILE)
    print(f"\nReport saved to {path}")
    print(f"Text snippets for reading saved to {write_samples(comparisons)}")


if __name__ == "__main__":
    main()