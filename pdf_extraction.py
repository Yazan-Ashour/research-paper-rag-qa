import io
import json
import re
import shutil
import unicodedata
from functools import partial
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image
from pypdf import PdfReader

OCR_DPI = 200
OCR_LANGUAGES = "eng"
ARABIC_OCR_LANGUAGES = "ara+eng"
WINDOWS_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)

HYPHEN_BREAK_PATTERN = re.compile(r"(?<=[a-z])-[ \t]*\n[ \t]*(?=[a-z])")
PAGE_NUMBER_LINE_PATTERN = re.compile(r"^[ \t]*\d{1,3}[ \t]*$", re.MULTILINE)
NAME_CITATION_PATTERN = re.compile(r"\(\s*[A-Z][^()]*?\b(?:19|20)\d{2}[a-z]?\s*\)")
YEAR_CITATION_PATTERN = re.compile(r"\(\s*(?:19|20)\d{2}[a-z]?\s*\)")
NUMBER_CITATION_PATTERN = re.compile(r"\[\s*\d+(?:\s*[,\u2013-]\s*\d+)*\s*\]")
SPACE_BEFORE_PUNCT_PATTERN = re.compile(r"\s+([,.;:!?])")
WHITESPACE_PATTERN = re.compile(r"\s+")


def extract_pypdf(path):
    """Return the text layer of every page, read with pypdf."""
    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def extract_pymupdf(path):
    """Return the text layer of every page, read with PyMuPDF."""
    with pymupdf.open(path) as document:
        return [page.get_text() for page in document]


def locate_tesseract():
    """Return the path of the Tesseract executable, or None if it is not installed."""
    found = shutil.which("tesseract")
    if found:
        return found
    return next((path for path in WINDOWS_TESSERACT_PATHS if Path(path).exists()), None)


def extract_ocr(path, languages=OCR_LANGUAGES):
    """Render every page to an image and read it with Tesseract OCR."""
    executable = locate_tesseract()
    if executable is None:
        raise RuntimeError("Tesseract is not installed or not on PATH")
    pytesseract.pytesseract.tesseract_cmd = executable
    pages = []
    with pymupdf.open(path) as document:
        for page in document:
            image = Image.open(io.BytesIO(page.get_pixmap(dpi=OCR_DPI).tobytes("png")))
            pages.append(pytesseract.image_to_string(image, lang=languages))
    return pages


EXTRACTORS = {
    "pypdf": extract_pypdf,
    "pymupdf": extract_pymupdf,
    "ocr": extract_ocr,
}
ARABIC_EXTRACTORS = {
    "pypdf": extract_pypdf,
    "pymupdf": extract_pymupdf,
    "ocr_ara": partial(extract_ocr, languages=ARABIC_OCR_LANGUAGES),
}
ALL_EXTRACTORS = {**EXTRACTORS, **ARABIC_EXTRACTORS}


def cache_file(path, extractor, cache_dir):
    """Return the cache location for the raw text extracted from one PDF."""
    return Path(cache_dir) / extractor / f"{Path(path).stem}.json"


def extract_pages(path, extractor, cache_dir, refresh=False):
    """Return the raw text of each page, using the on-disk cache unless refresh is set."""
    if extractor not in ALL_EXTRACTORS:
        raise ValueError(f"unknown extractor '{extractor}', choose from {sorted(ALL_EXTRACTORS)}")
    cache = cache_file(path, extractor, cache_dir)
    if cache.exists() and not refresh:
        with open(cache, encoding="utf-8") as file:
            return json.load(file)
    pages = ALL_EXTRACTORS[extractor](path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as file:
        json.dump(pages, file, ensure_ascii=False)
    return pages


def clean_page_text(text):
    """Normalize one page of extracted text and remove hyphenation, page numbers and citations."""
    text = unicodedata.normalize("NFKC", text)
    text = HYPHEN_BREAK_PATTERN.sub("", text)
    text = PAGE_NUMBER_LINE_PATTERN.sub(" ", text)
    text = text.replace("\n", " ")
    text = NAME_CITATION_PATTERN.sub(" ", text)
    text = YEAR_CITATION_PATTERN.sub(" ", text)
    text = NUMBER_CITATION_PATTERN.sub(" ", text)
    text = SPACE_BEFORE_PUNCT_PATTERN.sub(r"\1", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()