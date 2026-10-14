import re
import unicodedata

DIACRITICS_PATTERN = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
ARABIC_LETTER_PATTERN = re.compile(r"[\u0621-\u064A]")
LETTER_VARIANTS = str.maketrans(
    {
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0622": "\u0627",
        "\u0671": "\u0627",
        "\u0649": "\u064A",
        "\u0629": "\u0647",
    }
)

ARABIC_THRESHOLD = 0.3

def normalize_arabic(text):
    """Return text with compatibility forms, diacritics, tatweel and common letter variants normalized."""
    text = unicodedata.normalize("NFKC", text)
    text = DIACRITICS_PATTERN.sub("", text)
    return text.translate(LETTER_VARIANTS)


def arabic_ratio(text):
    """Return the share of alphabetic characters in the text that are Arabic letters."""
    letters = sum(1 for character in text if character.isalpha())
    return len(ARABIC_LETTER_PATTERN.findall(text)) / max(letters, 1)

def detect_language(text):
    """Return 'ar' if the text is mostly Arabic letters, otherwise 'en'."""
    return "ar" if arabic_ratio(text) >= ARABIC_THRESHOLD else "en"