import re
from collections import namedtuple

from arabic_text import detect_language, normalize_arabic

CITATION_PATTERN = re.compile(r"\[\s*(\d+(?:\s*[,،]\s*\d+)*)\s*\]")
TOKEN_PATTERN = re.compile(r"\w+")
ANCHOR_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?\u061f])\s+")
ARABIC_PREFIX_PATTERN = re.compile(r"^(?:وال|بال|كال|فال|لل|ال)(?=\w{3,})")
DIGIT_TABLE = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫", "01234567890123456789.")
MIN_OVERLAP = 0.5
MIN_SHARED_TOKENS = 3
ANCHOR_MIN_OVERLAP = 0.3
MIN_SHARED_ANCHORS = 2
MAX_HIGHLIGHTS_PER_SOURCE = 2
ENGLISH_STOPWORDS = frozenset(
    "a an the and or but if of in on at to for from by with as is are was were be been "
    "being it its this that these those we our they their them he she his her i you your "
    "not no do does did can could may might will would should have has had also than then "
    "so such which who whom what when where why how into over under about between both "
    "each other more most some any only same very there here".split()
)
ARABIC_STOPWORDS = frozenset(
    normalize_arabic(word)
    for word in (
        "في من على إلى الى عن أن ان إن او أو و هذا هذه ذلك تلك التي الذي الذين كان كانت يكون تكون "
        "هو هي هم هن ما ماذا كيف لا لم لن قد كل بعد قبل بين مع حتى أيضا أيضًا كما لقد إذا اذا ثم "
        "حيث هل لكن عند عندما منذ خلال ضمن هناك تم يتم جرى لدى هنا نحو فقط بشكل"
    ).split()
)
STOPWORDS = ENGLISH_STOPWORDS | ARABIC_STOPWORDS

Segment = namedtuple("Segment", ["text", "highlighted"])
HighlightedSource = namedtuple(
    "HighlightedSource", ["number", "chunk", "score", "cited", "segments"]
)


def content_tokens(text):
    """Return the set of normalized non-stopword tokens, with Arabic letters unified and the article stripped."""
    normalized = normalize_arabic(text).translate(DIGIT_TABLE).lower()
    tokens = set()
    for token in TOKEN_PATTERN.findall(normalized):
        if token in STOPWORDS:
            continue
        token = ARABIC_PREFIX_PATTERN.sub("", token)
        if token and token not in STOPWORDS:
            tokens.add(token)
    return tokens


def anchor_tokens(text):
    """Return the Latin-script words, acronyms and numbers (BERT, 0.96) that survive translation, with a weight each.

    Distinctive anchors (digits, hyphens, two or more capitals) weigh 2 and plain words weigh 1.
    """
    anchors = {}
    for raw in ANCHOR_PATTERN.findall(text.translate(DIGIT_TABLE)):
        token = raw.lower()
        if len(token) < 2 or token in STOPWORDS:
            continue
        distinctive = any(c.isdigit() for c in raw) or "-" in raw or "." in raw or sum(c.isupper() for c in raw) >= 2
        anchors[token] = max(anchors.get(token, 1), 2 if distinctive else 1)
    return anchors


def cited_numbers(answer, total):
    """Return the valid passage numbers cited in the answer, including lists like [1, 2] or [1، 2]."""
    numbers = {int(n) for group in CITATION_PATTERN.findall(answer) for n in re.findall(r"\d+", group)}
    return {number for number in numbers if 1 <= number <= total}


def sentence_score(sentence, answer_tokens):
    """Return how strongly a sentence is supported by the answer, from 0 to 1."""
    if not answer_tokens:
        return 0.0
    tokens = content_tokens(sentence)
    shared = len(tokens & answer_tokens)
    if not tokens or shared < min(MIN_SHARED_TOKENS, len(answer_tokens)):
        return 0.0
    return shared / min(len(tokens), len(answer_tokens))


def anchor_score(sentence, answer_anchors):
    """Return the weighted share of the answer's anchors found in a sentence, for a source in another language."""
    if not answer_anchors:
        return 0.0
    sentence_anchors = anchor_tokens(sentence)
    shared = [token for token in answer_anchors if token in sentence_anchors]
    has_distinctive = any(answer_anchors[token] == 2 for token in shared)
    if not has_distinctive and len(shared) < MIN_SHARED_ANCHORS:
        return 0.0
    return sum(answer_anchors[token] for token in shared) / sum(answer_anchors.values())


def merge_segments(segments):
    """Join neighbouring segments that share the same highlight state."""
    merged = []
    for segment in segments:
        if merged and merged[-1].highlighted == segment.highlighted:
            joined = f"{merged[-1].text} {segment.text}"
            merged[-1] = Segment(joined, segment.highlighted)
        else:
            merged.append(segment)
    return merged


def split_segments(chunk_text, answer_tokens, anchors_only=False):
    """Split a chunk into segments, highlighting its sentences that support the answer.

    With anchors_only, answer_tokens must be the answer's anchor tokens and only those are compared,
    which is used when the answer and the passage are in different languages.
    """
    sentences = SENTENCE_PATTERN.split(chunk_text)
    score_sentence, threshold = (anchor_score, ANCHOR_MIN_OVERLAP) if anchors_only else (sentence_score, MIN_OVERLAP)
    scores = [score_sentence(sentence, answer_tokens) for sentence in sentences]
    ranked = sorted(
        (index for index, score in enumerate(scores) if score >= threshold),
        key=lambda index: -scores[index],
    )
    chosen = set(ranked[:MAX_HIGHLIGHTS_PER_SOURCE])
    return merge_segments(
        [Segment(sentence, index in chosen) for index, sentence in enumerate(sentences)]
    )


def highlight_sources(answer, sources):
    """Mark the supporting sentences in every source cited by the answer, in the same or another language."""
    plain_answer = CITATION_PATTERN.sub(" ", answer)
    answer_language = detect_language(plain_answer)
    answer_tokens = content_tokens(plain_answer)
    answer_anchors = anchor_tokens(plain_answer)
    cited = cited_numbers(answer, len(sources)) or set(range(1, len(sources) + 1))
    highlighted = []
    for number, (chunk, score) in enumerate(sources, start=1):
        is_cited = number in cited
        if not is_cited:
            segments = [Segment(chunk.text, False)]
        elif detect_language(chunk.text) == answer_language:
            segments = split_segments(chunk.text, answer_tokens)
        else:
            segments = split_segments(chunk.text, answer_anchors, anchors_only=True)
        highlighted.append(HighlightedSource(number, chunk, score, is_cited, segments))
    return highlighted