import re
from collections import namedtuple

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
TOKEN_PATTERN = re.compile(r"\w+")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
MIN_OVERLAP = 0.5
MIN_SHARED_TOKENS = 3
MAX_HIGHLIGHTS_PER_SOURCE = 2
STOPWORDS = frozenset(
    "a an the and or but if of in on at to for from by with as is are was were be been "
    "being it its this that these those we our they their them he she his her i you your "
    "not no do does did can could may might will would should have has had also than then "
    "so such which who whom what when where why how into over under about between both "
    "each other more most some any only same very there here".split()
)

Segment = namedtuple("Segment", ["text", "highlighted"])
HighlightedSource = namedtuple(
    "HighlightedSource", ["number", "chunk", "score", "cited", "segments"]
)


def content_tokens(text):
    """Return the set of lowercase non-stopword tokens in the text."""
    return {
        token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS
    }


def cited_numbers(answer, total):
    """Return the valid passage numbers cited in the answer."""
    numbers = {int(number) for number in CITATION_PATTERN.findall(answer)}
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


def split_segments(chunk_text, answer_tokens):
    """Split a chunk into segments, highlighting its sentences that support the answer."""
    sentences = SENTENCE_PATTERN.split(chunk_text)
    scores = [sentence_score(sentence, answer_tokens) for sentence in sentences]
    ranked = sorted(
        (index for index, score in enumerate(scores) if score >= MIN_OVERLAP),
        key=lambda index: -scores[index],
    )
    chosen = set(ranked[:MAX_HIGHLIGHTS_PER_SOURCE])
    return merge_segments(
        [Segment(sentence, index in chosen) for index, sentence in enumerate(sentences)]
    )


def highlight_sources(answer, sources):
    """Mark the supporting sentences in every source cited by the answer."""
    answer_tokens = content_tokens(CITATION_PATTERN.sub(" ", answer))
    cited = cited_numbers(answer, len(sources)) or set(range(1, len(sources) + 1))
    highlighted = []
    for number, (chunk, score) in enumerate(sources, start=1):
        is_cited = number in cited
        if is_cited:
            segments = split_segments(chunk.text, answer_tokens)
        else:
            segments = [Segment(chunk.text, False)]
        highlighted.append(HighlightedSource(number, chunk, score, is_cited, segments))
    return highlighted