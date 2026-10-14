import html

import streamlit as st
from openai import OpenAIError

from arabic_text import detect_language
from feedback import FeedbackStore
from highlighting import highlight_sources
from rag_pipeline import TOP_K, RAGPipeline
from vector_store import INDEX_DIR, INDEX_FILE, VectorStore

ALL_PAPERS = "All papers"
MIN_PASSAGES = 1
MAX_PASSAGES = 10
EXAMPLE_QUESTIONS = (
    "What does KBQA stand for?",
    "Which datasets were used for sensitive data detection in Spanish clinical text?",
    "What is the core component for KBQA?",
    "Who won the 2018 FIFA World Cup?",
)
EXAMPLE_QUESTIONS_AR = (
    "ما الذي يرمز إليه الاختصار KBQA؟",
    "ما هي مجموعات البيانات السريرية المستخدمة في الورقة البحثية؟",
)
MARK_STYLE = "background-color:#ffe066;color:#000;padding:1px 3px;border-radius:3px"
PAGE_CSS = """
<style>
[data-testid="stChatInput"] textarea { unicode-bidi: plaintext; text-align: start; }
</style>
"""


@st.cache_resource(show_spinner="Loading embedding model and index...")
def load_pipeline():
    """Load the saved vector store and build the RAG pipeline once per server session."""
    return RAGPipeline(VectorStore.load())


@st.cache_resource
def load_feedback_store():
    """Create the feedback store once per server session."""
    return FeedbackStore()


def paper_options(store):
    """Return a mapping from paper title to paper id for the indexed papers."""
    return {chunk.title: chunk.paper_id for chunk in store.chunks}


def model_for(pipeline, question):
    """Return the model that will answer the question, which is the Arabic model for Arabic questions."""
    model_ar = getattr(pipeline, "model_ar", None)
    return model_ar if model_ar and detect_language(question) == "ar" else pipeline.model


def text_direction(text):
    """Return rtl for mostly Arabic text and ltr otherwise."""
    return "rtl" if detect_language(text) == "ar" else "ltr"


def directional_html(text):
    """Render plain text as a block that reads right to left or left to right, keeping line breaks."""
    safe = html.escape(text).replace("$", "&#36;").replace("\n", "<br>")
    return f'<div dir="{text_direction(text)}" style="text-align:start;line-height:1.7">{safe}</div>'


def segments_to_html(segments, direction="ltr"):
    """Render text segments as safe HTML with highlighted segments wrapped in mark tags."""
    parts = []
    for segment in segments:
        text = html.escape(segment.text).replace("$", "&#36;").replace("\n", " ")
        if segment.highlighted:
            text = f'<mark style="{MARK_STYLE}">{text}</mark>'
        parts.append(text)
    return f'<div dir="{direction}" style="text-align:start;line-height:1.7">{" ".join(parts)}</div>'


def describe_sources(sources):
    """Return JSON-serializable records describing the sources shown with an answer."""
    return [
        {
            "number": source.number,
            "chunk_id": source.chunk.chunk_id,
            "title": source.chunk.title,
            "section": source.chunk.section,
            "score": round(source.score, 4),
            "cited": source.cited,
        }
        for source in sources
    ]


def submit_feedback(rating, store):
    """Save the rating and comment for the current answer, with the model that wrote it."""
    result = st.session_state.result
    store.add(
        question=result["question"],
        answer=result["answer"],
        rating=rating,
        sources=describe_sources(result["sources"]),
        model=result["model"],
        comment=st.session_state.get("feedback_comment", "").strip(),
    )
    result["rating"] = rating
    st.session_state.feedback_comment = ""


def ask_question(pipeline, question, paper_id, top_k):
    """Run the pipeline and keep the answer with its highlighted sources in the session."""
    try:
        with st.spinner("Searching the papers and generating an answer..."):
            result = pipeline.ask(question, top_k=top_k, paper_id=paper_id)
    except OpenAIError as error:
        st.session_state.result = None
        st.error(f"The language model request failed: {error}")
        return
    st.session_state.result = {
        "question": question,
        "answer": result.answer,
        "sources": highlight_sources(result.answer, result.sources),
        "model": model_for(pipeline, question),
        "rating": None,
    }


def render_feedback(result, store):
    """Show the thumbs up and down buttons with an optional comment box."""
    rated = result["rating"] is not None
    st.text_input("Optional comment", key="feedback_comment", disabled=rated)
    up_column, down_column, _ = st.columns([1, 1, 8])
    up_column.button(
        "👍", key="rate_up", on_click=submit_feedback, args=("up", store), disabled=rated
    )
    down_column.button(
        "👎", key="rate_down", on_click=submit_feedback, args=("down", store), disabled=rated
    )
    if rated:
        st.caption("Thanks, your feedback was saved.")


def render_sources(sources):
    """Show each retrieved passage with the supporting sentences highlighted, in its own text direction."""
    if not sources:
        st.caption("No sources are shown because nothing relevant was found.")
        return
    st.markdown("#### Sources")
    st.caption("Highlighted sentences are the passages that support the answer.")
    for source in sources:
        chunk = source.chunk
        section = chunk.section.replace(":::", " >")
        label = f"[{source.number}] {chunk.title} | {section} (score {source.score:.2f})"
        with st.expander(label, expanded=source.cited):
            st.markdown(
                segments_to_html(source.segments, text_direction(chunk.text)),
                unsafe_allow_html=True,
            )


def render_result(result, store):
    """Show the question, the answer with feedback controls, and the sources."""
    with st.chat_message("user"):
        st.markdown(directional_html(result["question"]), unsafe_allow_html=True)
    with st.chat_message("assistant"):
        st.markdown(directional_html(result["answer"]), unsafe_allow_html=True)
        render_feedback(result, store)
    render_sources(result["sources"])


def render_sidebar(pipeline, papers, store):
    """Draw the settings sidebar and return the paper choice, top_k and clicked example."""
    st.sidebar.title("Settings")
    choice = st.sidebar.selectbox("Search in", [ALL_PAPERS, *papers])
    top_k = st.sidebar.slider("Passages to retrieve", MIN_PASSAGES, MAX_PASSAGES, TOP_K)

    st.sidebar.markdown("#### Try an example")
    selected = None
    for example in (*EXAMPLE_QUESTIONS, *EXAMPLE_QUESTIONS_AR):
        if st.sidebar.button(example):
            selected = example

    stats = store.summary()
    st.sidebar.markdown("#### Feedback so far")
    st.sidebar.write(f"👍 {stats['up']}   👎 {stats['down']}")
    st.sidebar.caption(f"LLM: {pipeline.model}")
    if getattr(pipeline, "model_ar", None):
        st.sidebar.caption(f"Arabic LLM: {pipeline.model_ar}")
    st.sidebar.caption(f"Embeddings: {pipeline.store.model_name}")
    return choice, top_k, selected


def main():
    """Run the Streamlit question answering app."""
    st.set_page_config(page_title="Research Paper Q&A", page_icon="📚", layout="wide")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.title("Research Paper Q&A")
    st.caption(
        "Retrieval-augmented answers with highlighted sources. "
        "Ask in English or Arabic about the QASPER research papers."
    )

    if not (INDEX_DIR / INDEX_FILE).exists():
        st.error("No index found. Run `python vector_store.py` first.")
        st.stop()

    pipeline = load_pipeline()
    store = load_feedback_store()
    papers = paper_options(pipeline.store)
    st.session_state.setdefault("result", None)

    choice, top_k, selected = render_sidebar(pipeline, papers, store)
    question = st.chat_input("Ask a question about the papers (English or Arabic)") or selected
    if question:
        ask_question(pipeline, question, papers.get(choice), top_k)

    result = st.session_state.result
    if result:
        render_result(result, store)
    else:
        st.info("Ask a question below or pick an example from the sidebar.")


if __name__ == "__main__":
    main()