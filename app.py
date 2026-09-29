"""One page for classification, archive questions, and field extraction."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import extract
import rag_answer
import retrieve
import serve

BANNER = (
    "BBC News 2004–2005. For editorial sorting, finding archive articles, and this course demo. "
    "Not for automated publishing, fact-checking, or legal review."
)
GROUNDED_QUESTION = "Which country awarded Burren Energy two new oil exploration contracts?"
ADVERSARIAL_QUESTION = "Who won the 2024 United States presidential election?"

st.set_page_config(page_title="NewsIQ", layout="wide")
if "calls" not in st.session_state:
    st.session_state.calls = 0
    st.session_state.usd = 0.0

st.title("NewsIQ")
st.info(BANNER)


def remember(result):
    if result.get("called_model"):
        st.session_state.calls += 1
        st.session_state.usd += float(result.get("usd") or 0)


def show_cost(result):
    st.write(
        f"This call: USD {float(result.get('usd') or 0):.6f} "
        f"({result.get('prompt_tokens', 0)} prompt tokens, "
        f"{result.get('completion_tokens', 0)} completion tokens)."
    )
    st.write(f"Session total: USD {st.session_state.usd:.4f} across {st.session_state.calls} calls.")


def load_classify_sample():
    st.session_state.classify_text = serve.sample_article()


def load_below_threshold():
    st.session_state.classify_text = "hello"


def load_archive_question():
    st.session_state.ask_text = GROUNDED_QUESTION


def load_outside_question():
    st.session_state.ask_text = ADVERSARIAL_QUESTION


def load_extract_sample():
    st.session_state.extract_text = serve.sample_article()


classify_tab, ask_tab, extract_tab = st.tabs(["Classify", "Ask the archive", "Extract"])

with classify_tab:
    st.caption("The abstention threshold was chosen on the validation split and is not changed here.")
    sample_col, short_col = st.columns(2)
    with sample_col:
        st.button("Load sample article", key="load_classify", on_click=load_classify_sample)
    with short_col:
        st.button("Load a text below the threshold", on_click=load_below_threshold)
    article = st.text_area("Article", key="classify_text", height=240)
    model_key = st.selectbox(
        "Classifier",
        list(serve.MODEL_LABELS),
        format_func=lambda key: serve.MODEL_LABELS[key],
    )
    if st.button("Classify", type="primary"):
        outcome = serve.classify_article(article, model_key)
        if outcome.get("error"):
            st.error(outcome["error"])
        elif outcome["abstain"]:
            st.warning(outcome["message"])
            st.write(
                f"Highest class was {outcome['top_label']} at {outcome['top_probability']:.2f}. "
                f"The threshold is {outcome['threshold']:.2f}, so the article is not classified."
            )
            st.dataframe(outcome["ranked"], hide_index=True)
        else:
            st.success(outcome["label"])
            st.write(
                f"Confidence {outcome['top_probability']:.2f}. "
                f"Accepted because it is at or above the threshold {outcome['threshold']:.2f}."
            )
            st.dataframe(outcome["ranked"], hide_index=True)

with ask_tab:
    st.caption(
        f"Temperature 0, max tokens {rag_answer.MAX_TOKENS}. "
        f"If the best passage is below cosine {retrieve.ABSTAIN_COSINE:.2f}, the page does not call the generator."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.button("Load an archive question", on_click=load_archive_question)
    with col_b:
        st.button("Load a question outside the archive", on_click=load_outside_question)
    question = st.text_input("Question", key="ask_text")
    if st.button("Ask", type="primary"):
        with st.spinner("Searching the 2004–2005 archive"):
            outcome = serve.answer_question(question, st.session_state.calls, st.session_state.usd)
        remember(outcome)
        if outcome.get("error") and not outcome.get("answer"):
            st.error(outcome["error"])
        else:
            if outcome.get("abstain"):
                st.warning(outcome["answer"])
                reason = "the best passage is below the cosine cutoff" if outcome.get("abstain_reason") == "retrieval" else "the generator found no answer in the passages"
                st.write(f"Abstaining because {reason}. Best cosine {outcome['best_cosine']:.4f}.")
            else:
                st.success(outcome["answer"])
                st.write(f"Best cosine {outcome['best_cosine']:.4f}.")
            if outcome.get("citations"):
                st.subheader("Citations")
                for item in outcome["citations"]:
                    st.markdown(
                        f"Passage {item['passage']}, article {item['article_id']} "
                        f"({item['category']}), cosine {item['cosine']:.4f}"
                    )
                    st.write(item["snippet"])
            elif outcome.get("passages"):
                st.subheader("Passages checked")
                for item in outcome["passages"]:
                    st.markdown(
                        f"Passage {item['passage']}, article {item['article_id']} "
                        f"({item['category']}), cosine {item['cosine']:.4f}"
                    )
                    st.write(item["text"][:240])
            show_cost(outcome)

with extract_tab:
    st.caption(f"Temperature 0, max tokens {extract.MAX_TOKENS}. Five fields: people, organisations, locations, dates, topic.")
    st.button("Load sample article", key="load_extract", on_click=load_extract_sample)
    article = st.text_area("Article", key="extract_text", height=240)
    if st.button("Extract", type="primary"):
        with st.spinner("Extracting fields"):
            outcome = serve.extract_article(article, st.session_state.calls, st.session_state.usd)
        remember(outcome)
        if outcome.get("error") and not outcome.get("called_model"):
            st.error(outcome["error"])
        elif outcome.get("error"):
            st.error(outcome["error"])
            show_cost(outcome)
        else:
            if outcome["l1_pass"]:
                st.success("The JSON has the five required fields.")
            else:
                st.error("Schema check failed: " + "; ".join(outcome["l1_errors"]))
            st.json(outcome["fields"])
            show_cost(outcome)

with st.sidebar:
    st.header("This session")
    st.write(f"Model calls: {st.session_state.calls} / {extract.SESSION_MAX_CALLS}")
    st.write(f"Cost: USD {st.session_state.usd:.4f} / {extract.SESSION_MAX_USD:.2f}")
    st.caption("The page stops at 30 calls or USD 0.05, whichever comes first.")
    st.caption("Price card: input USD 0.15 per million tokens, output USD 0.60 per million tokens.")
    st.caption(f"Generator: {extract.MODEL}, temperature 0.")
    st.caption(f"Archive answers use at most {rag_answer.MAX_TOKENS} tokens. Extraction uses at most {extract.MAX_TOKENS}.")
    st.caption(f"Retrieval abstains below cosine {retrieve.ABSTAIN_COSINE:.2f}. That check does not call the generator.")
