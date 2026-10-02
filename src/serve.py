"""Calls used by the one-page app. Prompts, prices, and thresholds stay as frozen."""

import json

import joblib

import extract
import rag_answer
import retrieve

ROOT = retrieve.ROOT
MODEL_FILES = {
    "calibrated_linear_svc": ROOT / "models" / "calibrated_linear_svc.joblib",
    "multinomial_nb": ROOT / "models" / "multinomial_nb.joblib",
}
MODEL_LABELS = {
    "calibrated_linear_svc": "TF-IDF + calibrated LinearSVC",
    "multinomial_nb": "TF-IDF + Multinomial NB",
}


def session_message(calls, usd):
    return (
        f"This session stopped at {calls} model calls and USD {usd:.4f}. "
        f"The cap is {extract.SESSION_MAX_CALLS} calls or USD {extract.SESSION_MAX_USD:.2f}."
    )


def _sample_body(name):
    lines = (ROOT / "data" / "sample" / name).read_text(encoding="utf-8").splitlines()
    body = [
        line for line in lines
        if line and not line.startswith("article_id:") and not line.startswith("category:")
    ]
    return "\n".join(body).strip()


def sample_article():
    return _sample_body("business.txt")


def below_threshold_article():
    return _sample_body("below_threshold.txt")


def thresholds():
    saved = json.loads((ROOT / "results" / "abstention.json").read_text(encoding="utf-8"))
    return {
        name: saved["models"][name]["chosen_threshold"]
        for name in MODEL_FILES
    }


def classify_article(text, model_key):
    text = (text or "").strip()
    if not text:
        return {"error": "Paste an article first."}
    path = MODEL_FILES[model_key]
    if not path.exists():
        return {"error": "Train the classifiers first with python src/classify.py."}
    model = joblib.load(path)
    proba = model.predict_proba([text])[0]
    classes = list(model.named_steps["clf"].classes_)
    ranked = [
        {"label": label, "probability": round(float(score), 4)}
        for label, score in zip(classes, proba)
    ]
    ranked.sort(key=lambda item: -item["probability"])
    threshold = thresholds()[model_key]
    top = ranked[0]
    abstain = top["probability"] < threshold
    return {
        "model": model_key,
        "model_label": MODEL_LABELS[model_key],
        "threshold": threshold,
        "abstain": abstain,
        "label": None if abstain else top["label"],
        "top_label": top["label"],
        "top_probability": top["probability"],
        "ranked": ranked,
        "message": "Not classified. Hand this article to a person." if abstain else top["label"],
    }


def retrieval_bundle():
    encoder = retrieve.SentenceTransformer(
        retrieve.MODEL_NAME, cache_folder=str(retrieve.CACHE / "model")
    )
    vectors, owners = retrieve.build_index(encoder)
    chunks, chunk_owners = retrieve.load_corpus()
    if len(chunks) != len(owners) or not (chunk_owners == owners).all():
        raise RuntimeError("Embedding cache does not match the current chunking.")
    return encoder, vectors, owners, chunks, rag_answer.load_categories()


def log_cost(task, prompt_tokens, completion_tokens, usd):
    path = ROOT / "results" / "cost_log.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "task": task,
            "model": extract.MODEL,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "usd": round(usd, 6),
        }) + "\n")


def _call(function, *args):
    try:
        return function(*args), None
    except (SystemExit, json.JSONDecodeError, TimeoutError) as error:
        message = error.code if isinstance(error, SystemExit) and isinstance(error.code, str) else str(error)
        return None, message or "OPENROUTER_API_KEY is missing from .env"


def _read_key():
    try:
        return extract.load_key(), None
    except SystemExit as error:
        message = error.code if isinstance(error.code, str) and error.code else "OPENROUTER_API_KEY is missing from .env"
        return None, message


def _usd(prompt_tokens, completion_tokens, input_rate, output_rate):
    return round(prompt_tokens * input_rate + completion_tokens * output_rate, 6)


def page_rag_view(abstain, answer, citations):
    """Page display only. This does not rescore results/rag.json."""
    if abstain or not citations:
        return True, rag_answer.INSUFFICIENT, []
    return False, answer, citations


def answer_question(question, calls, usd):
    question = (question or "").strip()
    if not question:
        return {"error": "Type a question first."}
    encoder, vectors, owners, chunks, categories = retrieval_bundle()
    query = retrieve.embed(encoder, [question])[0]
    hits = retrieve.top_hits(query, vectors, owners)
    best = hits[0]["cosine"]
    passages = []
    for number, hit in enumerate(hits, start=1):
        passages.append({
            "passage": number,
            "article_id": hit["article_id"],
            "category": categories[hit["article_id"]],
            "cosine": hit["cosine"],
            "text": chunks[hit["chunk_index"]],
        })
    result = {
        "question": question,
        "best_cosine": best,
        "abstain_cosine": retrieve.ABSTAIN_COSINE,
        "passages": passages,
        "called_model": False,
        "usd": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "temperature": 0,
        "max_tokens": rag_answer.MAX_TOKENS,
    }
    if best < retrieve.ABSTAIN_COSINE:
        result.update({
            "abstain": True,
            "abstain_reason": "retrieval",
            "answer": rag_answer.INSUFFICIENT,
            "citations": [],
        })
        return result
    if extract.session_blocked(calls, usd):
        result["error"] = session_message(calls, usd)
        return result
    key, key_error = _read_key()
    if key_error:
        result["error"] = key_error
        return result
    evidence = rag_answer.evidence_block(hits, chunks, categories)
    prompt_tokens, completion_tokens = 0, 0
    http_attempts = []
    parsed = None
    for _ in range(2):
        spent = _usd(
            prompt_tokens, completion_tokens,
            rag_answer.INPUT_USD_PER_TOKEN, rag_answer.OUTPUT_USD_PER_TOKEN,
        )
        if extract.session_blocked(calls + len(http_attempts), usd + spent):
            break
        payload, error = _call(rag_answer.call_model, key, question, evidence, http_attempts)
        if error:
            cost = _usd(
                prompt_tokens, completion_tokens,
                rag_answer.INPUT_USD_PER_TOKEN, rag_answer.OUTPUT_USD_PER_TOKEN,
            )
            result.update({
                "error": error,
                "called_model": bool(http_attempts),
                "http_attempts": len(http_attempts),
                "usd": cost,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })
            if prompt_tokens or completion_tokens:
                log_cost("ui_rag", prompt_tokens, completion_tokens, cost)
            return result
        piece, piece_prompt, piece_completion = payload
        prompt_tokens += piece_prompt
        completion_tokens += piece_completion
        try:
            abstain, answer, indexes = rag_answer.parse_payload(piece)
            citations = [] if abstain else rag_answer.citation_records(hits, chunks, categories, indexes)
            parsed = (abstain, answer, citations)
            if abstain or citations:
                break
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            parsed = None
    cost = _usd(
        prompt_tokens, completion_tokens,
        rag_answer.INPUT_USD_PER_TOKEN, rag_answer.OUTPUT_USD_PER_TOKEN,
    )
    if http_attempts and (prompt_tokens or completion_tokens):
        log_cost("ui_rag", prompt_tokens, completion_tokens, cost)
    if parsed is None:
        shown_abstain, shown_answer, shown_citations = True, rag_answer.INSUFFICIENT, []
    else:
        shown_abstain, shown_answer, shown_citations = page_rag_view(*parsed)
    result.update({
        "called_model": bool(http_attempts),
        "http_attempts": len(http_attempts),
        "abstain": shown_abstain,
        "abstain_reason": "model" if shown_abstain else None,
        "answer": shown_answer,
        "citations": shown_citations,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "usd": cost,
    })
    return result


def extract_article(article, calls, usd):
    article = (article or "").strip()
    if not article:
        return {"error": "Paste an article first."}
    if extract.session_blocked(calls, usd):
        return {
            "error": session_message(calls, usd),
            "called_model": False,
            "http_attempts": 0,
            "usd": 0.0,
        }
    key, key_error = _read_key()
    if key_error:
        return {"error": key_error, "called_model": False, "http_attempts": 0, "usd": 0.0}
    prompt_tokens = 0
    completion_tokens = 0
    data = None
    http_attempts = []
    for _ in range(2):
        spent = _usd(
            prompt_tokens, completion_tokens,
            extract.INPUT_USD_PER_TOKEN, extract.OUTPUT_USD_PER_TOKEN,
        )
        if extract.session_blocked(calls + len(http_attempts), usd + spent):
            break
        payload, error = _call(extract.call_model, key, article, http_attempts)
        if error:
            cost = _usd(
                prompt_tokens, completion_tokens,
                extract.INPUT_USD_PER_TOKEN, extract.OUTPUT_USD_PER_TOKEN,
            )
            if prompt_tokens or completion_tokens:
                log_cost("ui_extract", prompt_tokens, completion_tokens, cost)
            return {
                "error": error,
                "called_model": bool(http_attempts),
                "http_attempts": len(http_attempts),
                "usd": cost,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        raw, piece_prompt, piece_completion = payload
        prompt_tokens += piece_prompt
        completion_tokens += piece_completion
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                break
            data = None
        except json.JSONDecodeError:
            data = None
    cost = _usd(
        prompt_tokens, completion_tokens,
        extract.INPUT_USD_PER_TOKEN, extract.OUTPUT_USD_PER_TOKEN,
    )
    if prompt_tokens or completion_tokens:
        log_cost("ui_extract", prompt_tokens, completion_tokens, cost)
    if data is None:
        return {
            "error": "The model did not return JSON.",
            "called_model": bool(http_attempts),
            "http_attempts": len(http_attempts),
            "usd": cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    errors = extract.l1_errors(data)
    fields = {field: data.get(field) for field in extract.FIELDS}
    return {
        "called_model": bool(http_attempts),
        "http_attempts": len(http_attempts),
        "l1_pass": not errors,
        "l1_errors": errors,
        "fields": fields,
        "usd": cost,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "temperature": 0,
        "max_tokens": extract.MAX_TOKENS,
    }
