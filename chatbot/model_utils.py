from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import joblib
import spacy

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

try:
    NLP = spacy.load("ru_core_news_sm")
except OSError:
    NLP = None

_model = None
_vectorizer = None


def preprocess(text: str) -> str:
    text = (text or "").strip().lower()
    if not text:
        return ""

    if NLP is None:
        return text

    doc = NLP(text)
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            lemma = token.lemma_.strip().lower()
            if lemma:
                tokens.append(lemma)
    return " ".join(tokens)


def load_artifacts() -> Tuple[Optional[object], Optional[object]]:
    global _model, _vectorizer
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    if _vectorizer is None and VECTORIZER_PATH.exists():
        _vectorizer = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer


def model_is_ready() -> bool:
    model, vectorizer = load_artifacts()
    return model is not None and vectorizer is not None


def predict_intent(text: str) -> Tuple[str, float]:
    model, vectorizer = load_artifacts()
    if model is None or vectorizer is None:
        return "unknown", 0.0

    processed = preprocess(text)
    vector = vectorizer.transform([processed])
    intent = model.predict(vector)[0]

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)
        confidence = float(max(probabilities[0]))

    return intent, confidence
