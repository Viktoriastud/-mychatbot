from __future__ import annotations

import spacy

_nlp_cache = {}


def get_nlp(model_name: str = "ru_core_news_sm"):
    if model_name not in _nlp_cache:
        try:
            _nlp_cache[model_name] = spacy.load(model_name)
        except OSError as exc:
            raise OSError(
                f"Не удалось загрузить модель spaCy {model_name}. "
                "Установите её командой: python -m spacy download ru_core_news_sm"
            ) from exc
    return _nlp_cache[model_name]


def normalize_city(ent):
    words = []
    for token in ent:
        lemma = token.lemma_
        if lemma:
            words.append(lemma.capitalize())
    return " ".join(words)


def extract_city(text: str):
    try:
        nlp = get_nlp("ru_core_news_sm")
    except OSError:
        return None

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ("LOC", "GPE"):
            return normalize_city(ent)
    return None
