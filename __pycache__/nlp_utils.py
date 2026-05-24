import spacy

try:
    nlp = spacy.load("ru_core_news_sm")
except OSError:
    nlp = None


def normalize_city(ent):
    words = []

    for token in ent:
        lemma = token.lemma_
        if lemma:
            words.append(lemma.capitalize())

    return " ".join(words)


def extract_city(text: str):
    if nlp is None:
        return None

    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ in ("LOC", "GPE"):
            return normalize_city(ent)

    return None
