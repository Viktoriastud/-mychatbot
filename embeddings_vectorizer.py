from __future__ import annotations

from typing import Iterable, List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from nlp_utils import get_nlp


class WordEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = "ru_core_news_md"):
        self.model_name = model_name
        self._nlp = None
        self.vector_size = None

    def _get_nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp(self.model_name)
        return self._nlp

    def fit(self, texts: Iterable[str], y=None):
        nlp = self._get_nlp()
        self.vector_size = nlp.vocab.vectors_length
        if not self.vector_size:
            raise ValueError(
                "У выбранной spaCy-модели нет word embeddings. "
                "Установите модель ru_core_news_md: python -m spacy download ru_core_news_md"
            )
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        nlp = self._get_nlp()
        if self.vector_size is None:
            self.vector_size = nlp.vocab.vectors_length
        result: List[np.ndarray] = []
        for text in texts:
            doc = nlp((text or "").strip().lower())
            vector = np.asarray(doc.vector, dtype=np.float32)
            if vector.size == 0:
                vector = np.zeros(self.vector_size, dtype=np.float32)
            result.append(vector)
        return np.vstack(result)
