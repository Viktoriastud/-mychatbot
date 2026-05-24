from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from model_utils import preprocess, MODEL_PATH, VECTORIZER_PATH

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "intents_dataset.csv"


def main() -> None:
    data = pd.read_csv(DATASET_PATH)

    if "text" not in data.columns or "intent" not in data.columns:
        raise ValueError("В CSV должны быть столбцы text и intent")

    texts = data["text"].astype(str)
    labels = data["intent"].astype(str)
    processed_texts = [preprocess(text) for text in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    print("Accuracy:", round(model.score(X_test_vec, y_test), 4))
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Модель сохранена: {MODEL_PATH}")
    print(f"Векторизатор сохранён: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
