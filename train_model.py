from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "DeepPavlov/rubert-base-cased"
RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "intents_dataset.csv"
MODEL_DIR = BASE_DIR / "intent_model"
LABEL_MAP_PATH = BASE_DIR / "label_mapping.json"


class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)



def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Не найден файл датасета: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, encoding="utf-8")

    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError("В CSV должны быть столбцы 'text' и 'intent'")

    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["intent"] != "")]
    df = df.dropna(subset=["text", "intent"]).reset_index(drop=True)

    
    df = df[~df["text"].str.contains(r"\\x", regex=True)]

    if len(df) < 8:
        raise ValueError("Слишком мало данных для обучения")

    return df.reset_index(drop=True)



def build_label_maps(intents: list[str]):
    unique_labels = sorted(set(intents))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label



def tokenize_texts(tokenizer, texts: list[str]):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}



def main():
    print("Загрузка датасета...")
    df = load_dataset()
    label2id, id2label = build_label_maps(df["intent"].tolist())
    df["label"] = df["intent"].map(label2id)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"].tolist(),
    )

    print("Интенты:")
    for label, idx in label2id.items():
        print(f"  {idx}: {label}")

    print("Загрузка токенизатора и модели BERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label={str(k): v for k, v in id2label.items()},
        label2id=label2id,
    )

    print("Токенизация текстов...")
    train_encodings = tokenize_texts(tokenizer, train_texts)
    val_encodings = tokenize_texts(tokenizer, val_texts)

    train_dataset = IntentDataset(train_encodings, train_labels)
    val_dataset = IntentDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir=str(BASE_DIR / "training_outputs"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=RANDOM_STATE,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Запуск обучения...")
    trainer.train()

    print("Оценка на валидации...")
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = np.array(val_labels)

    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[id2label[i] for i in sorted(id2label.keys())],
            zero_division=0,
        )
    )

    print("Сохранение модели...")
    MODEL_DIR.mkdir(exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Модель сохранена в: {MODEL_DIR}")
    print(f"Словарь меток сохранён в: {LABEL_MAP_PATH}")


if __name__ == "__main__":
    main()
