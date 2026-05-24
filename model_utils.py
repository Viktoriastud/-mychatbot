from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "intent_model"
LABEL_MAP_PATH = BASE_DIR / "label_mapping.json"

_tokenizer = None
_model = None
_id2label = None


def load_artifacts():
    global _tokenizer, _model, _id2label

    if _tokenizer is None or _model is None or _id2label is None:
        if not MODEL_DIR.exists() or not LABEL_MAP_PATH.exists():
            return None, None, None

        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        _model.eval()

        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            label_data = json.load(f)
        _id2label = {int(k): v for k, v in label_data["id2label"].items()}

    return _tokenizer, _model, _id2label


def model_is_ready() -> bool:
    return MODEL_DIR.exists() and LABEL_MAP_PATH.exists()


def predict_intent(text: str) -> Tuple[str, float]:
    tokenizer, model, id2label = load_artifacts()
    if tokenizer is None or model is None or id2label is None:
        return "unknown", 0.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    intent = id2label.get(predicted_class, "unknown")
    return intent, confidence
