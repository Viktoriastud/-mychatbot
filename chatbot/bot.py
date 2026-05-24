from __future__ import annotations

import string

from logger import init_db, log_message
from handlers import (
    get_weather_by_city,
    handle_farewell,
    handle_greeting,
    handle_how_are_you,
)
from nlp_utils import extract_city
from dialog_manager import DialogState, get_state, set_state
from patterns import patterns
from model_utils import model_is_ready, predict_intent


USER_ID = 1
CONFIDENCE_THRESHOLD = 0.45


def process_message(user_id, message: str):
    original_message = message
    message = message.strip().lower()

    punctuation_without_plus = string.punctuation.replace("+", "")
    normalized_message = message.translate(
        str.maketrans("", "", punctuation_without_plus)
    )

    state = get_state(user_id)

    # FSM: если бот ждёт город
    if state == DialogState.WAIT_CITY:
        city = extract_city(original_message)

        if city:
            set_state(user_id, DialogState.START)
            return get_weather_by_city(city)

        # если NLP не нашёл сущность, считаем, что пользователь просто ввёл город текстом
        city = original_message.strip()
        set_state(user_id, DialogState.START)
        return get_weather_by_city(city)

    # Математический пример лучше оставить как отдельный pattern
    for pattern, handler in patterns:
        match = pattern.search(normalized_message)
        if match:
            return handler(match)

    # ML-модель интентов
    if model_is_ready():
        intent, confidence = predict_intent(original_message)

        if confidence >= CONFIDENCE_THRESHOLD:
            if intent == "greeting":
                return handle_greeting()
            if intent == "how_are_you":
                return handle_how_are_you()
            if intent == "goodbye":
                return handle_farewell()
            if intent == "weather":
                city = extract_city(original_message)
                if city:
                    return get_weather_by_city(city)
                set_state(user_id, DialogState.WAIT_CITY)
                return "Укажите город"

    return "Я не понял запрос. Попробуйте переформулировать сообщение."


if __name__ == "__main__":
    init_db()
    print("Бот запущен. Введите сообщение:")

    if not model_is_ready():
        print(
            "Внимание: model.pkl и vectorizer.pkl не найдены. "
            "Сначала запустите train_model.py"
        )

    while True:
        user_input = input("Вы: ")
        response = process_message(USER_ID, user_input)
        print("Бот:", response)
        log_message(user_input, response)
