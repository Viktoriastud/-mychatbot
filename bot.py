from __future__ import annotations

import asyncio
import re

from dialog_manager import DialogState, get_state, set_state
from model_utils import model_is_ready, predict_intent
from nlp_utils import extract_city
from router import route_intent
from tts_engine import preload_tts_async, speak_async
from voice import listen


USER_ID = 1
CONFIDENCE_THRESHOLD = 0.30


def detect_math_expression(text: str):
    match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*", text)
    if not match:
        return None

    a = float(match.group(1))
    op = match.group(2)
    b = float(match.group(3))

    try:
        if op == "+":
            result = a + b
        elif op == "-":
            result = a - b
        elif op == "*":
            result = a * b
        elif op == "/":
            if b == 0:
                return "На ноль делить нельзя."
            result = a / b
        else:
            return None
    except Exception:
        return "Не удалось вычислить выражение."

    if result.is_integer():
        return str(int(result))

    return str(round(result, 4))


def normalize_text(text: str) -> str:
    return re.sub(r"[^\w\sа-яё-]+", "", text.lower()).strip()


def handle_message(user_id: int, text: str) -> str:
    text = (text or "").strip()
    normalized_text = normalize_text(text)

    if not text:
        return "Введите сообщение."

    greeting_words = {
        "привет",
        "здравствуй",
        "здравствуйте",
        "добрый день",
        "добрый вечер",
        "доброе утро",
        "хай",
        "hello",
        "hi",
    }

    goodbye_words = {
        "пока",
        "до свидания",
        "прощай",
        "увидимся",
        "bye",
    }

    smalltalk_words = {
        "как дела",
        "как ты",
        "как поживаешь",
        "что нового",
        "как жизнь",
        "ты любишь свою работу",
    }

    time_words = {
        "сколько времени",
        "который час",
        "сколько сейчас времени",
        "скажи время",
        "текущее время",
        "скажи текущее время",
        "какое сейчас время",
        "покажи время",
        "сколько времени сейчас",
        "скажи который час",
        "мне нужно узнать время",
        "сколько там времени",
        "как узнать время",
        "можно узнать время",
        "сколько уже времени",
    }

    date_words = {
        "какое сегодня число",
        "какая сегодня дата",
        "скажи дату",
        "сегодняшняя дата",
        "какое завтра число",
        "какая дата завтра",
        "какая дата будет завтра",
        "что за число завтра",
        "что за дата завтра",
    }

    help_words = {
        "что ты умеешь",
        "что ты умеешь делать",
        "что ты можешь",
        "что ты можешь делать",
        "чем ты можешь помочь",
        "помощь",
        "help",
    }

    weather_hint_words = {
        "зонт",
        "дождь",
        "осадки",
        "ливень",
        "снег",
        "погода",
        "холодно",
        "облачно",
        "ветер",
    }

    interrupt_words = (
        greeting_words
        | goodbye_words
        | smalltalk_words
        | time_words
        | date_words
        | help_words
    )

    current_state = get_state(user_id)

    if current_state == DialogState.WAIT_CITY:
        if normalized_text in interrupt_words or len(normalized_text.split()) > 3:
            set_state(user_id, DialogState.START)
            return handle_message(user_id, text)

        set_state(user_id, DialogState.START)
        return route_intent("weather", text, user_id)

    math_result = detect_math_expression(text)
    if math_result is not None:
        return f"Результат: {math_result}"

    if normalized_text in greeting_words:
        return route_intent("greeting", text, user_id)

    if normalized_text in goodbye_words:
        return route_intent("goodbye", text, user_id)

    if normalized_text in smalltalk_words:
        return route_intent("smalltalk", text, user_id)

    if normalized_text in time_words:
        return route_intent("time", text, user_id)

    if normalized_text in date_words:
        return route_intent("date", text, user_id)

    if normalized_text in help_words:
        return route_intent("help", text, user_id)

    if any(word in normalized_text for word in weather_hint_words):
        city = extract_city(text)

        if city:
            return route_intent("weather", city, user_id)

        set_state(user_id, DialogState.WAIT_CITY)
        return "В каком городе вас интересует погода?"

    intent, confidence = predict_intent(text)
    print(f"[DEBUG] intent={intent}, confidence={confidence:.4f}")

    if confidence < CONFIDENCE_THRESHOLD:
        return "Я не понял запрос. Попробуйте переформулировать сообщение."

    if intent == "weather":
        city = extract_city(text)

        if city:
            return route_intent("weather", city, user_id)

        set_state(user_id, DialogState.WAIT_CITY)
        return "В каком городе вас интересует погода?"

    return route_intent(intent, text, user_id)


async def read_console_input(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


async def safe_speak(text: str):
    try:
        await speak_async(text)
    except Exception as exc:
        print(f"[TTS] Ошибка озвучивания: {exc}")


async def get_voice_text() -> str:
    try:
        return await asyncio.to_thread(listen)
    except Exception as exc:
        print(f"[ASR] Ошибка распознавания речи: {exc}")
        return ""


async def main_async():
    print("Бот запущен.")

    if not model_is_ready():
        print("Внимание: fine-tuned BERT модель не найдена. Сначала запустите train_model.py")

    try:
        await preload_tts_async()
    except Exception as exc:
        print(f"[TTS] Предварительная загрузка не удалась: {exc}")
        print("[TTS] Бот продолжит работу.")

    mode = "voice"

    print()
    print("Режимы работы:")
    print("  Enter  — записать голос")
    print("  /text  — перейти в текстовый ввод")
    print("  /voice — перейти в голосовой ввод")
    print("  /exit  — выйти")

    while True:
        try:
            if mode == "voice":
                command = (
                    await read_console_input("\nНажмите Enter и говорите, либо введите команду: ")
                ).strip()

                if command == "/exit":
                    print("Бот завершает работу.")
                    break

                if command == "/text":
                    mode = "text"
                    print("Включён текстовый ввод.")
                    continue

                if command == "/voice":
                    print("Голосовой ввод уже включён.")
                    continue

                user_text = await get_voice_text()

            else:
                user_text = (await read_console_input("\nВы: ")).strip()

                if user_text == "/exit":
                    print("Бот завершает работу.")
                    break

                if user_text == "/voice":
                    mode = "voice"
                    print("Включён голосовой ввод.")
                    continue

                if user_text == "/text":
                    print("Текстовый ввод уже включён.")
                    continue

        except (KeyboardInterrupt, EOFError):
            print("\nБот завершает работу.")
            break

        if not user_text:
            print("[ASR] Текст не распознан. Попробуйте ещё раз.")
            continue

        print("Вы:", user_text)

        response = handle_message(USER_ID, user_text)
        print("Бот:", response)

        await safe_speak(response)


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nБот завершает работу.")


if __name__ == "__main__":
    main()