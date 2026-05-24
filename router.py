from __future__ import annotations

from dialog_manager import DialogState, set_state
from skills.weather import weather_skill
from skills.time_skill import time_skill
from skills.date_skill import date_skill
from skills.greeting_skill import greeting_skill
from skills.help_skill import help_skill
from skills.smalltalk_skill import smalltalk_skill
from skills.goodbye_skill import goodbye_skill


def fallback_skill() -> str:
    return "Я пока не умею обрабатывать такой запрос."


def route_intent(intent: str, text: str, user_id: int) -> str:
    if intent == "weather":
        result = weather_skill(text, user_id)
        if result == "WAIT_CITY":
            set_state(user_id, DialogState.WAIT_CITY)
            return "В каком городе вас интересует погода?"
        return result

    if intent == "time":
        return time_skill()
    if intent == "date":
        return date_skill()
    if intent == "greeting":
        return greeting_skill()
    if intent == "help":
        return help_skill()
    if intent == "smalltalk":
        return smalltalk_skill()
    if intent == "goodbye":
        return goodbye_skill()

    return fallback_skill()
