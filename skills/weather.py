from __future__ import annotations

from handlers import get_weather_by_city
from nlp_utils import extract_city


def weather_skill(text: str, user_id: int) -> str:
    city = extract_city(text)
    if city:
        return get_weather_by_city(city)

    raw = text.strip()
    if len(raw.split()) <= 3 and len(raw) >= 2:
        return get_weather_by_city(raw)

    return "WAIT_CITY"
