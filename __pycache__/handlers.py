import requests

API_KEY = "0a1f27caccc790191ae7067b7e9458e3"


def handle_greeting():
    return "Здравствуйте! Чем могу помочь?"


def handle_farewell():
    return "До свидания!"


def handle_how_are_you():
    return "Спасибо, у меня всё хорошо."


def get_weather_by_city(city: str):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
        "lang": "ru"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
    except requests.RequestException as e:
        return f"Ошибка соединения с погодным сервисом: {e}"
    except ValueError:
        return "Сервис вернул не JSON."

    if response.status_code != 200:
        return f"Ошибка сервиса погоды: {data.get('message', 'неизвестная ошибка')}"

    temp = data["main"]["temp"]
    wind = data["wind"]["speed"]
    description = data["weather"][0]["description"]

    return (
        f"Температура: {round(temp, 1)} °C\n"
        f"Описание: {description}\n"
        f"Скорость ветра: {round(wind, 1)} м/с"
    )


def handle_weather(match):
    city = match.group(1).strip()
    return get_weather_by_city(city)


def handle_addition(match):
    a = float(match.group(1))
    b = float(match.group(2))
    return f"Результат: {a + b}"