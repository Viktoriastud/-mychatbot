from __future__ import annotations

import asyncio
import hashlib
import re
import subprocess
import sys
import threading
from pathlib import Path

from TTS.api import TTS


# Английская модель Coqui. Русский ею не читаем.
MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"
CACHE_DIR = Path("tts_cache")

_tts = None
_tts_lock = threading.RLock()
_async_lock = None


def has_russian_letters(text: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", text or ""))


def normalize_for_speech(text: str) -> str:
    text = text or ""

    text = re.sub(r"https?://\S+", " ссылка ", text)
    text = text.replace("°C", " градусов Цельсия")
    text = text.replace("км/ч", " километров в час")
    text = text.replace("м/с", " метров в секунду")
    text = text.replace("%", " процентов")

    text = re.sub(r"[*_`#>\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_cache_path(text: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)

    normalized = normalize_for_speech(text)
    text_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()

    return CACHE_DIR / f"{text_hash}.wav"


def get_tts():
    global _tts

    with _tts_lock:
        if _tts is None:
            print("[TTS] Загружаю английскую модель Coqui TTS...")
            _tts = TTS(
                model_name=MODEL_NAME,
                progress_bar=False,
                gpu=False,
            )
            print("[TTS] Английская модель Coqui загружена.")

        return _tts


def play_wav(path: Path):
    if sys.platform.startswith("win"):
        import winsound
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
    else:
        print(f"[TTS] WAV создан: {path}")


def speak_with_windows_voice(text: str):
    """
    Озвучка русских ответов через встроенный голос Windows.
    Работает без Coqui и без скачивания русской нейросетевой модели.
    """

    text = normalize_for_speech(text)

    if not text:
        return

    safe_text = text.replace("'", "''")

    command = f"""
Add-Type -AssemblyName System.Speech;
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer;
$speaker.Rate = 0;
$speaker.Volume = 100;
$speaker.Speak('{safe_text}');
"""

    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def speak_with_coqui(text: str):
    """
    Озвучка английских ответов через Coqui.
    """

    text_for_speech = normalize_for_speech(text)

    if not text_for_speech:
        return

    cache_path = get_cache_path(text_for_speech)

    with _tts_lock:
        if cache_path.exists():
            print(f"[TTS] Использую кэш: {cache_path}")
        else:
            print("[TTS] Генерирую английскую озвучку Coqui...")
            tts = get_tts()
            tts.tts_to_file(
                text=text_for_speech,
                file_path=str(cache_path),
            )
            print(f"[TTS] WAV сохранён: {cache_path}")

        play_wav(cache_path)


def speak_sync(text: str):
    text = normalize_for_speech(text)

    if not text:
        return

    if has_russian_letters(text):
        print("[TTS] Русский текст. Использую голос Windows.")
        speak_with_windows_voice(text)
        return

    print("[TTS] Английский текст. Использую Coqui.")
    speak_with_coqui(text)


def get_async_lock():
    global _async_lock

    if _async_lock is None:
        _async_lock = asyncio.Lock()

    return _async_lock


async def preload_tts_async():
    """
    Предзагрузка только английской модели Coqui.
    Русский голос Windows не требует загрузки.
    """

    try:
        await asyncio.to_thread(get_tts)
    except Exception as exc:
        print(f"[TTS] Coqui заранее не загрузился: {exc}")
        print("[TTS] Английская озвучка может быть недоступна, но русский голос Windows останется.")


async def speak_async(text: str):
    lock = get_async_lock()

    async with lock:
        await asyncio.to_thread(speak_sync, text)