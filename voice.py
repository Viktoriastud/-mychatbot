from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write


def prepare_ffmpeg_for_whisper():
    """
    OpenAI Whisper ищет команду ffmpeg в PATH.
    Если системного ffmpeg нет, берём ffmpeg из imageio-ffmpeg,
    копируем его в папку проекта как ffmpeg.exe и добавляем папку проекта в PATH.
    """

    try:
        source_ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
        local_ffmpeg = Path(__file__).resolve().parent / "ffmpeg.exe"

        if not local_ffmpeg.exists():
            shutil.copy2(source_ffmpeg, local_ffmpeg)

        os.environ["PATH"] = str(local_ffmpeg.parent) + os.pathsep + os.environ.get("PATH", "")
        os.environ["IMAGEIO_FFMPEG_EXE"] = str(local_ffmpeg)

        print(f"[ASR] FFmpeg подключен: {local_ffmpeg}")

    except Exception as exc:
        print(f"[ASR] Не удалось подключить FFmpeg: {exc}")


prepare_ffmpeg_for_whisper()

import whisper


WHISPER_MODEL_NAME = "base"
INPUT_WAV_PATH = Path("input.wav")
SAMPLE_RATE = 16000
RECORD_SECONDS = 5

_whisper_model = None


def get_whisper_model():
    global _whisper_model

    if _whisper_model is None:
        print(f"[ASR] Загружаю Whisper модель: {WHISPER_MODEL_NAME}")
        _whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
        print("[ASR] Whisper модель загружена.")

    return _whisper_model


def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("ё", "е").replace("Ё", "Е")

    # Оставляем русские/английские буквы, цифры, пробелы и математические знаки
    text = re.sub(r"[^А-Яа-яA-Za-z0-9\s+\-*/]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def record_audio(
    filename: str | Path = INPUT_WAV_PATH,
    seconds: int = RECORD_SECONDS,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    filename = Path(filename)

    print(f"[ASR] Говорите. Запись: {seconds} секунд...")

    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )

    sd.wait()

    audio = np.squeeze(audio)
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    write(str(filename), sample_rate, audio_int16)

    print("[ASR] Запись завершена.")

    return filename


def speech_to_text(filename: str | Path = INPUT_WAV_PATH) -> str:
    filename = Path(filename)

    if not filename.exists():
        return ""

    model = get_whisper_model()

    print("[ASR] Распознаю речь...")

    result = model.transcribe(
        str(filename),
        language="ru",
        fp16=False,
    )

    text = result.get("text", "")
    text = clean_text(text)

    print(f"[ASR] Распознано: {text}")

    return text


def listen() -> str:
    record_audio()
    text = speech_to_text()
    return text