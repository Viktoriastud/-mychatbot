from __future__ import annotations

from datetime import datetime


def date_skill() -> str:
    return f"Сегодня {datetime.now().strftime('%d.%m.%Y')}"
