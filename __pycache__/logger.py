import sqlite3
from datetime import datetime

DB_PATH = "bot.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_message TEXT NOT NULL,
            bot_message TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_message(user_message: str, bot_message: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO logs (timestamp, user_message, bot_message) VALUES (?, ?, ?)",
        (datetime.now().isoformat(sep=" ", timespec="seconds"),
         user_message,
         bot_message)
    )
    conn.commit()
    conn.close()