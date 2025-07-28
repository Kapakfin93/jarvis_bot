# database.py - Signal Logging Database
import sqlite3
from config import DATABASE

def init_db():
    conn = sqlite3.connect(DATABASE['DB_NAME'])
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            timeframe TEXT,
            direction TEXT,
            score INTEGER,
            reason TEXT,
            entry_price REAL,
            sl_price REAL,
            tp1_price REAL,
            tp2_price REAL,
            is_valid INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def log_signal(signal_data):
    conn = sqlite3.connect(DATABASE['DB_NAME'])
    c = conn.cursor()
    c.execute('''
        INSERT INTO signals (timestamp, symbol, timeframe, direction, score, reason, entry_price, sl_price, tp1_price, tp2_price)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        signal_data['timestamp'],
        signal_data['symbol'],
        signal_data['timeframe'],
        signal_data['direction'],
        signal_data['score'],
        signal_data['reason'],
        signal_data['entry_price'],
        signal_data['sl_price'],
        signal_data['tp1_price'],
        signal_data['tp2_price']
    ))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    from config import DATABASE
    init_db()
    print("Database initialized successfully.")