# parasitic_passenger_bot.py - Main bot script (V5 - Final)

import requests
import json
import asyncio
from datetime import datetime, timedelta
from telegram import Bot
import ta
import pandas as pd
import numpy as np
import scipy.signal as signal
import time
import subprocess
from database import log_signal
from config import API_KEYS, BOT_PARAMETERS

# === UTILS ===
async def send_telegram(message):
    bot = Bot(token=API_KEYS['TELEGRAM_BOT_TOKEN'])
    await bot.send_message(chat_id=API_KEYS['TELEGRAM_CHAT_ID'], text=message, parse_mode='Markdown')

def get_last_atr(df):
    atr_val = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=BOT_PARAMETERS['ATR_PERIOD'])
    return atr_val.average_true_range().iloc[-1]

async def run_curl_command(url, payload):
    json_payload = json.dumps(payload)
    command = [
        'curl',
        '-X', 'POST',
        '-H', 'Content-Type: application/json',
        '--data', json_payload,
        url
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            return json.loads(stdout)
        else:
            await send_telegram(f"⚠️ Curl command failed: {stderr.decode('utf-8')}")
            return None
    except Exception as e:
        await send_telegram(f"⚠️ Error running curl: {e}")
        return None

# === Fetch Data from Hyperliquid ===
async def fetch_hyperliquid_candles(symbol, timeframe, limit):
    url = "https://api.hyperliquid.xyz/info"
    current_time = int(datetime.now().timestamp() * 1000)
    interval_ms = {
        'M15': 900000,
        'H4': 14400000
    }.get(timeframe, 900000)
    start_time = current_time - (limit * interval_ms)
    data = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol.split('/')[0],
            "interval": timeframe.lower(),
            "startTime": start_time,
            "endTime": current_time
        }
    }
    candles = await run_curl_command(url, data)
    if candles:
        df = pd.DataFrame(candles, columns=["t", "o", "h", "l", "c", "v"])
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        return df
    return pd.DataFrame()

# === NEW: Fetch Funding and OI from Hyperliquid ===
async def fetch_hyperliquid_funding_oi(symbol):
    url = "https://api.hyperliquid.xyz/info"
    data = {
        "type": "metaAndAssetCtxs",
        "req": {"coin": symbol.split('/')[0]}
    }
    response = await run_curl_command(url, data)
    if response and 'assetCtxs' in response:
        ctx = response['assetCtxs'][0]
        oi = float(ctx['openInterest'])
        fr = float(ctx['fundingRate'])
        return oi, fr
    return 0, 0

# === Fetch Order Book from Hyperliquid ===
async def fetch_order_book(symbol):
    url = "https://api.hyperliquid.xyz/info"
    data = {
        "type": "orderbookSnapshot",
        "req": {"coin": symbol.split('/')[0]}
    }
    book = await run_curl_command(url, data)
    return book if book else {}

# === Liquidity Sweep Detector (V3) ===
def detect_liquidity_sweep(df_lower, df_higher):
    atr_higher = get_last_atr(df_higher)
    swing_order = max(5, int(atr_higher / df_higher['close'].mean() * 100))
    
    high_values = df_higher['high'].values
    low_values = df_higher['low'].values
    swing_high_indices = signal.argrelextrema(high_values, np.greater, order=swing_order)[0]
    swing_low_indices = signal.argrelextrema(low_values, np.less, order=swing_order)[0]
    
    recent_swing_high = high_values[swing_high_indices[-1]] if len(swing_high_indices) > 0 else df_higher['high'].max()
    recent_swing_low = low_values[swing_low_indices[-1]] if len(swing_low_indices) > 0 else df_higher['low'].min()
    
    last_candle_high = df_lower['high'].iloc[-1]
    last_candle_low = df_lower['low'].iloc[-1]
    last_candle_close = df_lower['close'].iloc[-1]
    
    if last_candle_high > recent_swing_high and last_candle_close < recent_swing_high:
        return "SHORT", "Liquidity Sweep at High", recent_swing_high
    if last_candle_low < recent_swing_low and last_candle_close > recent_swing_low:
        return "LONG", "Liquidity Sweep at Low", recent_swing_low
    return None, None, None

# === Order Block Analyzer (V3) ===
def detect_order_block(df):
    atr = get_last_atr(df)
    impulsive_move = atr * BOT_PARAMETERS['IMPULSE_THRESHOLD']
    
    down_candles = df[df['close'] < df['open']]
    if not down_candles.empty:
        last_down_candle = down_candles.iloc[-1]
        next_candle_index = df.index.get_loc(last_down_candle.name) + 1
        if next_candle_index < len(df) and (df['close'].iloc[next_candle_index] - df['low'].iloc[next_candle_index]) > impulsive_move:
            return "Bullish OB", last_down_candle['high'], last_down_candle['low']
            
    up_candles = df[df['close'] > df['open']]
    if not up_candles.empty:
        last_up_candle = up_candles.iloc[-1]
        next_candle_index = df.index.get_loc(last_up_candle.name) + 1
        if next_candle_index < len(df) and (df['high'].iloc[next_candle_index] - df['close'].iloc[next_candle_index]) > impulsive_move:
            return "Bearish OB", last_up_candle['high'], last_up_candle['low']
            
    return None, None, None

# === Fair Value Gap Detector (V3) ===
def detect_fvg(df):
    fvgs = []
    for i in range(len(df) - 2):
        if df['low'].iloc[i+2] > df['high'].iloc[i]:
            fvgs.append(("Bearish FVG", df['high'].iloc[i], df['low'].iloc[i+2]))
        if df['high'].iloc[i+2] < df['low'].iloc[i]:
            fvgs.append(("Bullish FVG", df['low'].iloc[i+2], df['high'].iloc[i]))
    return fvgs

# === Orderbook Liquidity Pool Detector ===
def detect_dom_liquidity(order_book, current_price):
    if not order_book: return None, None
    bids = pd.DataFrame(order_book['bids'], columns=['price', 'size'])
    asks = pd.DataFrame(order_book['asks'], columns=['price', 'size'])
    bids['price'] = bids['price'].astype(float)
    asks['price'] = asks['price'].astype(float)
    
    liquidity_above = asks[(asks['price'] > current_price) & (asks['price'] <= current_price * 1.01)].nlargest(1, 'size')
    liquidity_below = bids[(bids['price'] < current_price) & (bids['price'] >= current_price * 0.99)].nlargest(1, 'size')

    if not liquidity_above.empty: return "SHORT", liquidity_above['price'].iloc[0]
    if not liquidity_below.empty: return "LONG", liquidity_below['price'].iloc[0]
    return None, None

# === Multi-Timeframe Confirmation ===
def multi_tf_confirmation(df_higher):
    ema = ta.trend.EMAIndicator(df_higher['close'], window=BOT_PARAMETERS['EMA_PERIOD'])
    return "Bullish" if df_higher['close'].iloc[-1] > ema.ema_indicator().iloc[-1] else "Bearish"

# === Signal Aggregation & Filtering ===
async def aggregate_signals(df_lower, df_higher, oi, fr, order_book):
    current_price = df_lower['close'].iloc[-1]
    trend = multi_tf_confirmation(df_higher)
    direction, reason, sweep_level = detect_liquidity_sweep(df_lower, df_higher)
    ob_type, ob_high, ob_low = detect_order_block(df_lower)
    fvgs = detect_fvg(df_lower)
    dom_direction, dom_level = detect_dom_liquidity(order_book, current_price)

    score = 0
    reasons = []
    
    if direction:
        score += 3
        reasons.append(f"Liquidity Sweep detected at {sweep_level:.2f}")

    if direction and (direction.upper() == trend.upper()):
        score += 2
        reasons.append(f"H4 Trend matches ({trend})")

    if ob_type and ((direction == "LONG" and "Bullish" in ob_type) or (direction == "SHORT" and "Bearish" in ob_type)):
        score += 1
        reasons.append(f"Price at {ob_type} ({ob_low:.2f}-{ob_high:.2f})")
        
    if fvgs and ((direction == "LONG" and any(f[0] == "Bullish FVG" and f[2] > current_price for f in fvgs)) or
                  (direction == "SHORT" and any(f[0] == "Bearish FVG" and f[1] < current_price for f in fvgs))):
        score += 1
        reasons.append("Unfilled FVG detected")

    if dom_direction and dom_direction == direction:
        score += 1
        reasons.append(f"DOM shows liquidity pool at {dom_level:.2f}")

    if abs(fr) > 0.01 and oi > BOT_PARAMETERS['OI_THRESHOLD']:
        score += 1
        reasons.append("High OI and funding rate")

    if score >= BOT_PARAMETERS['SCORE_THRESHOLD']:
        return direction, reason, reasons, score, ob_high, ob_low, fvgs, sweep_level
    
    return None, None, [], 0, None, None, [], None

# === Dynamic Risk Management ===
def dynamic_risk_management(direction, df_lower, ob_high, ob_low, fvgs, sweep_level):
    last_atr = get_last_atr(df_lower)
    
    if direction == "SHORT":
        sl = sweep_level * 1.001
        
        tp_levels = sorted([f[2] for f in fvgs if f[0] == "Bullish FVG"] + [ob_low if ob_low else 0], reverse=True)
        tp1 = tp_levels[0] if tp_levels else df_lower['close'].iloc[-1] - last_atr * 2
        tp2 = tp_levels[1] if len(tp_levels) > 1 else tp1 - last_atr * 3
        
    elif direction == "LONG":
        sl = sweep_level * 0.999
        
        tp_levels = sorted([f[1] for f in fvgs if f[0] == "Bearish FVG"] + [ob_high if ob_high else 0])
        tp1 = tp_levels[0] if tp_levels else df_lower['close'].iloc[-1] + last_atr * 2
        tp2 = tp_levels[1] if len(tp_levels) > 1 else tp1 + last_atr * 3
        
    else:
        return None, None, None
    
    return sl, tp1, tp2

# === Main Analysis and Alert ===
async def analyze_market():
    df_lower = await fetch_hyperliquid_candles(BOT_PARAMETERS['SYMBOL'], BOT_PARAMETERS['LOWER_TIMEFRAME'], BOT_PARAMETERS['CANDLE_LIMIT'])
    df_higher = await fetch_hyperliquid_candles(BOT_PARAMETERS['SYMBOL'], BOT_PARAMETERS['HIGHER_TIMEFRAME'], BOT_PARAMETERS['CANDLE_LIMIT'])
    if df_lower.empty or df_higher.empty:
        return

    oi, fr = await fetch_hyperliquid_funding_oi(BOT_PARAMETERS['SYMBOL'])
    order_book = await fetch_order_book(BOT_PARAMETERS['SYMBOL'])

    direction, reason, reasons, score, ob_high, ob_low, fvgs, sweep_level = await aggregate_signals(df_lower, df_higher, oi, fr, order_book)

    if direction:
        sl, tp1, tp2 = dynamic_risk_management(direction, df_lower, ob_high, ob_low, fvgs, sweep_level)
        entry_price = df_lower['close'].iloc[-1]
        
        signal_data = {
            'timestamp': str(datetime.now()),
            'symbol': BOT_PARAMETERS['SYMBOL'],
            'timeframe': BOT_PARAMETERS['LOWER_TIMEFRAME'],
            'direction': direction,
            'score': score,
            'reason': reason,
            'entry_price': entry_price,
            'sl_price': sl,
            'tp1_price': tp1,
            'tp2_price': tp2
        }
        await log_signal(signal_data)

        alert = f"""
🚨 **HIGH-PROBABILITY [{direction}] SIGNAL** 🚨
------------------------------------------
**Symbol:** {BOT_PARAMETERS['SYMBOL']} (Hyperliquid)
**Timeframe:** {BOT_PARAMETERS['LOWER_TIMEFRAME']}
**Signal Score:** {score}
**Confirmation:** {', '.join(reasons)}
**Suggested Entry:** {entry_price:.2f}
**Suggested SL:** {sl:.2f}
**Suggested TP1:** {tp1:.2f}
**Suggested TP2:** {tp2:.2f}
"""
        await send_telegram(alert)

# === Bot Loop ===
async def bot_loop():
    while True:
        await analyze_market()
        await asyncio.sleep(60 * 15)

# === Start Bot ===
async def start_bot():
    await send_telegram("✅ Parasitic Passenger Bot Started!")
    await bot_loop()

if __name__ == "__main__":
    from database import init_db
    init_db()
    asyncio.run(start_bot())