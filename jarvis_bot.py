# jarvis_bot.py - Main bot script (V8 - FINAL & ROBUST)

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
import pytz

# === UTILS ===
async def send_telegram(message):
    bot = Bot(token=API_KEYS['TELEGRAM_BOT_TOKEN'])
    await bot.send_message(chat_id=API_KEYS['TELEGRAM_CHAT_ID'], text=message, parse_mode='Markdown')

def get_last_atr(df):
    atr_val = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=BOT_PARAMETERS['ATR_PERIOD'])
    if atr_val.average_true_range().empty or atr_val.average_true_range().isnull().any():
        return 0.0 
    return atr_val.average_true_range().iloc[-1]

def calculate_vwap(df):
    df['volume'] = pd.to_numeric(df['volume'])
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_volume'] = df['tp'] * df['volume']
    
    if df['tp_volume'].isnull().all() or df['volume'].isnull().all():
        return 0.0

    df['cumulative_tp_volume'] = df['tp_volume'].cumsum()
    df['cumulative_volume'] = df['volume'].cumsum()
    
    vwap = df['cumulative_tp_volume'] / df['cumulative_volume']
    return vwap.iloc[-1] if not vwap.empty and not pd.isna(vwap.iloc[-1]) else 0.0

# --- FINAL FIX: Use Curl Subprocess ---
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
            error_msg = stderr.decode('utf-8').strip()
            if not error_msg: 
                error_msg = f"Curl failed with code {process.returncode}, no stderr msg. Possible network/API block."
            await send_telegram(f"⚠️ Curl command failed: {error_msg}")
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

# === Fetch Funding and OI from Hyperliquid ===
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
    if atr_higher == 0.0: return None, None, None 
    swing_order = max(5, int(atr_higher / df_higher['close'].mean() * 100)) 
    
    high_values = df_higher['high'].values
    low_values = df_higher['low'].values
    
    if len(high_values) < swing_order * 2 + 1: return None, None, None

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
    if atr == 0.0: return None, None, None
    impulsive_move_threshold = atr * BOT_PARAMETERS['IMPULSE_THRESHOLD']
    
    if len(df) < BOT_PARAMETERS['ATR_PERIOD']:
        return None, None, None

    for i in range(len(df) - 2, 0, -1):
        if df['close'].iloc[i] < df['open'].iloc[i]: 
            if (df['close'].iloc[i+1] - df['low'].iloc[i+1]) > impulsive_move_threshold and df['close'].iloc[i+1] > df['open'].iloc[i+1]:
                return "Bullish OB", df['high'].iloc[i], df['low'].iloc[i]
            
    for i in range(len(df) - 2, 0, -1):
        if df['close'].iloc[i] > df['open'].iloc[i]:
            if (df['high'].iloc[i+1] - df['close'].iloc[i+1]) > impulsive_move_threshold and df['close'].iloc[i+1] < df['open'].iloc[i+1]:
                return "Bearish OB", df['high'].iloc[i], df['low'].iloc[i]
                
    return None, None, None

# === Fair Value Gap Detector (V3) ===
def detect_fvg(df):
    fvgs = []
    if len(df) < 3: return []

    for i in range(len(df) - 2):
        if df['low'].iloc[i+2] > df['high'].iloc[i]:
            if df['high'].iloc[i+1] < df['low'].iloc[i] or df['low'].iloc[i+1] > df['high'].iloc[i+2]:
                continue 
            fvgs.append(("Bearish FVG", df['high'].iloc[i], df['low'].iloc[i+2]))
        
        if df['high'].iloc[i+2] < df['low'].iloc[i]:
            if df['low'].iloc[i+1] > df['high'].iloc[i] or df['high'].iloc[i+1] < df['low'].iloc[i+2]:
                continue
            fvgs.append(("Bullish FVG", df['low'].iloc[i], df['high'].iloc[i+2]))
    return fvgs

# === Orderbook Liquidity Pool Detector ===
def detect_dom_liquidity(order_book, current_price):
    if not order_book or 'bids' not in order_book or 'asks' not in order_book: return None, None
    
    bids = pd.DataFrame(order_book['bids'], columns=['price', 'size'])
    asks = pd.DataFrame(order_book['asks'], columns=['price', 'size'])
    bids['price'] = pd.to_numeric(bids['price'])
    asks['price'] = pd.to_numeric(asks['price'])
    bids['size'] = pd.to_numeric(bids['size'])
    asks['size'] = pd.to_numeric(asks['size'])

    range_percent = 0.005 # 0.5% range around current price
    
    liquidity_above = asks[(asks['price'] > current_price) & (asks['price'] <= current_price * (1 + range_percent))]['size'].sum()
    liquidity_below = bids[(bids['price'] < current_price) & (bids['price'] >= current_price * (1 - range_percent))]['size'].sum()

    significant_liquidity_threshold = BOT_PARAMETERS['OI_THRESHOLD'] / 1000 

    if liquidity_above > significant_liquidity_threshold and liquidity_above > liquidity_below * 1.5: 
        return "SHORT", asks['price'].iloc[0] 
    if liquidity_below > significant_liquidity_threshold and liquidity_below > liquidity_above * 1.5: 
        return "LONG", bids['price'].iloc[0] 
    return None, None

# === Multi-Timeframe Confirmation ===
def multi_tf_confirmation(df_higher):
    if len(df_higher) < BOT_PARAMETERS['EMA_PERIOD']: return "Neutral"
    ema = ta.trend.EMAIndicator(df_higher['close'], window=BOT_PARAMETERS['EMA_PERIOD'])
    if ema.ema_indicator().iloc[-1] is None or pd.isna(ema.ema_indicator().iloc[-1]):
        return "Neutral"
    return "Bullish" if df_higher['close'].iloc[-1] > ema.ema_indicator().iloc[-1] else "Bearish"

# === NEW: Smart Time Filter ===
import pytz
def is_trading_session_active():
    now_utc = datetime.now(pytz.utc) 
    
    if (now_utc.hour >= 0 and now_utc.hour < 9) or \
       (now_utc.hour >= 13 and now_utc.hour < 22):
        return True
    return False

# NEW: Chop Zone Detector
def detect_chop_zone(df_lower):
    if len(df_lower) < BOT_PARAMETERS['ATR_PERIOD'] + 20: return False
    
    last_atr = get_last_atr(df_lower)
    if last_atr == 0.0: return True 

    recent_atrs = ta.volatility.AverageTrueRange(df_lower['high'], df_lower['low'], df_lower['close'], window=BOT_PARAMETERS['ATR_PERIOD']).average_true_range().iloc[-20:].dropna()
    if recent_atrs.empty: return False 
    
    avg_recent_atr = recent_atrs.mean()
    
    if avg_recent_atr == 0: return False 
    
    if last_atr < avg_recent_atr * BOT_PARAMETERS['CHOP_ZONE_ATR_MULTIPLIER']:
        range_lookback_candles = 20
        recent_high = df_lower['high'].iloc[-range_lookback_candles:].max()
        recent_low = df_lower['low'].iloc[-range_lookback_candles:].min()
        
        if (recent_high - recent_low) < last_atr * (BOT_PARAMETERS['CHOP_ZONE_ATR_MULTIPLIER'] * 1.5):
            avg_volume = df_lower['volume'].iloc[-range_lookback_candles:].mean()
            if df_lower['volume'].iloc[-1] < avg_volume * 0.7:
                return True
            
    return False

# NEW: Position Size Dynamic
def calculate_suggested_position_size(score, current_balance_usd, current_price):
    if current_balance_usd <= 0: return BOT_PARAMETERS['MIN_POSITION_SIZE_USD'] / current_price 

    confidence_factor = score / 10.0
    if confidence_factor > 1: confidence_factor = 1.0 
    
    risk_percentage_of_balance = BOT_PARAMETERS['MAX_POSITION_SIZE_PERCENT'] 

    risk_amount_usd = current_balance_usd * risk_percentage_of_balance * confidence_factor
    
    position_size_coins = risk_amount_usd / current_price
    
    if position_size_coins * current_price < BOT_PARAMETERS['MIN_POSITION_SIZE_USD']:
        position_size_coins = BOT_PARAMETERS['MIN_POSITION_SIZE_USD'] / current_price
        
    return position_size_coins

# === Signal Aggregation & Filtering ===
async def aggregate_signals(df_lower, df_higher, oi, fr, order_book):
    current_price = df_lower['close'].iloc[-1]
    trend = multi_tf_confirmation(df_higher)
    direction, reason, sweep_level = detect_liquidity_sweep(df_lower, df_higher)
    ob_type, ob_high, ob_low = detect_order_block(df_lower)
    fvgs = detect_fvg(df_lower)
    dom_direction, dom_level = detect_dom_liquidity(order_book, current_price)
    vwap_val = calculate_vwap(df_lower)
    
    if detect_chop_zone(df_lower):
        return None, None, [], 0, None, None, [], None, None, None

    score = 0
    reasons = []
    
    if direction:
        score += 3
        reasons.append(f"Primary: Liquidity Sweep at {sweep_level:.2f}")

    if direction and (direction.upper() == trend.upper()):
        score += 2
        reasons.append(f"Trend: H4 matches ({trend})")

    if ob_type and ((direction == "LONG" and "Bullish" in ob_type and current_price >= ob_low and current_price <= ob_high) or
                    (direction == "SHORT" and "Bearish" in ob_type and current_price <= ob_high and current_price >= ob_low)):
        score += 1
        reasons.append(f"Structure: Price near {ob_type} ({ob_low:.2f}-{ob_high:.2f})")
        
    last_atr = get_last_atr(df_lower)
    if fvgs:
        if direction == "LONG" and any(f[0] == "Bullish FVG" and f[2] > current_price and abs(f[1] - current_price) < last_atr * 3 for f in fvgs):
            score += 1
            reasons.append("Structure: Bullish FVG detected")
        elif direction == "SHORT" and any(f[0] == "Bearish FVG" and f[1] < current_price and abs(f[2] - current_price) < last_atr * 3 for f in fvgs):
            score += 1
            reasons.append("Structure: Bearish FVG detected")

    if dom_direction and dom_direction == direction:
        score += 1
        reasons.append(f"Liquidity: DOM shows pool at {dom_level:.2f}")

    if (direction == "LONG" and fr > 0.01 and oi > BOT_PARAMETERS['OI_THRESHOLD']) or \
       (direction == "SHORT" and fr < -0.01 and oi > BOT_PARAMETERS['OI_THRESHOLD']):
        score += 1
        reasons.append("Sentiment: High OI & favorable FR")
    
    if vwap_val != 0.0:
        if direction == "LONG" and current_price < vwap_val:
            score += 1
            reasons.append(f"VWAP: Price below VWAP ({vwap_val:.2f})")
        elif direction == "SHORT" and current_price > vwap_val:
            score += 1
            reasons.append(f"VWAP: Price above VWAP ({vwap_val:.2f})")
    
    if score >= BOT_PARAMETERS['SCORE_THRESHOLD']:
        return direction, reason, reasons, score, ob_high, ob_low, fvgs, sweep_level, vwap_val
    
    return None, None, [], 0, None, None, [], None, None, None

# === Dynamic Risk Management ===
def dynamic_risk_management(direction, df_lower, ob_high, ob_low, fvgs, sweep_level, vwap_val, score):
    last_atr = get_last_atr(df_lower)
    current_price = df_lower['close'].iloc[-1]
    
    if last_atr == 0.0: 
        last_atr = (df_lower['high'].iloc[-1] - df_lower['low'].iloc[-1]) / 2 
        if last_atr == 0: last_atr = 0.5 

    if direction == "SHORT":
        sl = sweep_level * 1.0005 + last_atr * 0.5
        
        tp_options = []
        if vwap_val < current_price: tp_options.append(vwap_val)
        tp_options.extend([f[2] for f in fvgs if f[0] == "Bullish FVG" and f[2] < current_price])
        if ob_low and ob_low < current_price: tp_options.append(ob_low)
        
        tp_options = sorted([tp for tp in tp_options if tp < current_price and tp > (current_price - last_atr * 5)], reverse=True) 

        tp1 = tp_options[0] if tp_options else current_price - last_atr * 2
        tp2 = tp_options[1] if len(tp_options) > 1 else current_price - last_atr * 3
        tp3 = tp_options[2] if len(tp_options) > 2 else current_price - last_atr * 4
        
        tp1 = min(tp1, current_price - last_atr * 0.5) 
        tp2 = min(tp1 - last_atr, tp2) 
        tp3 = min(tp2 - last_atr, tp3)

    elif direction == "LONG":
        sl = sweep_level * 0.9995 - last_atr * 0.5
        
        tp_options = []
        if vwap_val > current_price: tp_options.append(vwap_val)
        tp_options.extend([f[1] for f in fvgs if f[0] == "Bearish FVG" and f[1] > current_price])
        if ob_high and ob_high > current_price: tp_options.append(ob_high)
        
        tp_options = sorted([tp for tp in tp_options if tp > current_price and tp < (current_price + last_atr * 5)])

        tp1 = tp_options[0] if tp_options else current_price + last_atr * 2
        tp2 = tp_options[1] if len(tp_options) > 1 else current_price + last_atr * 3
        tp3 = tp_options[2] if len(tp_options) > 2 else current_price + last_atr * 4
        
        tp1 = max(tp1, current_price + last_atr * 0.5) 
        tp2 = max(tp1 + last_atr, tp2) 
        tp3 = max(tp2 + last_atr, tp3)
        
    else:
        return None, None, None, None
    
    if (direction == "LONG" and sl >= current_price) or \
       (direction == "SHORT" and sl <= current_price):
        sl = current_price - last_atr * (1.5 if direction == "LONG" else -1.5)
    
    return sl, tp1, tp2, tp3

# NEW: Calculate Position Size Dynamic
def calculate_suggested_position_size(score, current_balance_usd, current_price):
    if current_balance_usd <= 0: return BOT_PARAMETERS['MIN_POSITION_SIZE_USD'] / current_price 

    confidence_factor = score / 10.0
    if confidence_factor > 1: confidence_factor = 1.0 
    
    risk_percentage_of_balance = BOT_PARAMETERS['MAX_POSITION_SIZE_PERCENT'] 

    risk_amount_usd = current_balance_usd * risk_percentage_of_balance * confidence_factor
    
    position_size_coins = risk_amount_usd / current_price
    
    if position_size_coins * current_price < BOT_PARAMETERS['MIN_POSITION_SIZE_USD']:
        position_size_coins = BOT_PARAMETERS['MIN_POSITION_SIZE_USD'] / current_price
        
    return position_size_coins

# === Main Analysis and Alert ===
async def analyze_market():
    df_lower = await fetch_hyperliquid_candles(BOT_PARAMETERS['SYMBOL'], BOT_PARAMETERS['LOWER_TIMEFRAME'], BOT_PARAMETERS['CANDLE_LIMIT'])
    df_higher = await fetch_hyperliquid_candles(BOT_PARAMETERS['SYMBOL'], BOT_PARAMETERS['HIGHER_TIMEFRAME'], BOT_PARAMETERS['CANDLE_LIMIT'])
    if df_lower.empty or df_higher.empty:
        return

    oi, fr = await fetch_hyperliquid_funding_oi(BOT_PARAMETERS['SYMBOL'])
    order_book = await fetch_order_book(BOT_PARAMETERS['SYMBOL'])

    if detect_chop_zone(df_lower):
        return None, None, [], 0, None, None, [], None, None, None

    direction, reason, reasons, score, ob_high, ob_low, fvgs, sweep_level, vwap_val = await aggregate_signals(df_lower, df_higher, oi, fr, order_book)

    if direction:
        sl, tp1, tp2, tp3 = dynamic_risk_management(direction, df_lower, ob_high, ob_low, fvgs, sweep_level, vwap_val, score)
        entry_price = df_lower['close'].iloc[-1]
        
        suggested_position_size = calculate_suggested_position_size(score, 1000, entry_price) 

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
            'tp2_price': tp2,
            'tp3_price': tp3
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
**Suggested TP3:** {tp3:.2f}
**Suggested Position Size:** {suggested_position_size:.4f} {BOT_PARAMETERS['SYMBOL'].split('/')[0]} (approx ${suggested_position_size * entry_price:.2f} USD)
"""
        await send_telegram(alert)

# === Bot Loop ===
async def bot_loop():
    while True:
        await analyze_market()
        await asyncio.sleep(60 * 15)

# === Start Bot ===
async def start_bot():
    await send_telegram("✅ Parasitic Passenger Bot Started! (V8)")
    await bot_loop()

if __name__ == "__main__":
    from database import init_db
    init_db()
    asyncio.run(start_bot())
