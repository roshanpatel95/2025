# stock_alerter.py
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import time
import logging
from datetime import datetime

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# --- Updated list of tickers to monitor ---
TICKERS = [
    "NVDA", "TSLA", "SPY", "QQQ", "AAPL", "AMZN", "AMD", "MSFT", "META", "RDDT",
    "CRWV", "GOOGL", "AVGO", "BRK.B", "TSM", "LLY", "WMT", "JPM", "V", "ORCL",
    "NFLX", "MA", "XOM", "COST", "JNJ", "PG", "KO", "PLTR", "UNH", "BABA"
]

# Indicator parameters (matching Pine Script defaults where applicable)
EMA_35_LEN = 35
EMA_50_LEN = 50
EMA_200_LEN = 200
RSI_LEN = 14
RSI_BUYING_THRESHOLD = 30 # RSI < 30 is buying range
KC_LEN = 20
KC_MULTIPLIER = 2.0
KC_ATR_LEN = 10
MACD_FAST_LEN = 12
MACD_SLOW_LEN = 26
MACD_SIGNAL_LEN = 9

# Delay between fetching data for each ticker (in seconds)
# This helps prevent rate limiting issues with Yahoo Finance.
FETCH_DELAY_SECONDS = 0.5 # Half a second delay

# --- Helper Functions for Indicator Calculations ---

def calculate_ema(data, window):
    """Calculates Exponential Moving Average."""
    return data.ewm(span=window, adjust=False).mean()

def calculate_rsi(data, window):
    """Calculates Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, window):
    """Calculates Average True Range (ATR)."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.ewm(span=window, adjust=False).mean()
    return atr

def calculate_macd(data, fast_len, slow_len, signal_len):
    """Calculates Moving Average Convergence Divergence (MACD)."""
    ema_fast = calculate_ema(data, fast_len)
    ema_slow = calculate_ema(data, slow_len)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_len)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# --- Main Logic Functions ---

def get_stock_data(ticker, period="1y", interval="1d"):
    """Fetches historical stock data from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        # Access info to ensure ticker is valid and prime the yfinance object
        stock_info = stock.info
        data = stock.history(period=period, interval=interval)
        if data.empty:
            logging.warning(f"No data found for {ticker}")
            return None
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None

def analyze_stock(df, ticker_symbol):
    """Calculates all indicators and checks conditions for a single stock."""
    if df is None or df.empty:
        logging.warning(f"Skipping analysis for {ticker_symbol} due to empty or invalid DataFrame.")
        return None

    # Ensure enough data points for all calculations (e.g., 200 for EMA200)
    required_data_points = max(EMA_200_LEN, RSI_LEN, KC_LEN, KC_ATR_LEN, MACD_SLOW_LEN + MACD_SIGNAL_LEN)
    if len(df) < required_data_points:
        logging.warning(f"Not enough data for {ticker_symbol} to calculate all indicators. Needs at least {required_data_points} bars, has {len(df)}.")
        return None

    close_prices = df['Close']
    high_prices = df['High']
    low_prices = df['Low']

    # EMAs
    ema35 = calculate_ema(close_prices, EMA_35_LEN)
    ema50 = calculate_ema(close_prices, EMA_50_LEN)
    ema200 = calculate_ema(close_prices, EMA_200_LEN)

    # RSI
    rsi = calculate_rsi(close_prices, RSI_LEN)

    # Keltner Channels
    kc_basis = calculate_ema(close_prices, KC_LEN)
    kc_atr = calculate_atr(high_prices, low_prices, close_prices, KC_ATR_LEN)
    kc_upper = kc_basis + kc_atr * KC_MULTIPLIER
    kc_lower = kc_basis - kc_atr * KC_MULTIPLIER

    # MACD
    macd_line, signal_line, macd_hist = calculate_macd(close_prices, MACD_FAST_LEN, MACD_SLOW_LEN, MACD_SIGNAL_LEN)

    # Get latest values (ensure they are not NaN after calculations)
    # Check if the last elements are NaN before accessing
    if any(pd.isna([ema35.iloc[-1], ema50.iloc[-1], ema200.iloc[-1], rsi.iloc[-1], kc_lower.iloc[-1], macd_line.iloc[-1], signal_line.iloc[-1], macd_hist.iloc[-1]])):
        logging.warning(f"NaN values encountered for {ticker_symbol} in latest indicator calculations. Skipping.")
        return None


    latest_close = close_prices.iloc[-1]
    latest_ema35 = ema35.iloc[-1]
    latest_ema50 = ema50.iloc[-1]
    latest_ema200 = ema200.iloc[-1]
    latest_rsi = rsi.iloc[-1]
    latest_kc_lower = kc_lower.iloc[-1]
    latest_macd_line = macd_line.iloc[-1]
    latest_signal_line = signal_line.iloc[-1]
    latest_macd_hist = macd_hist.iloc[-1]

    # --- Conditions ---
    price_over_ema35 = latest_close > latest_ema35
    price_over_ema50 = latest_close > latest_ema50
    price_over_ema200 = latest_close > latest_ema200
    rsi_in_buy_range = latest_rsi < RSI_BUYING_THRESHOLD
    kc_in_buy_range = latest_close < latest_kc_lower # Price below lower Keltner band

    # Check for actual crossover (current MACD > Signal AND previous MACD <= previous Signal)
    macd_bullish_crossover = False
    if len(macd_line) > 1 and len(signal_line) > 1: # Ensure there are at least two data points for crossover check
        macd_bullish_crossover = (latest_macd_line > latest_signal_line and
                                  macd_line.iloc[-2] <= signal_line.iloc[-2])

    # Overall BUY condition: ALL must be true (EMAs, RSI, KC, MACD)
    overall_buy = (price_over_ema35 and price_over_ema50 and price_over_ema200 and
                   rsi_in_buy_range and kc_in_buy_range and macd_bullish_crossover)

    results = {
        "ticker": ticker_symbol,
        "price": latest_close,
        "ema35": latest_ema35,
        "ema50": latest_ema50,
        "ema200": latest_ema200,
        "rsi": latest_rsi,
        "kc_lower": latest_kc_lower,
        "macd_line": latest_macd_line,
        "signal_line": latest_signal_line,
        "macd_hist": latest_macd_hist,
        "price_over_ema35": price_over_ema35,
        "price_over_ema50": price_over_ema50,
        "price_over_ema200": price_over_ema200,
        "rsi_in_buy_range": rsi_in_buy_range,
        "kc_in_buy_range": kc_in_buy_range,
        "macd_bullish_crossover": macd_bullish_crossover,
        "overall_buy": overall_buy
    }
    return results

def send_discord_consolidated_alert(webhook_url, buy_signals):
    """Sends a consolidated alert message to Discord for all BUY signals."""
    if not webhook_url:
        logging.error("Discord Webhook URL environment variable is not set. Skipping alert.")
        return

    if not buy_signals:
        logging.info("No BUY signals found to send a Discord alert.")
        return

    MAX_FIELDS_PER_EMBED = 20 # Discord limit is 25 fields per embed
    embeds = []
    current_description = []
    
    main_content = f"**Daily Stock Analysis - BUY Signals ({datetime.now().strftime('%Y-%m-%d')})**"

    for signal in buy_signals:
        ticker = signal['ticker']
        details = (
            f"**{ticker}:** "
            f"Price ${signal['price']:.2f} | "
            f"EMA35 ${signal['ema35']:.2f} | "
            f"EMA50 ${signal['ema50']:.2f} | "
            f"EMA200 ${signal['ema200']:.2f} | "
            f"RSI {signal['rsi']:.2f} | "
            f"KC Lower ${signal['kc_lower']:.2f} | "
            f"MACD Hist {signal['macd_hist']:.2f}"
        )
        current_description.append(details)

        if len(current_description) >= MAX_FIELDS_PER_EMBED:
            embeds.append({
                "title": "📈 Consolidated BUY Signals 📈",
                "description": "\n".join(current_description),
                "color": 65280, # Green
                "footer": {
                    "text": f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                }
            })
            current_description = [] # Reset for next embed

    # Add any remaining signals to the last embed
    if current_description:
        embeds.append({
            "title": "📈 Consolidated BUY Signals 📈",
            "description": "\n".join(current_description),
            "color": 65280, # Green
            "footer": {
                "text": f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
        })

    payload = {
        "content": main_content,
        "embeds": embeds
    }

    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors
        logging.info(f"Consolidated Discord alert sent successfully with {len(buy_signals)} BUY signals.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send consolidated Discord alert: {e}")

# --- Main Script Execution ---
if __name__ == "__main__":
    if not DISCORD_WEBHOOK_URL:
        logging.error("DISCORD_WEBHOOK_URL environment variable is not set. Please set it as a GitHub Secret.")
        exit(1)

    buy_signals_found = []
    total_tickers = len(TICKERS)
    
    logging.info(f"Starting analysis for {total_tickers} tickers.")

    for i, ticker in enumerate(TICKERS):
        logging.info(f"({i+1}/{total_tickers}) Analyzing {ticker}...")
        data = get_stock_data(ticker)
        if data is not None and not data.empty:
            analysis = analyze_stock(data, ticker)
            if analysis and analysis["overall_buy"]:
                logging.info(f"*** {ticker}: Overall BUY Signal detected! ***")
                buy_signals_found.append(analysis)
            elif analysis: # Only log if analysis was successful but not a buy
                logging.info(f"--- {ticker}: No Overall BUY Signal. Holding. ---")
            else:
                logging.warning(f"Could not complete analysis for {ticker}.")
        else:
            logging.warning(f"Skipping analysis for {ticker} due to no data.")

        # Add a delay between requests to avoid hitting rate limits
        if i < total_tickers - 1: # Don't sleep after the last ticker
            time.sleep(FETCH_DELAY_SECONDS)

    if buy_signals_found:
        logging.info(f"Found {len(buy_signals_found)} stocks with BUY signals. Sending consolidated alert...")
        send_discord_consolidated_alert(DISCORD_WEBHOOK_URL, buy_signals_found)
    else:
        logging.info("No stocks met all BUY conditions today.")
