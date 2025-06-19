# stock_alerter.py
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# --- Configuration ---
# The Discord Webhook URL will be read from environment variables when run via GitHub Actions.
# For local testing, you can uncomment and set it directly, but for GitHub, it MUST be a secret.
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# List of tickers to monitor
TICKERS = ["AAPL", "MSFT", "GOOGL"] # Example tickers, modify as needed

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
        data = stock.history(period=period, interval=interval)
        if data.empty:
            print(f"No data found for {ticker}")
            return None
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def analyze_stock(df):
    """Calculates all indicators and checks conditions."""
    if df is None or df.empty:
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

    # Get latest values
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
    macd_bullish_crossover = latest_macd_line > latest_signal_line and macd_line.iloc[-2] <= signal_line.iloc[-2] # Check for actual crossover

    # Overall BUY condition: ALL must be true
    overall_buy = (price_over_ema35 and price_over_ema50 and price_over_ema200 and
                   rsi_in_buy_range and kc_in_buy_range and macd_bullish_crossover)

    results = {
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

def send_discord_alert(webhook_url, ticker, analysis_results):
    """Sends a formatted alert message to Discord."""
    if not webhook_url: # Check if webhook_url is None or empty
        print("Discord Webhook URL not set in environment variables. Skipping alert.")
        return

    buy_status = "✅ BUY SIGNAL! ✅" if analysis_results["overall_buy"] else "❌ HOLD ❌"
    color = 65280 if analysis_results["overall_buy"] else 16711680 # Green or Red

    description = (
        f"**Current Price:** ${analysis_results['price']:.2f}\n"
        f"--- Indicator Details ---\n"
        f"- **Price > EMA 35**: {'True' if analysis_results['price_over_ema35'] else 'False'} (EMA 35: {analysis_results['ema35']:.2f})\n"
        f"- **Price > EMA 50**: {'True' if analysis_results['price_over_ema50'] else 'False'} (EMA 50: {analysis_results['ema50']:.2f})\n"
        f"- **Price > EMA 200**: {'True' if analysis_results['price_over_ema200'] else 'False'} (EMA 200: {analysis_results['ema200']:.2f})\n"
        f"- **RSI < {RSI_BUYING_THRESHOLD}**: {'True' if analysis_results['rsi_in_buy_range'] else 'False'} (RSI: {analysis_results['rsi']:.2f})\n"
        f"- **Price < KC Lower**: {'True' if analysis_results['kc_in_buy_range'] else 'False'} (KC Lower: {analysis_results['kc_lower']:.2f})\n"
        f"- **MACD Bullish Crossover**: {'True' if analysis_results['macd_bullish_crossover'] else 'False'} (MACD Hist: {analysis_results['macd_hist']:.2f})"
    )

    embed = {
        "title": f"📈 Stock Analysis for {ticker} 📈",
        "description": description,
        "color": color,
        "footer": {
            "text": f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        }
    }

    payload = {
        "content": f"**{ticker} Daily Chart Analysis:** {buy_status}",
        "embeds": [embed]
    }

    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors
        print(f"Discord alert sent successfully for {ticker}.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Discord alert for {ticker}: {e}")

# --- Main Script Execution ---
if __name__ == "__main__":
    # In GitHub Actions, DISCORD_WEBHOOK_URL will be set by the secret.
    # For local testing, you might need to set it manually or via your shell.
    # Example for local testing: export DISCORD_WEBHOOK_URL="your_webhook_url_here"
    
    if not DISCORD_WEBHOOK_URL:
        print("DISCORD_WEBHOOK_URL environment variable is not set.")
        print("Please set it before running the script, especially for local testing.")
        print("When using GitHub Actions, ensure you've configured it as a repository secret.")
        exit(1) # Exit if webhook URL is not set

    for ticker in TICKERS:
        print(f"\nAnalyzing {ticker}...")
        data = get_stock_data(ticker)
        if data is not None and not data.empty:
            analysis = analyze_stock(data)
            if analysis:
                print(f"Analysis Results for {ticker}:")
                for key, value in analysis.items():
                    if isinstance(value, (float, np.float64)):
                        print(f"- {key}: {value:.2f}")
                    else:
                        print(f"- {key}: {value}")

                if analysis["overall_buy"]:
                    print(f"*** {ticker}: Overall BUY Signal detected! ***")
                    send_discord_alert(DISCORD_WEBHOOK_URL, ticker, analysis)
                else:
                    print(f"--- {ticker}: No Overall BUY Signal. Holding. ---")
            else:
                print(f"Could not perform analysis for {ticker}.")
        else:
            print(f"Skipping analysis for {ticker} due to no data.")

