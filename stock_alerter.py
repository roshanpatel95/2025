# stock_alerter.py
import yfinance as yf
import requests
import os
import logging
from datetime import datetime

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# --- Only monitor SPY for this test ---
TICKERS = ["SPY"]

# --- Main Logic Functions ---

def get_stock_data(ticker, period="1d", interval="1d"):
    """Fetches historical stock data (just current day's close for simplicity)."""
    try:
        stock = yf.Ticker(ticker)
        # Fetching the last day's data to get the most recent closing price
        data = stock.history(period=period, interval=interval)
        if data.empty:
            logging.warning(f"No data found for {ticker}")
            return None
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None

def send_simple_discord_alert(webhook_url, ticker, price):
    """Sends a simple alert message to Discord with the current price."""
    if not webhook_url:
        logging.error("Discord Webhook URL environment variable is not set. Skipping alert.")
        return

    main_content = f"**Daily Price Update for {ticker}**"
    
    embed = {
        "title": f"Current Price for {ticker}",
        "description": f"The current closing price for **{ticker}** is **${price:.2f}**.",
        "color": 3447003, # A standard blue color for Discord embeds
        "footer": {
            "text": f"Update Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        }
    }

    payload = {
        "content": main_content,
        "embeds": [embed]
    }

    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors
        logging.info(f"Discord alert sent successfully for {ticker} with price ${price:.2f}.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send Discord alert for {ticker}: {e}")

# --- Main Script Execution ---
if __name__ == "__main__":
    if not DISCORD_WEBHOOK_URL:
        logging.error("DISCORD_WEBHOOK_URL environment variable is not set. Please set it as a GitHub Secret.")
        exit(1)

    for ticker in TICKERS:
        logging.info(f"Fetching price for {ticker}...")
        data = get_stock_data(ticker)
        if data is not None and not data.empty:
            current_price = data['Close'].iloc[-1]
            logging.info(f"Latest price for {ticker}: ${current_price:.2f}")
            send_simple_discord_alert(DISCORD_WEBHOOK_URL, ticker, current_price)
        else:
            logging.warning(f"Could not get price data for {ticker}. Skipping alert.")

