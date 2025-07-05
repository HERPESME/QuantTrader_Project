import yfinance as yf
import time
import pandas as pd


def fetch_real_time(symbol, interval="1m", max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers=symbol, period="1d", interval=interval, progress=False
            )
            if data.empty:
                raise ValueError("Empty data received from yfinance.")
            return data
        except Exception as e:
            print(f"Error fetching data (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(5)
    raise RuntimeError("Failed to fetch real-time data after multiple attempts")


def stream_realtime_data(symbol, duration_minutes=5, interval_seconds=60):
    end_time = time.time() + duration_minutes * 60
    print(f"Starting real-time stream for {symbol} using yfinance...")

    while time.time() < end_time:
        df = fetch_real_time(symbol)
        if df.empty:
            print("No data received. Skipping this iteration.")
            time.sleep(interval_seconds)
            continue

        latest = df.iloc[-1]
        timestamp = df.index[-1]  # Use index directly

        close_price = (
            float(latest["Close"].iloc[0])
            if isinstance(latest["Close"], pd.Series)
            else float(latest["Close"])
        )
        print(f"[{timestamp}] Price: {close_price:.2f}")

        time.sleep(interval_seconds)


if __name__ == "__main__":
    SYMBOL = "AAPL"
    stream_realtime_data(SYMBOL, duration_minutes=5, interval_seconds=60)
