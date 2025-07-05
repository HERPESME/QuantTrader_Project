# src/data/data_pipeline.py

import yfinance as yf
import os


def fetch_data_yf(symbol="AAPL", start="2018-01-01", end=None):
    df = yf.download(symbol, start=start, end=end)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)

    os.makedirs("data/historical", exist_ok=True)
    save_path = os.path.join("data", "historical", f"{symbol}.csv")
    df.to_csv(save_path)
    print(f"Yahoo Finance data for {symbol} saved to {save_path}")
    return df


if __name__ == "__main__":
    fetch_data_yf("AAPL")
