import yfinance as yf
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import deque


def fetch_latest_price(symbol, interval="1m"):
    data = yf.download(tickers=symbol, period="1d", interval=interval, progress=False)
    if data.empty:
        return None, None
    latest = data.iloc[-1]
    timestamp = data.index[-1]
    close_price = float(latest["Close"])
    return timestamp, close_price


def stream_and_plot(symbol="AAPL", duration_minutes=5, interval_seconds=60):
    print(f"Starting real-time stream and plot for {symbol} using yfinance...")

    timestamps = deque(maxlen=100)  # keep last 100 data points
    prices = deque(maxlen=100)

    plt.ion()  # interactive mode on
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], marker="o")
    ax.set_title(f"Live Stock Price: {symbol}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    end_time = time.time() + duration_minutes * 60

    while time.time() < end_time:
        timestamp, close_price = fetch_latest_price(symbol)
        if timestamp is not None and close_price is not None:
            timestamps.append(timestamp)
            prices.append(close_price)

            line.set_xdata(timestamps)
            line.set_ydata(prices)
            ax.relim()
            ax.autoscale_view()

            fig.autofmt_xdate()
            plt.draw()
            plt.pause(0.01)

            print(f"[{timestamp}] Price: {close_price:.2f}")
        else:
            print("No data this interval.")

        time.sleep(interval_seconds)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    stream_and_plot(symbol="AAPL", duration_minutes=5, interval_seconds=60)
