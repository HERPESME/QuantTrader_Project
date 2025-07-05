# QuantTrader - Reinforcement Learning Trading Agent

QuantTrader is a deep reinforcement learning agent that autonomously learns to trade stocks for maximum profit using historical and real-time market data. It leverages custom Gym environments and RLlib (Ray) to train, simulate, and optionally paper trade in real-time.

## 🌐 Features
- PPO-based stock trading agent
- Yahoo Finance integration for historical and real-time data
- Real-time price streaming support for paper/live trading
- Custom OpenAI Gym-compatible trading environment
- Reward shaping with transaction costs and portfolio tracking
- Modular codebase for training, inference, and evaluation
- Jupyter notebooks for EDA and visualization
- Config-driven setup for reproducibility

## 🛠 Tech Stack
- Python, PyTorch
- Ray + RLlib
- OpenAI Gym
- Yahoo Finance (`yfinance`)
- Pandas, NumPy, Matplotlib
- YAML for config management

## 📁 Project Structure
```
QuantTrader/
├── config/
│   ├── env.yaml                  # Trading environment parameters
│   ├── train.yaml                # RL training hyperparameters
│   └── logging.yaml              # Logging settings
├── data/
│   ├── historical/               # Downloaded data for training
│   └── raw/                      # (Optional) raw tick-level data
├── notebooks/
│   ├── analysis.ipynb            # EDA and feature exploration
│   └── results_visualization.ipynb   # Training result visualization
├── src/
│   ├── env/
│   │   ├── stock_trading_env.py     # Gym-compatible trading environment
│   │   └── utils.py                 # Performance metrics and helpers
│   ├── data/
│   │   └── data_pipeline.py         # Data fetching (historical + real-time)
│   ├── agent/
│   │   └── model.py                 # (Optional) custom model architecture
│   ├── train.py                     # RLlib training script
│   └── run.py                       # Paper/live trading runner
├── tests/
│   └── test_environment.py         # Unit tests
├── requirements.txt                # Dependencies
└── README.md
```

## 🚀 Setup Instructions

### 1. Clone and Install
```bash
git clone https://github.com/HERPESME/PYTHON-AUTOMATION.git
cd QuantTrader_Project
pip install -r requirements.txt
```

### 2. Fetch Historical Stock Data
```bash
python src/data/data_pipeline.py
```
This will save historical OHLCV data to `data/historical/
!!!!!!! Remember to Add Date in first line of csv if not present`.

### 3. Run Real-Time Streaming (Optional) and Live Plotting (Optional)
```bash
python src/data/realtime_stream.py
python src/data/live_plot.py
```
Prints and Plots real-time price updates (default: every 60 seconds for 5 minutes).

### 4. Train the Agent
```bash
python -m src.train
```

### 5. Evaluate or Paper Trade
```bash
python -m src.run
```

## 🧠 Agent Algorithm

- Uses **Proximal Policy Optimization (PPO)** via Ray RLlib.
- Custom Gym environment with state shape `[window_size, features]`, including OHLCV + technical indicators.
- Discrete action space: **Buy, Hold, Sell**.
- Reward function: `portfolio_return - transaction_cost`.

You can tweak hyperparameters in `config/train.yaml` and environment settings in `config/env.yaml`.

## 📉 Visualization
To analyze performance, use:
```bash
notebooks/results_visualization.ipynb
```

Visualizations include:
- Reward per episode
- Portfolio growth over time
- Action distributions

## 🧪 Testing
Run unit tests with:
```bash
PYTHONPATH=. pytest tests/test_environment.py
```

## 📦 Deployment

### Docker Support (Optional)
To build and run in a containerized environment:

```bash
# Build Docker image
docker build -t quanttrader .

# Run container
docker run quanttrader

# (Optional) Mount local data folder
docker run -v $(pwd)/data:/app/data quanttrader
```

> No API keys required – all data is fetched using Yahoo Finance (public).

## ☁️ Cloud Hosting
QuantTrader is Docker-ready and deployable on:
- **Amazon EC2**
- **Render**
- **Heroku (via Docker)**
- Any VM or containerized cloud service

## 📄 License
MIT License

---

Made with using PyTorch + Ray RLlib
