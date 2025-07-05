import time
import numpy as np
import pandas as pd
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from src.env.stock_trading_env import StockTradingEnv
from ray.tune.registry import register_env

# Load recent stock data
df = pd.read_csv("data/historical/AAPL.csv", skiprows=2)
df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df = df.dropna()


# Define custom environment
def env_creator(env_config):
    return StockTradingEnv(
        df=env_config["df"],
        initial_balance=env_config.get("initial_balance", 10000),
        transaction_cost=env_config.get("transaction_cost", 0.001),
        window_size=env_config.get("window_size", 10),
    )


# Register environment
register_env("StockTradingEnv-v0", lambda config: env_creator({**config, "df": df}))

# Rebuild PPO config used during training
config = (
    PPOConfig()
    .environment(
        env="StockTradingEnv-v0",
        env_config={
            "df": df,
            "initial_balance": 10000,
            "transaction_cost": 0.001,
            "window_size": 10,
        },
    )
    .framework("torch")
    .api_stack(
        enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
    )
    .rl_module(
        model_config={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        }
    )
)

# 🔒 Use fixed checkpoint path (same as in run.py)
model_path = "/home/user/ray_results/PPO_2025-05-24_13-33-23/PPO_StockTradingEnv-v0_907bf_00000_0_2025-05-24_13-33-23/checkpoint_000000"
model = PPO(config=config)
model.restore(model_path)

# 🎬 Begin simulation
env = StockTradingEnv(df=df)
obs, info = env.reset()
done = False
total_reward = 0

print("\n🎬 Starting live simulation...\n")

while not done:
    action = model.compute_single_action(obs)
    obs, reward, done, _, info = env.step(action)
    total_reward += reward

    action_name = ["HOLD", "BUY", "SELL"][action]
    print(
        f"Step: {info['step']}, Action: {action_name}, Net Worth: ${info['net_worth']:.2f}"
    )
    time.sleep(0.5)

print("\n✅ Simulation ended.")
print(f"🏁 Final Net Worth: ${info['net_worth']:.2f}")
print(f"🏆 Total Reward: {total_reward:.2f}")
