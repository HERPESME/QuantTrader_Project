# src/run.py
import os
import ray
import pandas as pd
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.registry import register_env
from src.env.stock_trading_env import StockTradingEnv
from dotenv import load_dotenv

load_dotenv()


def load_data():
    data_path = os.path.join("data", "historical", "AAPL.csv")
    df = pd.read_csv(data_path, skiprows=2)
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna()
    return df


def env_creator(env_config):
    return StockTradingEnv(
        df=env_config["df"],
        initial_balance=env_config.get("initial_balance", 10000),
        transaction_cost=env_config.get("transaction_cost", 0.001),
        window_size=env_config.get("window_size", 10),
    )


def run_trained_agent(checkpoint_path):
    df = load_data()
    register_env("StockTradingEnv-v0", lambda config: env_creator({**config, "df": df}))

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
        .api_stack(  # Must match training
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            }
        )
    )

    # Restore trained agent
    agent = PPO(config=config)
    agent.restore(checkpoint_path)

    env = StockTradingEnv(df=df)
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.compute_single_action(obs)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        env.render()

    print(f"\n✅ Evaluation Complete — Total Reward: {total_reward:.2f}")


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    checkpoint_path = "/Users/eeshansingh/ray_results/PPO_2025-07-05_18-20-04/PPO_StockTradingEnv-v0_9295a_00000_0_2025-07-05_18-20-04/checkpoint_000000"  # TODO: Set actual path
    run_trained_agent(checkpoint_path)
