import os
import ray
import pandas as pd
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from dotenv import load_dotenv
from src.env.stock_trading_env import StockTradingEnv
from src.utils.logging_callback import RewardLoggingCallback

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


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

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
        .env_runners(num_env_runners=1)
        .callbacks(RewardLoggingCallback)  # ✅ Logging to CSV
        .api_stack(  # ✅ Disabling unstable RLModule + EnvRunner v2
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            }
        )
        .training(
            gamma=0.99,
            lr=1e-4,
            train_batch_size=200,
        )
        .resources(num_gpus=int(os.environ.get("NUM_GPUS", 0)))
        .evaluation(
            evaluation_interval=1,
            evaluation_config={"explore": False},
            evaluation_num_env_runners=1,
        )
    )

    config.sgd_minibatch_size = 64
    config.num_sgd_iter = 10

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            stop={"training_iteration": 20},
            checkpoint_config=tune.CheckpointConfig(checkpoint_at_end=True),
        ),
    )

    tuner.fit()
