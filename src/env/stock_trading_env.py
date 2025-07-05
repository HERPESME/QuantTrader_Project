import gymnasium as gym
from gymnasium import spaces
import numpy as np


class StockTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self, df, initial_balance=10000, transaction_cost=0.001, window_size=10
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size

        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size * 5,),
            dtype=np.float32,
        )

        self.seed()
        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self, *, seed=None, options=None):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = []
        self.current_step = self.window_size
        self.prev_net_worth = self.initial_balance
        return self._get_observation(), {}

    def _get_observation(self):
        frame = self.df.iloc[self.current_step - self.window_size : self.current_step]
        obs = frame[["Open", "High", "Low", "Close", "Volume"]].values
        return obs.flatten().astype(np.float32)

    def step(self, action):
        done = False
        reward = 0.0

        action = int(action)
        if action not in [0, 1, 2]:
            raise ValueError(f"Invalid action: {action}")

        current_price = self.df.iloc[self.current_step]["Close"]

        if np.isnan(current_price) or current_price <= 0:
            done = True
            return self._get_observation(), reward, done, False, {}

        # Buy
        if action == 1:
            max_shares = self.balance // current_price
            if max_shares > 0:
                self.shares_held += max_shares
                cost = max_shares * current_price * (1 + self.transaction_cost)
                self.balance -= cost
                self.trades.append(("buy", self.current_step, current_price))

        # Sell
        elif action == 2:
            if self.shares_held > 0:
                proceeds = (
                    self.shares_held * current_price * (1 - self.transaction_cost)
                )
                self.balance += proceeds
                self.shares_held = 0
                self.trades.append(("sell", self.current_step, current_price))

        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        reward = float(self.net_worth - self.prev_net_worth)
        self.prev_net_worth = self.net_worth

        if action == 0:
            reward -= 0.1

        self.current_step += 1
        if self.current_step >= len(self.df):
            done = True

        obs = self._get_observation()
        info = {
            "step": self.current_step,
            "net_worth": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "profit": self.net_worth - self.initial_balance,  # ✅ Needed by callback
        }

        return obs, reward, done, False, info

    def render(self, mode="human"):
        profit = self.net_worth - self.initial_balance
        print(
            f"Step: {self.current_step}, Balance: {self.balance:.2f}, Shares: {self.shares_held}, "
            f"Net worth: {self.net_worth:.2f}, Profit: {profit:.2f}"
        )
