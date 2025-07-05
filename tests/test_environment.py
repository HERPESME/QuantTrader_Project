import unittest
import numpy as np
import pandas as pd
from src.env.stock_trading_env import StockTradingEnv


def get_dummy_df():
    data = {
        "Open": np.random.rand(100) * 100,
        "High": np.random.rand(100) * 100,
        "Low": np.random.rand(100) * 100,
        "Close": np.random.rand(100) * 100,
        "Volume": np.random.rand(100) * 1000,
    }
    return pd.DataFrame(data)


class TestStockTradingEnv(unittest.TestCase):
    def setUp(self):
        self.df = get_dummy_df()
        self.env = StockTradingEnv(df=self.df)

    def test_environment_reset(self):
        obs, info = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertGreater(obs.shape[0], 0)
        self.assertIsInstance(info, dict)
        # It's okay if info is empty after reset

    def test_environment_step(self):
        obs, info = self.env.reset()
        action = self.env.action_space.sample()
        new_obs, reward, done, truncated, info = self.env.step(action)

        self.assertIsInstance(new_obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

        # Check keys that are returned in the info dict after step
        self.assertIn("net_worth", info)
        self.assertIn("balance", info)
        self.assertIn("shares_held", info)
        self.assertIn("profit", info)
        self.assertIn("step", info)

    def test_action_space(self):
        self.assertEqual(self.env.action_space.n, 3)

    def test_observation_space(self):
        self.assertIsInstance(self.env.observation_space.shape, tuple)
        self.assertGreater(self.env.observation_space.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
