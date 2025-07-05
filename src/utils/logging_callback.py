# src/utils/logging_callback.py

import os
import csv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ray.rllib.evaluation.episode import Episode


class RewardLoggingCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.file_path = "results/training_rewards.csv"
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        os.makedirs("results", exist_ok=True)
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "reward", "profit"])

    def on_episode_end(self, *, episode, **kwargs):
        # ✅ WORKING on Ray 2.x+
        reward = episode.total_reward  # <- this is correct now
        profit = episode.last_info_for().get("profit", 0)

        with open(self.file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode.episode_id, reward, profit])

        print(
            f"[RewardLoggingCallback] Episode {episode.episode_id} - Reward: {reward:.2f}, Profit: {profit:.2f}"
        )
