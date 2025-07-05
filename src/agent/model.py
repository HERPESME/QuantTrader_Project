# Python file
import torch
import torch.nn as nn


class CustomTradingModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomTradingModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    dummy_input = torch.randn(1, 50)  # assuming input size 50
    model = CustomTradingModel(input_dim=50, output_dim=3)  # 3 actions: buy, sell, hold
    print(model(dummy_input))
