import torch.nn as nn


class IncomeClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual block 1
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Residual block 2
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.block1(x)  # residual
        x = x + self.block2(x)  # residual
        return self.head(x)
