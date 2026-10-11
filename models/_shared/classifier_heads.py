import torch.nn as nn


class LinearClassifierHead(nn.Module):
    """Minimal linear head for supervised baselines."""

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPClassifierHead(nn.Module):
    """One-hidden-layer classifier head."""

    def __init__(self, in_features, hidden_features, num_classes, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x):
        return self.net(x)
