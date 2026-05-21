"""
autoencoder.py
TabularAE: a symmetric autoencoder for tabular particle detector data.

Architecture:
    Encoder: N → 64 → 32 → bottleneck
    Decoder: bottleneck → 32 → 64 → N
    Activation: ReLU (hidden), linear (output)
    Loss: MSE reconstruction loss

The bottleneck forces the model to compress normal data into a
low-dimensional representation. Corrupted rows fall off the learned
manifold — their reconstruction error is the anomaly score.

Nothing in this file is corruption-aware. It learns normal only.
"""

import torch
import torch.nn as nn
from pathlib import Path


class TabularAE(nn.Module):
    """
    Symmetric tabular autoencoder.

    Args:
        n_features:   Number of input features (must match scaler output).
        bottleneck:   Bottleneck dimension. Start with n_features // 2.
                      Too small: underfits clean data.
                      Too large: memorises rather than compresses.
        dropout:      Dropout rate applied in encoder hidden layers.
                      Helps prevent overfitting on clean training data.
    """

    def __init__(
        self,
        n_features:  int,
        bottleneck:  int = 4,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.bottleneck = bottleneck

        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, bottleneck),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
            # Linear output — matches StandardScaler output range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-row MSE reconstruction error.
        Returns shape (n_rows,) — the anomaly score for each row.
        """
        with torch.no_grad():
            x_hat = self.forward(x)
            return ((x - x_hat) ** 2).mean(dim=1)

    def per_feature_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-row, per-feature squared reconstruction error.
        Returns shape (n_rows, n_features).
        Used to identify WHICH field is anomalous — the bridge to interpretability.
        """
        with torch.no_grad():
            x_hat = self.forward(x)
            return (x - x_hat) ** 2

    def save(self, path: Path) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "n_features":  self.n_features,
            "bottleneck":  self.bottleneck,
        }, path)
        print(f"  Model saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "TabularAE":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            n_features = ckpt["n_features"],
            bottleneck = ckpt["bottleneck"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model