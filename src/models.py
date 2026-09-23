"""
src/models.py

Neural network architectures for LS-Vine and baseline methods.

Contains:
    - MLP         : shared two-hidden-layer perceptron backbone
    - LSVineNet   : encoder-decoder for LS-Vine
    - AENet       : plain autoencoder (AE-Vine, WAE-Vine)
    - VAEEncoder  : VAE encoder (mu + logvar)
    - VAENet      : variational autoencoder (VAE-Vine, InfoVAE-Vine)
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Two-hidden-layer MLP with BatchNorm + ELU."""

    def __init__(self, d_in: int, d_out: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ELU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


class LSVineNet(nn.Module):
    """Encoder-Decoder architecture for LS-Vine."""

    def __init__(self, d_in: int, d_lat: int, hidden: int = 256):
        super().__init__()
        self.encoder = MLP(d_in, d_lat, hidden)
        self.decoder = MLP(d_lat, d_in, hidden)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


class AENet(nn.Module):
    """Plain Autoencoder (MSE reconstruction only)."""

    def __init__(self, d_in: int, d_lat: int, hidden: int = 256):
        super().__init__()
        self.encoder = MLP(d_in, d_lat, hidden)
        self.decoder = MLP(d_lat, d_in, hidden)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


class VAEEncoder(nn.Module):
    """VAE encoder with reparameterization (mu + logvar)."""

    def __init__(self, d_in: int, d_lat: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ELU(),
        )
        self.mu     = nn.Linear(hidden, d_lat)
        self.logvar = nn.Linear(hidden, d_lat)

    def forward(self, x):
        h = self.shared(x)
        return self.mu(h), self.logvar(h)


class VAENet(nn.Module):
    """Variational Autoencoder."""

    def __init__(self, d_in: int, d_lat: int, hidden: int = 256):
        super().__init__()
        self.encoder = VAEEncoder(d_in, d_lat, hidden)
        self.decoder = MLP(d_lat, d_in, hidden)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, self.decoder(z), mu, logvar
