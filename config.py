"""
Configuration and hyperparameters for LS-Vine.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Global configuration for LS-Vine experiments."""

    # Training parameters
    epochs: int = 500
    batch_size: int = 512
    lr: float = 1e-3
    patience: int = 15
    pretrain_epochs: int = 60

    # LS-Vine specific parameters
    gamma: float = 0.01
    lam_var: float = 0.002
    alpha: float = 0.5
    beta_tau: float = 1.0
    K: int = 5  # Vine re-estimation interval
    tau_w: float = 15.0  # Warmup timescale
    lam_max: float = 1.0  # Maximum vine loss weight
    frac_pairs: float = 0.5  # Fraction of pairs for soft tau
    n_quantiles: int = 20
    tail_weight: float = 2.0
    trunc: int = 3  # Vine truncation depth

    # VAE parameters
    vae_beta: float = 0.1

    # WAE / InfoVAE
    wae_lambda: float = 10.0

    # Experiment settings
    n_seeds: int = 10
    seeds: List[int] = field(default_factory=lambda: list(range(42, 52)))
    hidden: int = 256
    d_lat_var_target: float = 0.90
    n_boot_real: int = 20
    trunc_short: int = 1

    # Compute stability (for debugging)
    compute_stability: bool = False


# Default configuration instance
CFG = Config()
