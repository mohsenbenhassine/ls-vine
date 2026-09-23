"""
src/config.py

Global hyperparameter configuration for LS-Vine experiments.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Global hyperparameters for LS-Vine experiments."""

    # -------------------------------------------------------------------------
    # Training parameters
    # -------------------------------------------------------------------------
    epochs:          int   = 500       # max epochs
    batch_size:      int   = 512
    lr:              float = 1e-3
    patience:        int   = 60        # early stopping patience
    pretrain_epochs: int   = 10        # warmup-free pretraining phase

    # -------------------------------------------------------------------------
    # LS-Vine specific
    # -------------------------------------------------------------------------
    gamma:           float = 0.01      # weight of mean term in reg_loss
    lam_var:         float = 0.002     # weight of log-variance term in reg_loss
    alpha:           float = 0.5       # weight of soft-tau loss in L_rec
    beta_tau:        float = 1.0       # temperature for soft Kendall's tau
    K:               int   = 5         # vine re-estimation interval (epochs)
    tau_w:           float = 15.0      # warmup timescale
    lam_max:         float = 1.0       # max weight for L_v
    frac_pairs:      float = 0.5       # fraction of dim pairs sampled per batch
    n_quantiles:     int   = 20        # quantile grid size for L_v
    tail_weight:     float = 2.0       # >1 concentrates grid on strong dep.
    trunc:           int   = 3         # vine truncation depth

    # -------------------------------------------------------------------------
    # VAE / WAE / InfoVAE
    # -------------------------------------------------------------------------
    vae_beta:        float = 0.1       # KL weight for VAE/InfoVAE
    wae_lambda:      float = 10.0      # MMD weight for WAE/InfoVAE

    # -------------------------------------------------------------------------
    # Baseline: truncated vine
    # -------------------------------------------------------------------------
    trunc_short:     int   = 1

    # -------------------------------------------------------------------------
    # Experiment settings
    # -------------------------------------------------------------------------
    n_seeds:         int   = 10
    seeds:           List  = field(default_factory=lambda: list(range(42, 52)))
    hidden:          int   = 256
    d_lat_var_target: float = 0.90    # variance threshold for auto d_lat

    # -------------------------------------------------------------------------
    # Optional flags
    # -------------------------------------------------------------------------
    compute_stability: bool = False    # extra runs for structural stability
    n_boot_real:       int  = 20       # bootstrap iterations on real data


CFG = Config()
