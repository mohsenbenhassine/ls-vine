"""
LS-Vine: Learning Vine-Friendly Latent Representations for Improved
Multivariate Dependence Modeling

A framework for learning latent representations that are compatible with
vine copula decomposition, improving multivariate dependence modeling.

Author: Mohsen Ben Hassine
"""

__version__ = "2.0.0"
__all__ = [
    "Config",
    "LSVineNet",
    "AENet",
    "VAENet",
    "train_lsvine",
    "train_ae",
    "train_vae",
    "train_wae",
    "train_infovae",
    "train_ae_selected",
    "run_benchmark",
    "run_ablation",
    "run_sensitivity",
    "make_student_dvine",
    "make_mixed_rvine",
    "make_sp500_calibrated",
    "make_era5_calibrated",
    "make_block_factor_student_t",
    "split_data",
]

from .config import Config
from .models import LSVineNet, AENet, VAENet
from .train import (
    train_lsvine,
    train_ae,
    train_vae,
    train_wae,
    train_infovae,
    train_ae_selected,
)
from .experiments import run_benchmark, run_ablation, run_sensitivity
from .data import (
    make_student_dvine,
    make_mixed_rvine,
    make_sp500_calibrated,
    make_era5_calibrated,
    make_block_factor_student_t,
    split_data,
)
