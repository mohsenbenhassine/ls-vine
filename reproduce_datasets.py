"""
reproduce_datasets.py

Reproduce all simulated and empirically calibrated synthetic datasets used in 
the LS-Vine benchmark. Saves each dataset as a CSV file in the data/ directory.

Usage
-----
python reproduce_datasets.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from scipy.stats import t as t_dist
import pyvinecopulib as pv

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def save(X, name):
    """Save dataset X as CSV in data/ directory."""
    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(X.shape[1])])
    path = DATA_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {name}: shape={X.shape} -> {path}")


def make_student_dvine(d, rho, nu, n, seed=42):
    """S1, S2 — Student-t D-vine."""
    pc = []
    for tree in range(d - 1):
        pc.append([pv.Bicop(family=pv.BicopFamily.student,
                            parameters=np.array([[rho], [float(nu)]]))
                   for _ in range(d - 1 - tree)])
    vine = pv.Vinecop.from_structure(
        structure=pv.DVineStructure(list(range(1, d + 1))),
        pair_copulas=pc)
    U = vine.simulate(n, seeds=[seed])
    return norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))


def make_mixed_rvine(d, n, seed=42):
    """S3 — Mixed R-vine (heterogeneous families)."""
    fams = [pv.BicopFamily.student, pv.BicopFamily.clayton,
            pv.BicopFamily.gumbel, pv.BicopFamily.frank,
            pv.BicopFamily.student, pv.BicopFamily.clayton]
    params = [np.array([[0.5], [4.]]), np.array([[3.]]), np.array([[3.]]),
              np.array([[4.]]), np.array([[0.4], [6.]]), np.array([[2.]])]
    pc = []
    for tree in range(d - 1):
        row = [pv.Bicop(family=fams[(tree + e) % len(fams)],
                        parameters=params[(tree + e) % len(params)])
               for e in range(d - 1 - tree)]
        pc.append(row)
    vine = pv.Vinecop.from_structure(
        structure=pv.DVineStructure(list(range(1, d + 1))),
        pair_copulas=pc)
    U = vine.simulate(n, seeds=[seed])
    return norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))


def make_sp500_calibrated(n=1500, d=20, seed=42):
    """R1 — S&P500-calibrated financial returns."""
    np.random.seed(seed)
    s = 4; sectors = d // s
    Sigma = np.full((d, d), 0.25)
    for i in range(sectors):
        Sigma[i*s:(i+1)*s, i*s:(i+1)*s] = 0.65
    np.fill_diagonal(Sigma, 1.0)
    L = np.linalg.cholesky(Sigma)
    X = t_dist.rvs(df=4.5, size=(n, d)) @ L.T
    vols = np.random.uniform(0.20, 0.35, d) / np.sqrt(252)
    X *= vols
    X[500:520] *= 3.5
    X[500:510] -= 0.025
    return X


def make_era5_calibrated(n=3000, d=15, seed=43):
    """R2 — ERA5-calibrated meteorological data."""
    np.random.seed(seed)
    c = 5; n_cl = d // c
    Sigma = np.full((d, d), 0.18)
    for i in range(n_cl):
        Sigma[i*c:(i+1)*c, i*c:(i+1)*c] = 0.72
    np.fill_diagonal(Sigma, 1.0)
    L = np.linalg.cholesky(Sigma)
    X = t_dist.rvs(df=5.0, size=(n, d)) @ L.T
    t_idx = np.arange(n)
    for c_i in range(n_cl):
        X[:, c_i*c:(c_i+1)*c] += (0.3 + 0.1*c_i) * np.sin(
            2*np.pi*t_idx/365)[:, None]
    X[1000:1365] *= 0.4
    return X


def make_gaussian_factor(d=15, n=1500, n_factors=3, seed=42):
    """S4 — Gaussian Factor Model."""
    np.random.seed(seed)
    factors = np.random.randn(n, n_factors)
    loadings = np.random.uniform(0.5, 0.9, size=(d, n_factors))
    noise = np.random.randn(n, d) * 0.5
    return factors @ loadings.T + noise


def make_noisy_lowdim(d=10, n=1500, noise_ratio=0.30, seed=42):
    """S5 — Noisy Low-Dim (30% noise)."""
    np.random.seed(seed)
    n_signal = int(d * (1 - noise_ratio))
    n_noise = d - n_signal
    signal = np.random.randn(n, n_signal)
    noise = np.random.randn(n, n_noise)
    X = np.hstack([signal, noise])
    idx = np.random.permutation(d)
    return X[:, idx]


def make_pure_gaussian_copula(d=15, n=1500, rho=0.5, seed=42):
    """S6 — Pure Gaussian Copula."""
    np.random.seed(seed)
    Sigma = np.full((d, d), rho)
    np.fill_diagonal(Sigma, 1.0)
    L = np.linalg.cholesky(Sigma)
    Z = np.random.randn(n, d) @ L.T
    U = norm.cdf(Z)
    return norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))


def make_block_factor_student_t(d=20, n_blocks=4, n_per_block=5,
                                 rho=0.7, nu=4, n_samples=2000, seed=42):
    """S7 — Block Factor Student-t (no localized shocks)."""
    np.random.seed(seed)
    factors = t_dist.rvs(df=nu, size=(n_samples, n_blocks))
    noise = t_dist.rvs(df=nu, size=(n_samples, d))
    X = np.zeros((n_samples, d))
    for b in range(n_blocks):
        start = b * n_per_block
        end = start + n_per_block
        X[:, start:end] = rho * factors[:, b:b+1] + np.sqrt(1-rho**2) * noise[:, start:end]
    return X


def main():
    print("=" * 60)
    print("  Reproducing all LS-Vine datasets")
    print("=" * 60)

    print("\n[S1] Student-t D-vine (d=10, nu=4) ...")
    save(make_student_dvine(10, 0.4, 4, 3500, seed=42),
         "S1_student_dvine_d10")

    print("\n[S2] Student-t D-vine (d=20, nu=4) ...")
    save(make_student_dvine(20, 0.4, 4, 3500, seed=42),
         "S2_student_dvine_d20")

    print("\n[S3] Mixed R-vine (d=12, heterogeneous) ...")
    save(make_mixed_rvine(12, 3500, seed=42),
         "S3_mixed_rvine_d12")

    print("\n[R1] S&P500-calibrated financial returns (d=20) ...")
    save(make_sp500_calibrated(1500, 20, seed=42),
         "R1_sp500_calibrated_d20")

    print("\n[R2] ERA5-calibrated meteorological data (d=15) ...")
    save(make_era5_calibrated(3000, 15, seed=43),
         "R2_era5_calibrated_d15")

    print("\n[S4] Gaussian Factor Model (d=15) ...")
    save(make_gaussian_factor(15, 1500, 3, seed=42),
         "S4_gaussian_factor_d15")

    print("\n[S5] Noisy Low-Dim (d=10, 30% noise) ...")
    save(make_noisy_lowdim(10, 1500, 0.30, seed=42),
         "S5_noisy_lowdim_d10")

    print("\n[S6] Pure Gaussian Copula (d=15) ...")
    save(make_pure_gaussian_copula(15, 1500, 0.5, seed=42),
         "S6_pure_gaussian_d15")

    print("\n[S7] Block Factor Student-t (d=20, no shocks) ...")
    save(make_block_factor_student_t(20, 4, 5, 0.7, 4, 2000, seed=42),
         "S7_block_factor_d20")

    print("\n" + "=" * 60)
    print("  All datasets reproduced in data/")
    print("=" * 60)


if __name__ == "__main__":
    main()
