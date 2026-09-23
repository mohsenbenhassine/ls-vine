"""
src/datasets.py

Dataset generators for all 9 experimental scenarios (S1-S7, R1, R2).

Contains:
    - make_student_dvine           : S1, S2 - Student-t D-vine
    - make_mixed_rvine             : S3     - Mixed R-vine
    - make_sp500_calibrated        : R1     - S&P500-calibrated returns
    - make_era5_calibrated         : R2     - ERA5-calibrated meteorology
    - make_gaussian_factor         : S4     - Gaussian Factor Model
    - make_noisy_lowdim            : S5     - Noisy Low-Dim
    - make_pure_gaussian_copula    : S6     - Pure Gaussian Copula
    - make_block_factor_student_t  : S7     - Block Factor Student-t
    - split                        : synthetic train/val/test split
    - split_real                   : calibrated 60/20/20 split
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import t as t_dist

try:
    import pyvinecopulib as pv
    VINE_OK = True
except ImportError:
    VINE_OK = False


# =============================================================================
# Primary scenarios
# =============================================================================
def make_student_dvine(d, rho, nu, n, seed=42):
    """S1, S2 - Student-t D-vine with equicorrelated pair copulas."""
    if not VINE_OK:
        raise RuntimeError("pyvinecopulib is required for make_student_dvine")

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
    """S3 - Mixed R-vine (heterogeneous families across trees)."""
    if not VINE_OK:
        raise RuntimeError("pyvinecopulib is required for make_mixed_rvine")

    fams   = [pv.BicopFamily.student,  pv.BicopFamily.clayton,
              pv.BicopFamily.gumbel,   pv.BicopFamily.frank,
              pv.BicopFamily.student,  pv.BicopFamily.clayton]
    params = [np.array([[0.5], [4.]]), np.array([[3.]]), np.array([[3.]]),
              np.array([[4.]]),       np.array([[0.4], [6.]]),
              np.array([[2.]])]
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
    """
    R1 - S&P500-calibrated returns:
        - block correlation (4 sectors x 5 assets)
        - Student-t heavy tails (df=4.5)
        - localized market shocks
    """
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
    """
    R2 - ERA5-calibrated meteorological data:
        - block correlation (5 clusters)
        - Student-t heavy tails (df=5.0)
        - seasonal cycle + damping window
    """
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
        X[:, c_i*c:(c_i+1)*c] += (0.3 + 0.1 * c_i) * np.sin(
            2 * np.pi * t_idx / 365)[:, None]
    X[1000:1365] *= 0.4
    return X


# =============================================================================
# Boundary conditions
# =============================================================================
def make_gaussian_factor(d=15, n=1500, n_factors=3, seed=42):
    """S4 - Gaussian Factor Model (purely linear dependencies)."""
    np.random.seed(seed)
    factors  = np.random.randn(n, n_factors)
    loadings = np.random.uniform(0.5, 0.9, size=(d, n_factors))
    noise    = np.random.randn(n, d) * 0.5
    return factors @ loadings.T + noise


def make_noisy_lowdim(d=10, n=1500, noise_ratio=0.30, seed=42):
    """S5 - Noisy Low-Dim (70% signal + 30% pure noise, shuffled)."""
    np.random.seed(seed)
    n_signal = int(d * (1 - noise_ratio))
    n_noise  = d - n_signal
    signal = np.random.randn(n, n_signal)
    noise  = np.random.randn(n, n_noise)
    X = np.hstack([signal, noise])
    idx = np.random.permutation(d)
    return X[:, idx]


def make_pure_gaussian_copula(d=15, n=1500, rho=0.5, seed=42):
    """S6 - Pure Gaussian Copula (no heavy tails, no asymmetry)."""
    np.random.seed(seed)
    Sigma = np.full((d, d), rho)
    np.fill_diagonal(Sigma, 1.0)
    L = np.linalg.cholesky(Sigma)
    Z = np.random.randn(n, d) @ L.T
    U = norm.cdf(Z)
    return norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))


def make_block_factor_student_t(d=20, n_blocks=4, n_per_block=5,
                                 rho=0.7, nu=4, n_samples=2000, seed=42):
    """
    S7 - Block Factor Student-t (no localized shocks).

    Used to isolate the effect of transient shocks (present in R1) on
    LS-Vine's performance.
    """
    np.random.seed(seed)
    factors = t_dist.rvs(df=nu, size=(n_samples, n_blocks))
    noise   = t_dist.rvs(df=nu, size=(n_samples, d))
    X = np.zeros((n_samples, d))
    for b in range(n_blocks):
        start = b * n_per_block
        end   = start + n_per_block
        X[:, start:end] = (rho * factors[:, b:b+1]
                           + np.sqrt(1 - rho**2) * noise[:, start:end])
    return X


# =============================================================================
# Train / validation / test splits
# =============================================================================
def split(X, n_tr=2000, n_vl=500):
    """Split synthetic scenarios: 2000 train / 500 val / rest test."""
    return dict(X_train=X[:n_tr],
                X_val=X[n_tr:n_tr + n_vl],
                X_test=X[n_tr + n_vl:])


def split_real(X, frac=(0.6, 0.2, 0.2)):
    """Split real/calibrated scenarios: 60% / 20% / 20%."""
    n = len(X)
    n1 = int(n * frac[0])
    n2 = int(n * (frac[0] + frac[1]))
    return dict(X_train=X[:n1],
                X_val=X[n1:n2],
                X_test=X[n2:])
