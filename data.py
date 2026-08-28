"""
Data generators for synthetic and real-world scenarios.
"""

import numpy as np
from scipy.stats import norm, t

try:
    import pyvinecopulib as pv
    VINE_AVAILABLE = True
except ImportError:
    VINE_AVAILABLE = False


def make_student_dvine(d, rho, nu, n, seed=42):
    """Generate Student-t D-vine data."""
    if not VINE_AVAILABLE:
        raise RuntimeError("pyvinecopulib is required for D-vine generation")

    pc = []
    for tree in range(d - 1):
        pc.append([
            pv.Bicop(
                family=pv.BicopFamily.student,
                parameters=np.array([[rho], [float(nu)]])
            )
            for _ in range(d - 1 - tree)
        ])

    vine = pv.Vinecop.from_structure(
        structure=pv.DVineStructure(list(range(1, d + 1))),
        pair_copulas=pc,
    )

    U = vine.simulate(n, seeds=[seed])
    return norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))


def make_mixed_rvine(d, n, seed=42):
    """
    Generate mixed R-vine data with heterogeneous dependencies.

    Steps:
    1. Generate Gaussian data with block correlation structure
    2. Fit a true R-vine with mixed families (Student, Clayton, Gumbel, Frank)
    3. Sample from the fitted R-vine
    """
    if not VINE_AVAILABLE:
        raise RuntimeError("pyvinecopulib is required for mixed R-vine generation")

    np.random.seed(seed)

    # Step 1: Gaussian data with block correlations
    Sigma = np.eye(d)
    block_size = max(1, d // 3)

    # Block 1: Strong correlation
    for i in range(block_size):
        for j in range(block_size):
            if i != j:
                Sigma[i, j] = 0.7

    # Block 2: Moderate correlation
    for i in range(block_size, 2 * block_size):
        for j in range(block_size, 2 * block_size):
            if i != j:
                Sigma[i, j] = 0.5

    # Block 3: Weak correlation
    for i in range(2 * block_size, d):
        for j in range(2 * block_size, d):
            if i != j:
                Sigma[i, j] = 0.3

    # Cross-block correlations
    for i in range(d):
        for j in range(d):
            if Sigma[i, j] == 0 and i != j:
                Sigma[i, j] = 0.15

    L = np.linalg.cholesky(Sigma)
    Z = np.random.randn(n, d) @ L.T
    U = norm.cdf(Z)

    # Step 2: Fit R-vine with mixed families
    U_safe = np.clip(U, 1e-5, 1.0 - 1e-5)

    controls = pv.FitControlsVinecop(
        family_set=[
            pv.BicopFamily.student,
            pv.BicopFamily.clayton,
            pv.BicopFamily.gumbel,
            pv.BicopFamily.frank,
        ],
        trunc_lvl=min(3, d - 1),
        selection_criterion="aic",
    )

    vine = pv.Vinecop.from_data(data=U_safe, controls=controls)

    # Step 3: Sample from fitted R-vine
    U_sim = vine.simulate(n, seeds=[seed])

    return norm.ppf(np.clip(U_sim, 1e-6, 1 - 1e-6))


def make_sp500_calibrated(n=3000, d=20, seed=42):
    """Generate synthetic S&P500-calibrated financial returns."""
    np.random.seed(seed)
    from scipy.stats import t as t_dist

    s = 4
    sectors = d // s

    Sigma = np.full((d, d), 0.25)
    for i in range(sectors):
        Sigma[i * s:(i + 1) * s, i * s:(i + 1) * s] = 0.65
    np.fill_diagonal(Sigma, 1.0)

    L = np.linalg.cholesky(Sigma)
    X = t_dist.rvs(df=4.5, size=(n, d)) @ L.T

    vols = np.random.uniform(0.20, 0.35, d) / np.sqrt(252)
    X *= vols

    # Add localized market shocks
    X[500:520] *= 3.5
    X[500:510] -= 0.025

    return X


def make_era5_calibrated(n=3000, d=15, seed=43):
    """Generate synthetic ERA5-calibrated meteorological data."""
    np.random.seed(seed)
    from scipy.stats import t as t_dist

    c = 5
    n_cl = d // c

    Sigma = np.full((d, d), 0.18)
    for i in range(n_cl):
        Sigma[i * c:(i + 1) * c, i * c:(i + 1) * c] = 0.72
    np.fill_diagonal(Sigma, 1.0)

    L = np.linalg.cholesky(Sigma)
    X = t_dist.rvs(df=5.0, size=(n, d)) @ L.T

    # Add seasonal patterns
    t_idx = np.arange(n)
    for c_i in range(n_cl):
        X[:, c_i * c:(c_i + 1) * c] += (
            (0.3 + 0.1 * c_i) * np.sin(2 * np.pi * t_idx / 365)[:, None]
        )

    # Add seasonal damping
    X[1000:1365] *= 0.4

    return X


def make_block_factor_student_t(d=20, n_blocks=4, n_per_block=5,
                                 rho=0.7, nu=4, n_samples=2000, seed=42):
    """
    Generate block-factor Student-t data without localized shocks (S7).

    This is a controlled synthetic scenario used to isolate the effect of
    localized market shocks on LS-Vine's performance.
    """
    np.random.seed(seed)
    n_factors = n_blocks

    # Common factors for each block (Student-t for heavy tails)
    factors = t.rvs(df=nu, size=(n_samples, n_factors))

    # Idiosyncratic noise
    noise = t.rvs(df=nu, size=(n_samples, d))

    # Linear combination to create block structure
    X = np.zeros((n_samples, d))
    for b in range(n_blocks):
        start = b * n_per_block
        end = start + n_per_block
        X[:, start:end] = rho * factors[:, b:b + 1] + np.sqrt(1 - rho**2) * noise[:, start:end]

    return X


def split_data(X, n_train=2000, n_val=500):
    """Split data into train, validation, and test sets."""
    return {
        "X_train": X[:n_train],
        "X_val": X[n_train:n_train + n_val],
        "X_test": X[n_train + n_val:],
    }


def split_data_real(X, frac=(0.6, 0.2, 0.2)):
    """Split real data with specified fractions."""
    n = len(X)
    n1 = int(n * frac[0])
    n2 = int(n * (frac[0] + frac[1]))
    return {
        "X_train": X[:n1],
        "X_val": X[n1:n2],
        "X_test": X[n2:],
    }
