"""
Utility functions: PIT, vine fitting, metrics.
"""

import numpy as np
from scipy.stats import rankdata, kendalltau
import warnings

try:
    import pyvinecopulib as pv
    VINE_AVAILABLE = True
except ImportError:
    VINE_AVAILABLE = False
    warnings.warn("pyvinecopulib not available. Vine functionality will be limited.")


def empirical_pit(X):
    """Empirical Probability Integral Transform (rank-based)."""
    n, d = X.shape
    return np.column_stack([rankdata(X[:, j]) / (n + 1) for j in range(d)])


def fit_vine(U, trunc_lvl=3):
    """Fit a vine copula using AIC selection over 5 families."""
    if not VINE_AVAILABLE:
        raise RuntimeError("pyvinecopulib is not available")

    U_safe = np.clip(U, 1e-5, 1.0 - 1e-5)

    ctrl = pv.FitControlsVinecop(
        family_set=[
            pv.BicopFamily.gaussian,
            pv.BicopFamily.student,
            pv.BicopFamily.clayton,
            pv.BicopFamily.gumbel,
            pv.BicopFamily.frank,
        ],
        trunc_lvl=min(trunc_lvl, U_safe.shape[1] - 1),
        selection_criterion="aic",
    )
    return pv.Vinecop.from_data(data=U_safe, controls=ctrl)


def fit_vine_robust(U, trunc_lvl=3, verbose=False):
    """Robust vine fitting with fallback on simpler models."""
    if not VINE_AVAILABLE:
        return None

    if U is None or len(U) == 0 or U.shape[1] < 2:
        return None

    U_safe = np.clip(U, 1e-5, 1.0 - 1e-5)

    # Remove constant columns
    var = np.var(U_safe, axis=0)
    if (var < 1e-8).any():
        U_safe = U_safe[:, var > 1e-8]
        if U_safe.shape[1] < 2:
            return None

    try:
        return fit_vine(U_safe, trunc_lvl)
    except Exception:
        # Fallback: trunc_lvl=1 with only Gaussian + Student
        try:
            ctrl = pv.FitControlsVinecop(
                family_set=[pv.BicopFamily.gaussian, pv.BicopFamily.student],
                trunc_lvl=1,
                selection_criterion="aic",
            )
            return pv.Vinecop.from_data(data=U_safe, controls=ctrl)
        except Exception:
            return None


def vine_metrics(vine, U):
    """Compute log-likelihood and AIC for a vine model on data U."""
    if U is None or len(U) == 0:
        return np.nan, np.inf

    U_safe = np.clip(U, 1e-5, 1.0 - 1e-5)

    try:
        ll = vine.loglik(U_safe) / len(U_safe)
        if np.isnan(ll):
            return np.nan, np.inf
        aic = -2 * vine.loglik(U_safe) + 2 * vine.npars
        return ll, aic
    except Exception:
        return np.nan, np.inf


def vine_bic(vine, U):
    """Compute BIC for a vine model on data U."""
    if U is None or len(U) == 0:
        return np.inf

    U_safe = np.clip(U, 1e-5, 1.0 - 1e-5)

    try:
        bic = -2 * vine.loglik(U_safe) + vine.npars * np.log(len(U_safe))
        return np.nan if np.isnan(bic) else bic
    except Exception:
        return np.inf


def vine_nparams(vine):
    """Get the number of parameters in a vine model."""
    return vine.npars


def kendall_matrix(X):
    """Compute the full Kendall's tau matrix for a dataset."""
    d = X.shape[1]
    return np.array([[kendalltau(X[:, i], X[:, j])[0] for j in range(d)] for i in range(d)])
