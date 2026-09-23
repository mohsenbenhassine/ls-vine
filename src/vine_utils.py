"""
src/vine_utils.py

Vine copula utilities built on top of pyvinecopulib.

Contains:
    - empirical_pit         : rank-based empirical PIT (non-parametric)
    - fit_vine              : fit a vine copula via AIC over 5 families
    - vine_metrics          : return (log-likelihood, AIC) on pseudo-obs
    - vine_bic              : return BIC on pseudo-obs
    - vine_nparams          : number of parameters of a fitted vine
    - kendall_matrix        : empirical Kendall's tau matrix
    - get_t1_edges          : extract edges of the first tree
    - structural_stability  : mean overlap across multiple tree-1 edge sets
"""

import numpy as np
from itertools import combinations
from scipy.stats import kendalltau, rankdata

try:
    import pyvinecopulib as pv
    VINE_OK = True
except ImportError:
    VINE_OK = False


def empirical_pit(X):
    """Empirical Probability Integral Transform (rank-based, non-parametric)."""
    n, d = X.shape
    return np.column_stack([rankdata(X[:, j]) / (n + 1) for j in range(d)])


def fit_vine(U: np.ndarray, trunc: int = 3) -> "pv.Vinecop":
    """
    Fit a vine copula via AIC selection over 5 families
    (Gaussian, Student-t, Clayton, Gumbel, Frank).

    IMPORTANT: `from_data` in pyvinecopulib 1.0.0 requires:
      - positional argument (not `data=...`)
      - Fortran-order float64 array
    """
    if not VINE_OK:
        raise RuntimeError("pyvinecopulib is not available")

    U_safe = np.asfortranarray(np.clip(U.astype(np.float64), 1e-5, 1 - 1e-5))
    ctrl = pv.FitControlsVinecop(
        family_set=[pv.BicopFamily.gaussian, pv.BicopFamily.student,
                    pv.BicopFamily.clayton,  pv.BicopFamily.gumbel,
                    pv.BicopFamily.frank],
        trunc_lvl=trunc,
        selection_criterion="aic",
    )
    return pv.Vinecop.from_data(U_safe, controls=ctrl)


def vine_metrics(vine, U: np.ndarray):
    """Return (mean log-likelihood, AIC) on pseudo-observations U."""
    if U is None or len(U) == 0:
        return np.nan, np.inf
    U64 = U.astype(np.float64)
    ll  = vine.loglik(U64) / len(U64)
    aic = -2 * vine.loglik(U64) + 2 * vine.npars
    return ll, aic


def vine_bic(vine, U: np.ndarray) -> float:
    """Bayesian Information Criterion on pseudo-observations U."""
    if U is None or len(U) == 0:
        return np.inf
    U64 = U.astype(np.float64)
    n   = len(U64)
    return -2 * vine.loglik(U64) + vine.npars * np.log(n)


def vine_nparams(vine) -> float:
    """Return number of parameters of a fitted vine."""
    return vine.npars


def kendall_matrix(X: np.ndarray) -> np.ndarray:
    """Empirical Kendall's tau matrix for data X of shape (n, d)."""
    d = X.shape[1]
    return np.array([[kendalltau(X[:, i], X[:, j])[0] for j in range(d)]
                     for i in range(d)])


def get_t1_edges(vine):
    """Extract the set of edges of tree T_1 from a fitted vine."""
    mat = vine.matrix
    k = mat.shape[0]
    return {frozenset([int(mat[k - 1 - j, j]) - 1, int(mat[0, j]) - 1])
            for j in range(k - 1)}


def structural_stability(edge_sets):
    """Mean Jaccard-like overlap across a list of tree-1 edge sets."""
    R = len(edge_sets)
    if R < 2:
        return 1.0
    k = max(len(e) for e in edge_sets)
    pairs = list(combinations(range(R), 2))
    return sum(len(edge_sets[a] & edge_sets[b]) / k
               for a, b in pairs) / len(pairs)
