"""
Loss functions for LS-Vine: soft Kendall's tau, rank distribution matching,
and regularization.
"""

import torch
import numpy as np


def _robust_beta_per_dim(X, eps=1e-3):
    """
    Compute dimension-specific temperature parameters for soft Kendall's tau.

    Uses median of pairwise differences for robustness to heavy tails.
    """
    n, d = X.shape
    mask = ~torch.eye(n, dtype=torch.bool, device=X.device)
    betas = []
    for k in range(d):
        diffs = (X[:, k].unsqueeze(0) - X[:, k].unsqueeze(1)).abs()
        med = diffs[mask].median()
        betas.append(1.0 / (med + eps))
    return torch.stack(betas)


def soft_tau_loss(X, X_hat, beta=1.0, frac_pairs=None, gen=None, robust=True):
    """
    Differentiable soft Kendall's tau reconstruction loss.

    Args:
        X: Original data (batch_size, d)
        X_hat: Reconstructed data (batch_size, d)
        beta: Temperature parameter
        frac_pairs: Fraction of dimension pairs to sample
        gen: Random generator for reproducibility
        robust: Use robust per-dimension beta scaling

    Returns:
        Soft tau reconstruction loss (scalar)
    """
    n, d = X.shape

    # Sample dimension pairs
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    total = len(pairs)
    n_pairs = total if frac_pairs is None else max(10, int(frac_pairs * total))

    if n_pairs < total:
        idx = (
            torch.randperm(total, generator=gen) if gen else torch.randperm(total)
        )[:n_pairs]
        pairs = [pairs[k] for k in idx.tolist()]
        scale = total / n_pairs
    else:
        scale = 1.0

    # Compute beta scaling
    beta_vec = _robust_beta_per_dim(X) if robust else torch.full((d,), float(beta), device=X.device)

    # Upper triangular mask for pairwise comparisons
    mask = torch.triu(torch.ones(n, n, device=X.device, dtype=torch.bool), diagonal=1)

    total_loss = 0.0
    for i, j in pairs:
        bi, bj = beta_vec[i], beta_vec[j]

        # Compute pairwise differences
        dxi_t = X[:, i].unsqueeze(0) - X[:, i].unsqueeze(1)
        dxj_t = X[:, j].unsqueeze(0) - X[:, j].unsqueeze(1)
        dxi_h = X_hat[:, i].unsqueeze(0) - X_hat[:, i].unsqueeze(1)
        dxj_h = X_hat[:, j].unsqueeze(0) - X_hat[:, j].unsqueeze(1)

        # Soft Kendall's tau
        tau_t = torch.sigmoid((dxi_t * dxj_t)[mask] * bi * bj).mean() * 2 - 1
        tau_h = torch.sigmoid((dxi_h * dxj_h)[mask] * bi * bj).mean() * 2 - 1

        total_loss += (tau_t - tau_h) ** 2

    return scale * total_loss / max(len(pairs), 1)


def reg_loss(Z, lam_var: float = 0.0):
    """
    Latent regularization loss.

    Penalizes deviations from zero mean and unit variance.
    """
    mu_pen = (Z.mean(0) ** 2).sum()
    if lam_var <= 0:
        return mu_pen
    log_std = torch.log(Z.std(0) + 1e-6)
    return mu_pen + lam_var * (log_std ** 2).sum()


def lambda_warmup(t, lam_max=1.0, tau=15.0):
    """Exponential warmup schedule for vine loss weight."""
    return lam_max * (1 - np.exp(-t / tau))


def soft_kendall_tau_matrix(X, beta=1.0, frac_pairs=None, gen=None, robust=True):
    """
    Compute the full soft Kendall's tau matrix for a dataset.

    Args:
        X: Data matrix (n_samples, d)
        beta: Temperature parameter
        frac_pairs: Fraction of dimension pairs to sample
        gen: Random generator
        robust: Use robust per-dimension scaling

    Returns:
        Soft Kendall's tau matrix (d, d)
    """
    n, d = X.shape

    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    total = len(pairs)
    n_pairs = total if frac_pairs is None else max(10, int(frac_pairs * total))

    if n_pairs < total:
        idx = torch.randperm(total, generator=gen)[:n_pairs] if gen else torch.randperm(total)[:n_pairs]
        pairs = [pairs[k] for k in idx.tolist()]

    beta_vec = _robust_beta_per_dim(X) if robust else torch.full((d,), float(beta), device=X.device)
    mask = ~torch.eye(n, dtype=torch.bool, device=X.device)

    T = torch.eye(d, device=X.device, dtype=X.dtype)

    for i, j in pairs:
        dxi = X[:, i].unsqueeze(0) - X[:, i].unsqueeze(1)
        dxj = X[:, j].unsqueeze(0) - X[:, j].unsqueeze(1)
        T[i, j] = (torch.sigmoid(beta_vec[i] * dxi) * torch.sigmoid(beta_vec[j] * dxj))[mask].mean()
        T[j, i] = T[i, j]

    return T


def rank_dependence_loss(X, Z, beta=1.0, frac_pairs=None, gen=None,
                         n_quantiles: int = 20, tail_weight: float = 2.0):
    """
    Rank-distribution matching loss (L_v).

    Forces the latent space to preserve the distribution of pairwise
    dependence strengths from the observed data.

    Args:
        X: Original data (n_samples, d)
        Z: Latent representations (n_samples, k)
        beta: Temperature parameter for soft Kendall's tau
        frac_pairs: Fraction of pairs to sample
        gen: Random generator
        n_quantiles: Number of quantiles to compare
        tail_weight: Weight exponent for tail emphasis

    Returns:
        Rank distribution matching loss (scalar)
    """
    Tx = soft_kendall_tau_matrix(X, beta=beta, frac_pairs=frac_pairs, gen=gen)
    Tz = soft_kendall_tau_matrix(Z, beta=beta, frac_pairs=frac_pairs, gen=gen)

    mask_x = ~torch.eye(Tx.shape[0], dtype=torch.bool, device=X.device)
    mask_z = ~torch.eye(Tz.shape[0], dtype=torch.bool, device=Z.device)

    tx_abs, tz_abs = Tx[mask_x].abs(), Tz[mask_z].abs()

    q_lin = torch.linspace(0.0, 1.0, n_quantiles, device=X.device, dtype=tx_abs.dtype)
    q_levels = q_lin ** (1.0 / max(tail_weight, 1e-6))

    return ((torch.quantile(tx_abs, q_levels) - torch.quantile(tz_abs, q_levels)) ** 2).mean()
