"""
src/losses.py

Loss functions for LS-Vine training.

Contains:
    - gaussian_pit              : differentiable Gaussian PIT
    - _robust_beta_per_dim      : robust per-dimension scale for soft Kendall
    - soft_tau_loss             : soft Kendall's tau reconstruction loss
    - soft_kendall_tau_matrix   : full soft Kendall's tau matrix
    - rank_dependence_loss      : L_v (rank-distribution matching)
    - reg_loss                  : L_reg (latent regularization)
    - lambda_warmup             : exponential warmup schedule
    - mmd_penalty               : MMD penalty for WAE / InfoVAE
"""

import numpy as np
import torch


def gaussian_pit(Z, mu=None, std=None, eps=1e-6):
    """Gaussian Probability Integral Transform (differentiable)."""
    if mu  is None: mu  = Z.mean(0, keepdim=True)
    if std is None: std = Z.std(0,  keepdim=True)
    return (0.5 * (1 + torch.erf((Z - mu) / ((std + eps) * 2.**0.5)))
            ).clamp(1e-4, 1 - 1e-4)


def _robust_beta_per_dim(X, eps=1e-3):
    """
    Compute a robust per-dimension scale for soft Kendall's tau.

    beta_k = 1 / (median_{i!=j} |X_ik - X_jk| + eps)

    Median heuristic is robust to heavy tails and outliers (unlike std),
    which is essential for scenarios like R1 (S&P500-calibrated).
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

    Enforces local pairwise concordance between X and its reconstruction
    X_hat. Subsampled over `frac_pairs * total` dimension pairs per batch.
    """
    n, d = X.shape
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    total = len(pairs)
    n_pairs = total if frac_pairs is None else max(10, int(frac_pairs * total))
    if n_pairs < total:
        idx = (torch.randperm(total, generator=gen) if gen
               else torch.randperm(total))[:n_pairs]
        pairs = [pairs[k] for k in idx.tolist()]
        scale = total / n_pairs
    else:
        scale = 1.0

    if robust:
        beta_vec = _robust_beta_per_dim(X)
    else:
        beta_vec = torch.full((d,), float(beta), device=X.device)

    mask = torch.triu(torch.ones(n, n, device=X.device, dtype=torch.bool),
                      diagonal=1)
    total_loss = 0.0
    for (i, j) in pairs:
        bi, bj = beta_vec[i], beta_vec[j]
        dxi_t = X[:, i].unsqueeze(0)     - X[:, i].unsqueeze(1)
        dxj_t = X[:, j].unsqueeze(0)     - X[:, j].unsqueeze(1)
        dxi_h = X_hat[:, i].unsqueeze(0) - X_hat[:, i].unsqueeze(1)
        dxj_h = X_hat[:, j].unsqueeze(0) - X_hat[:, j].unsqueeze(1)
        c_t = (dxi_t * dxj_t)[mask]
        c_h = (dxi_h * dxj_h)[mask]
        tau_t = torch.sigmoid(c_t * bi * bj).mean() * 2 - 1
        tau_h = torch.sigmoid(c_h * bi * bj).mean() * 2 - 1
        total_loss = total_loss + (tau_t - tau_h) ** 2

    return scale * total_loss / max(len(pairs), 1)


def soft_kendall_tau_matrix(X, beta=1.0, frac_pairs=None, gen=None,
                             robust=True):
    """
    Full differentiable soft Kendall's tau matrix (d x d).

    Uses tanh-based softsign approximation; symmetric with unit diagonal.
    """
    n, d = X.shape
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    total = len(pairs)
    n_pairs = total if frac_pairs is None else max(10, int(frac_pairs * total))
    if n_pairs < total:
        idx = (torch.randperm(total, generator=gen) if gen
               else torch.randperm(total))[:n_pairs]
        pairs = [pairs[k] for k in idx.tolist()]

    beta_vec = (_robust_beta_per_dim(X) if robust
                else torch.full((d,), float(beta), device=X.device))

    mask = ~torch.eye(n, dtype=torch.bool, device=X.device)
    T = torch.eye(d, device=X.device, dtype=X.dtype)
    for (i, j) in pairs:
        xi, xj = X[:, i], X[:, j]
        dxi = xi.unsqueeze(0) - xi.unsqueeze(1)
        dxj = xj.unsqueeze(0) - xj.unsqueeze(1)
        s = torch.tanh(beta_vec[i] * dxi) * torch.tanh(beta_vec[j] * dxj)
        v = s[mask].mean()
        T[i, j] = v
        T[j, i] = v
    return T


def rank_dependence_loss(X, Z, beta=1.0, frac_pairs=None, gen=None,
                          n_quantiles: int = 20, tail_weight: float = 2.0):
    """
    Rank-distribution matching loss L_v.

    Compares the entire quantile function of |tau| between X and Z.
    Dimension-agnostic (works when X has d columns and Z has d_lat columns).
    100% differentiable, no parametric copula family used.
    """
    Tx = soft_kendall_tau_matrix(X, beta=beta, frac_pairs=frac_pairs, gen=gen)
    Tz = soft_kendall_tau_matrix(Z, beta=beta, frac_pairs=frac_pairs, gen=gen)

    dx = Tx.shape[0]; dz = Tz.shape[0]
    mask_x = ~torch.eye(dx, dtype=torch.bool, device=X.device)
    mask_z = ~torch.eye(dz, dtype=torch.bool, device=Z.device)
    tx_abs = Tx[mask_x].abs()
    tz_abs = Tz[mask_z].abs()

    q_lin = torch.linspace(0.0, 1.0, n_quantiles, device=X.device,
                            dtype=tx_abs.dtype)
    q_levels = q_lin ** (1.0 / max(tail_weight, 1e-6))

    qx = torch.quantile(tx_abs, q_levels)
    qz = torch.quantile(tz_abs, q_levels)
    return ((qx - qz) ** 2).mean()


def reg_loss(Z, lam_var: float = 0.0):
    """Latent regularisation: mean + soft log-variance."""
    mu_pen = (Z.mean(0) ** 2).sum()
    if lam_var <= 0:
        return mu_pen
    log_std = torch.log(Z.std(0) + 1e-6)
    var_pen = (log_std ** 2).sum()
    return mu_pen + lam_var * var_pen


def lambda_warmup(t, lam_max=1.0, tau=15.0):
    """Exponential warmup schedule for L_v weight."""
    return lam_max * (1 - np.exp(-t / tau))


def mmd_penalty(z, lam=10.0, sigma=1.0):
    """RBF MMD against N(0,1). Used by WAE / InfoVAE."""
    if z.dim() == 1:
        z = z.unsqueeze(0)
    z_prior = torch.randn_like(z)
    n = z.shape[0]
    k_xx = torch.exp(-torch.cdist(z, z)**2 / (2 * sigma**2))
    k_yy = torch.exp(-torch.cdist(z_prior, z_prior)**2 / (2 * sigma**2))
    k_xy = torch.exp(-torch.cdist(z, z_prior)**2 / (2 * sigma**2))
    mmd = ((k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1) + 1e-8)
           + (k_yy.sum() - k_yy.diag().sum()) / (n * (n - 1) + 1e-8)
           - 2 * k_xy.mean())
    return lam * torch.clamp(mmd, min=0.0)
