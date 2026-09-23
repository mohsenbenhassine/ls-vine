"""
src/train.py

Training loops for LS-Vine and baseline methods.

Contains:
    - train_lsvine         : full LS-Vine training (L_rec + L_v + L_reg)
    - train_ae             : plain autoencoder (MSE only)
    - train_ae_selected    : AE with soft-tau loss, selected by validation AIC
    - train_vae            : variational autoencoder
    - train_wae            : Wasserstein autoencoder (MMD penalty)
    - train_infovae        : InfoVAE (reconstruction + MMD + KL)
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from .config import CFG, Config
from .models import LSVineNet, AENet, VAENet
from .losses import (soft_tau_loss, rank_dependence_loss, reg_loss,
                     lambda_warmup, mmd_penalty)
from .vine_utils import empirical_pit, fit_vine, vine_metrics

# -----------------------------------------------------------------------------
# Device and reproducibility
# -----------------------------------------------------------------------------
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# =============================================================================
# LS-Vine training
# =============================================================================
def train_lsvine(X_tr, X_vl, d_lat, cfg: Config = CFG,
                 verbose=True, seed=42):
    """
    Train LS-Vine with alternating optimization.

    Loss:
        L = L_rec + lambda(t) * L_v + gamma * L_reg
        L_rec = MSE + alpha * soft_tau_loss
        L_v   = rank_dependence_loss (Wasserstein-1D on |tau|)
        L_reg = mean + lam_var * log-variance

    pyvinecopulib is used EXCLUSIVELY for:
        (i)  validation AIC (checkpoint selection / early stopping)
        (ii) the returned vine (used downstream for LL / AIC / BIC / DepRec)
    It is NEVER called inside the gradient computation.

    Returns
    -------
    model, best_vine, history, best_mu, best_std
    """
    set_seed(seed)
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)

    d     = X_tr.shape[1]
    model = LSVineNet(d, d_lat, cfg.hidden).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs,
                                                 eta_min=1e-5)
    scaler = GradScaler() if USE_AMP else None

    Xtr = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    Xvl = torch.tensor(X_vl, dtype=torch.float32).to(DEVICE)

    hist = dict(epoch=[], train_loss=[], val_aic=[], lam=[], lv_train=[])
    curr_vine = best_vine = best_mu = best_std = None
    global_mu = global_std = None
    best_aic  = np.inf
    best_state = None
    patience   = 0

    try:
        for ep in range(1, cfg.epochs + 1):
            in_pre = ep <= cfg.pretrain_epochs
            lam    = 0.0 if in_pre else lambda_warmup(
                ep - cfg.pretrain_epochs, cfg.lam_max, cfg.tau_w)

            # -----------------------------------------------------------------
            # Training epoch
            # -----------------------------------------------------------------
            model.train()
            ep_loss = 0.0; ep_lv = 0.0; n_lv = 0
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), cfg.batch_size):
                xb = Xtr[perm[i:i + cfg.batch_size]]
                with autocast(enabled=USE_AMP):
                    z, xhat = model(xb)
                    Lr = (F.mse_loss(xhat, xb)
                          + cfg.alpha * soft_tau_loss(
                              xb, xhat, cfg.beta_tau, cfg.frac_pairs, gen))
                    Lg = reg_loss(z, cfg.lam_var)

                Lv = torch.zeros((), device=DEVICE, dtype=torch.float32)
                if lam > 0 and not in_pre:
                    Lv = rank_dependence_loss(
                        xb.float(), z.float(),
                        beta=cfg.beta_tau, frac_pairs=cfg.frac_pairs,
                        gen=gen, n_quantiles=cfg.n_quantiles,
                        tail_weight=cfg.tail_weight)
                    if torch.isfinite(Lv):
                        ep_lv += float(Lv); n_lv += 1
                    else:
                        Lv = torch.zeros((), device=DEVICE,
                                          dtype=torch.float32)

                loss = Lr + lam * Lv + cfg.gamma * Lg
                if not torch.isfinite(loss):
                    opt.zero_grad(); continue

                opt.zero_grad()
                if USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                ep_loss += float(loss)
            sched.step()

            # -----------------------------------------------------------------
            # Periodic vine re-estimation (for validation / checkpointing)
            # -----------------------------------------------------------------
            if ep % cfg.K == 0 and not in_pre:
                model.eval()
                with torch.no_grad():
                    Zf, _ = model(Xtr); Zf = Zf.float()
                    global_mu  = Zf.mean(0, keepdim=True)
                    global_std = Zf.std(0, keepdim=True)
                    Uf = empirical_pit(Zf.cpu().numpy()).astype(np.float64)
                try:
                    curr_vine = fit_vine(Uf, cfg.trunc)
                except Exception:
                    pass

            # -----------------------------------------------------------------
            # Validation AIC
            # -----------------------------------------------------------------
            model.eval(); val_aic = np.nan
            if curr_vine is not None:
                with torch.no_grad():
                    Zv, _ = model(Xvl)
                    Uv    = empirical_pit(
                        Zv.float().cpu().numpy()).astype(np.float64)
                try:
                    _, val_aic = vine_metrics(curr_vine, Uv)
                except Exception:
                    pass

            hist["epoch"].append(ep)
            hist["train_loss"].append(ep_loss)
            hist["val_aic"].append(val_aic)
            hist["lam"].append(lam)
            hist["lv_train"].append(ep_lv / n_lv if n_lv else np.nan)

            # -----------------------------------------------------------------
            # Early stopping (patience only after first valid estimation)
            # -----------------------------------------------------------------
            if (ep > cfg.pretrain_epochs and curr_vine is not None
                    and np.isfinite(val_aic)):
                if val_aic < best_aic:
                    best_aic   = val_aic
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    best_vine  = curr_vine
                    best_mu    = (global_mu.clone()
                                  if global_mu is not None else None)
                    best_std   = (global_std.clone()
                                  if global_std is not None else None)
                    patience   = 0
                else:
                    patience += 1
                if patience >= cfg.patience:
                    if verbose:
                        print(f"  Early stop ep {ep} "
                              f"(best AIC={best_aic:.2f})")
                    break
            else:
                patience = 0

            if verbose and ep % 50 == 0:
                print(f"  ep {ep:4d} | val_aic={val_aic:.2f} | "
                      f"lam={lam:.3f} | Lv_train={hist['lv_train'][-1]:.4f}")

        if best_state:
            model.load_state_dict({k: v.to(DEVICE)
                                   for k, v in best_state.items()})

    finally:
        del Xtr, Xvl
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return model, best_vine, hist, best_mu, best_std


# =============================================================================
# Plain autoencoder
# =============================================================================
def train_ae(X_tr, X_vl, d_lat, cfg=CFG, seed=42):
    """Train a plain Autoencoder (MSE reconstruction only)."""
    set_seed(seed)
    d     = X_tr.shape[1]
    model = AENet(d, d_lat, cfg.hidden).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs,
                                                 eta_min=1e-5)
    scaler = GradScaler() if USE_AMP else None

    Xtr = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    Xvl = torch.tensor(X_vl, dtype=torch.float32).to(DEVICE)

    best_loss  = np.inf
    best_state = None
    patience   = 0

    try:
        for ep in range(1, cfg.epochs + 1):
            model.train(); ep_loss = 0.0
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), cfg.batch_size):
                xb = Xtr[perm[i:i + cfg.batch_size]]
                with autocast(enabled=USE_AMP):
                    _, xhat = model(xb)
                    loss    = F.mse_loss(xhat, xb)
                opt.zero_grad()
                if USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                ep_loss += float(loss)
            sched.step()

            model.eval()
            with torch.no_grad():
                _, xhat_v = model(Xvl)
                val_loss  = float(F.mse_loss(xhat_v, Xvl))

            if val_loss < best_loss:
                best_loss  = val_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                patience   = 0
            else:
                patience += 1
            if patience >= cfg.patience:
                break

        if best_state:
            model.load_state_dict({k: v.to(DEVICE)
                                   for k, v in best_state.items()})
    finally:
        del Xtr, Xvl
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return model


# =============================================================================
# AE with soft-tau loss, checkpoint selected by vine AIC
# =============================================================================
def train_ae_selected(X_tr, X_vl, d_lat, cfg=CFG, seed=42, verbose=False):
    """
    Ablation baseline: AE trained with MSE + soft-tau loss, but the
    checkpoint is selected by the validation AIC of a flexible vine.

    Isolates the contribution of L_v (rank-distribution matching) from the
    contribution of simple checkpoint selection by vine AIC.
    """
    set_seed(seed)
    d     = X_tr.shape[1]
    model = AENet(d, d_lat, cfg.hidden).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs,
                                                 eta_min=1e-5)
    gen   = torch.Generator(device="cpu"); gen.manual_seed(seed)

    Xtr = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    Xvl = torch.tensor(X_vl, dtype=torch.float32).to(DEVICE)

    curr_vine = best_vine = best_mu = best_std = None
    global_mu = global_std = None
    best_aic  = np.inf
    best_state = None
    patience   = 0

    try:
        for ep in range(1, cfg.epochs + 1):
            model.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), cfg.batch_size):
                xb = Xtr[perm[i:i + cfg.batch_size]]
                z, xhat = model(xb)
                loss = (F.mse_loss(xhat, xb)
                        + cfg.alpha * soft_tau_loss(
                            xb, xhat, cfg.beta_tau, cfg.frac_pairs, gen))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            if ep % cfg.K == 0 and ep > cfg.pretrain_epochs:
                model.eval()
                with torch.no_grad():
                    Zf, _ = model(Xtr); Zf = Zf.float()
                    global_mu  = Zf.mean(0, keepdim=True)
                    global_std = Zf.std(0, keepdim=True)
                    Uf = empirical_pit(Zf.cpu().numpy()).astype(np.float64)
                try:
                    curr_vine = fit_vine(Uf, cfg.trunc)
                except Exception:
                    pass

            model.eval(); val_aic = np.nan
            if curr_vine is not None:
                with torch.no_grad():
                    Zv, _ = model(Xvl)
                    Uv    = empirical_pit(
                        Zv.float().cpu().numpy()).astype(np.float64)
                try:
                    _, val_aic = vine_metrics(curr_vine, Uv)
                except Exception:
                    pass

            if (ep > cfg.pretrain_epochs and curr_vine is not None
                    and np.isfinite(val_aic)):
                if val_aic < best_aic:
                    best_aic   = val_aic
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    best_vine  = curr_vine
                    best_mu    = (global_mu.clone()
                                  if global_mu is not None else None)
                    best_std   = (global_std.clone()
                                  if global_std is not None else None)
                    patience   = 0
                else:
                    patience += 1
                if patience >= cfg.patience:
                    if verbose:
                        print(f"  [AE-Vine-Selected] Early stop ep {ep} "
                              f"(best AIC={best_aic:.2f})")
                    break
            else:
                patience = 0

        if best_state:
            model.load_state_dict({k: v.to(DEVICE)
                                   for k, v in best_state.items()})
    finally:
        del Xtr, Xvl
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return model, best_vine, best_mu, best_std


# =============================================================================
# Variational autoencoder
# =============================================================================
def train_vae(X_tr, X_vl, d_lat, cfg=CFG, seed=42):
    """Train a VAE with beta-weighted ELBO."""
    set_seed(seed)
    beta  = cfg.vae_beta
    d     = X_tr.shape[1]
    model = VAENet(d, d_lat, cfg.hidden).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs,
                                                 eta_min=1e-5)

    Xtr = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    Xvl = torch.tensor(X_vl, dtype=torch.float32).to(DEVICE)

    best_loss  = np.inf
    best_state = None
    patience   = 0

    try:
        for ep in range(1, cfg.epochs + 1):
            model.train(); ep_loss = 0.0
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), cfg.batch_size):
                xb = Xtr[perm[i:i + cfg.batch_size]]
                _, xhat, mu, lv = model(xb)
                rec  = F.mse_loss(xhat, xb)
                kl   = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                loss = rec + beta * kl
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); ep_loss += float(loss)
            sched.step()

            model.eval()
            with torch.no_grad():
                _, xv, mv, lv_v = model(Xvl)
                kl_val   = -0.5 * torch.mean(1 + lv_v - mv.pow(2)
                                              - lv_v.exp())
                val_loss = float(F.mse_loss(xv, Xvl) + beta * kl_val)

            if val_loss < best_loss:
                best_loss  = val_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                patience   = 0
            else:
                patience += 1
            if patience >= cfg.patience:
                break

        if best_state:
            model.load_state_dict({k: v.to(DEVICE)
                                   for k, v in best_state.items()})
    finally:
        del Xtr, Xvl
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return model


# =============================================================================
# Wasserstein autoencoder
# =============================================================================
def train_wae(X_tr, X_vl, d_lat, cfg=CFG, seed=42):
    """Train a Wasserstein Autoencoder (MMD penalty to N(0,1))."""
    set_seed(seed)
    d     = X_tr.shape[1]
    model = AENet(d, d_lat, cfg.hidden).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs,
                                                 eta_min=1e-5)

    Xtr = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    Xvl = torch.tensor(X_vl, dtype=torch.float32).to(DEVICE)

    best_loss  = np.inf
    best_state = None
    patience   = 0

    try:
        for ep in range(1, cfg.epochs + 1):
            model.train(); ep_loss = 0.0
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), cfg.batch_size):
                xb = Xtr[perm[i:i + cfg.batch_size]]
                z, xhat = model(xb)
                loss = (F.mse_loss(xhat, xb)
                        + mmd_penalty(z, lam=cfg.wae_lambda))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); ep_loss += float(loss)
            sched.step()

            model.eval()
            with torch.no_grad():
                z_v, xhat_v = model(Xvl)
                val_loss = float(F.mse_loss(xhat_v, Xvl)
                                 + mmd_penalty(z_v, lam=cfg.wae_lambda))

            if val_loss < best_loss:
                best_loss  = val_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                patience   = 0
            else:
                patience += 1
            if patience >= cfg.patience:
                break

        if best_state:
            model.load_state_dict({k: v.to(DEVICE)
                                   for k, v in best_state.items()})
    finally:
        del Xtr, Xvl
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return model


# =============================================================================
# InfoVAE
# =============================================================================
def train_infovae(X_tr, X_vl, d_lat, cfg=CFG, seed=42):
    """Train an InfoVAE: reconstruction + MMD + beta * KL."""
    set_seed(seed)
    d     = X_tr.shape[1]
    model = VAENet(d, d_lat, cfg.hidden).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs,
                                                 eta_min=1e-5)

    Xtr = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    Xvl = torch.tensor(X_vl, dtype=torch.float32).to(DEVICE)

    best_loss  = np.inf
    best_state = None
    patience   = 0

    try:
        for ep in range(1, cfg.epochs + 1):
            model.train(); ep_loss = 0.0
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), cfg.batch_size):
                xb = Xtr[perm[i:i + cfg.batch_size]]
                z, xhat, mu, lv = model(xb)
                rec_loss = F.mse_loss(xhat, xb)
                kl_loss  = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                mmd_loss = mmd_penalty(z, lam=cfg.wae_lambda)
                loss = rec_loss + mmd_loss + cfg.vae_beta * kl_loss
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); ep_loss += float(loss)
            sched.step()

            model.eval()
            with torch.no_grad():
                _, xv, mv, lv_v = model(Xvl)
                val_loss = float(
                    F.mse_loss(xv, Xvl)
                    + mmd_penalty(mv, lam=cfg.wae_lambda)
                    + cfg.vae_beta * (-0.5 * torch.mean(
                        1 + lv_v - mv.pow(2) - lv_v.exp())))

            if val_loss < best_loss:
                best_loss  = val_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                patience   = 0
            else:
                patience += 1
            if patience >= cfg.patience:
                break

        if best_state:
            model.load_state_dict({k: v.to(DEVICE)
                                   for k, v in best_state.items()})
    finally:
        del Xtr, Xvl
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return model
