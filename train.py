"""
Training loops for LS-Vine and baseline methods.
"""

import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from .config import CFG, Config
from .models import LSVineNet, AENet, VAENet
from .losses import soft_tau_loss, reg_loss, lambda_warmup, rank_dependence_loss
from .utils import empirical_pit, fit_vine, vine_metrics, fit_vine_robust


def _memory_usage():
    """Get current GPU memory usage in MB if available."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


def train_lsvine(X_tr, X_vl, d_lat, cfg=CFG, verbose=True, seed=42):
    """
    Train LS-Vine model with alternating optimization.

    Args:
        X_tr: Training data (n_train, d)
        X_vl: Validation data (n_val, d)
        d_lat: Latent dimension
        cfg: Configuration object
        verbose: Print progress
        seed: Random seed

    Returns:
        model: Trained LSVineNet
        vine: Fitted vine copula
        hist: Training history
        best_mu: Best latent mean
        best_std: Best latent std
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    d = X_tr.shape[1]
    model = LSVineNet(d, d_lat, cfg.hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-5)
    scaler = GradScaler() if use_amp else None

    Xtr = torch.tensor(X_tr, dtype=torch.float32).to(device)
    Xvl = torch.tensor(X_vl, dtype=torch.float32).to(device)

    hist = {
        "epoch": [],
        "train_loss": [],
        "val_aic": [],
        "lam": [],
        "lv_train": [],
    }

    curr_vine = None
    best_vine = None
    best_aic = np.inf
    best_state = None
    best_mu = None
    best_std = None
    patience = 0
    global_mu = None
    global_std = None

    try:
        for ep in range(1, cfg.epochs + 1):
            in_pre = ep <= cfg.pretrain_epochs
            lam = 0.0 if in_pre else lambda_warmup(
                ep - cfg.pretrain_epochs, cfg.lam_max, cfg.tau_w
            )

            model.train()
            ep_loss = 0.0
            ep_lv = 0.0
            n_lv = 0
            perm = torch.randperm(len(Xtr))

            for i in range(0, len(Xtr), cfg.batch_size):
                xb = Xtr[perm[i:i + cfg.batch_size]]

                with autocast(enabled=use_amp):
                    z, xhat = model(xb)
                    Lr = F.mse_loss(xhat, xb) + cfg.alpha * soft_tau_loss(
                        xb, xhat, cfg.beta_tau, cfg.frac_pairs, gen
                    )
                    Lg = reg_loss(z, cfg.lam_var)

                    Lv = torch.zeros((), device=device, dtype=torch.float32)
                    if lam > 0 and not in_pre:
                        Lv = rank_dependence_loss(
                            xb.float(), z.float(),
                            beta=cfg.beta_tau,
                            frac_pairs=cfg.frac_pairs,
                            gen=gen,
                            n_quantiles=cfg.n_quantiles,
                            tail_weight=cfg.tail_weight,
                        )
                        if torch.isfinite(Lv):
                            ep_lv += float(Lv)
                            n_lv += 1
                        else:
                            Lv = torch.zeros((), device=device, dtype=torch.float32)

                loss = Lr + lam * Lv + cfg.gamma * Lg

                if not torch.isfinite(loss):
                    opt.zero_grad()
                    continue

                opt.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                ep_loss += float(loss)

            sched.step()

            # Periodic vine re-estimation
            if ep % cfg.K == 0 and VINE_AVAILABLE and not in_pre:
                model.eval()
                with torch.no_grad():
                    Zf, _ = model(Xtr)
                    Zf = Zf.float()
                    global_mu = Zf.mean(0, keepdim=True)
                    global_std = Zf.std(0, keepdim=True)
                    Uf = empirical_pit(Zf.cpu().numpy()).astype(np.float64)

                curr_vine = fit_vine_robust(Uf, cfg.trunc)

            # Validation
            model.eval()
            val_aic = np.nan
            if curr_vine is not None:
                with torch.no_grad():
                    Zv, _ = model(Xvl)
                    Uv = empirical_pit(Zv.float().cpu().numpy()).astype(np.float64)
                _, val_aic = vine_metrics(curr_vine, Uv)

            hist["epoch"].append(ep)
            hist["train_loss"].append(ep_loss)
            hist["val_aic"].append(val_aic)
            hist["lam"].append(lam)
            hist["lv_train"].append(ep_lv / n_lv if n_lv else np.nan)

            if np.isfinite(val_aic) and val_aic < best_aic:
                best_aic = val_aic
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_vine = curr_vine
                best_mu = global_mu.clone() if global_mu is not None else None
                best_std = global_std.clone() if global_std is not None else None
                patience = 0
            else:
                patience += 1

            if patience >= cfg.patience:
                if verbose:
                    print(f"  Early stop ep {ep} (best AIC={best_aic:.2f})")
                break

            if verbose and ep % 50 == 0:
                print(f"  ep {ep:4d} | val_aic={val_aic:.2f} | lam={lam:.3f}")

        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items
