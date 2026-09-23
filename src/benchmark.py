"""
src/benchmark.py

Main experiment runner for LS-Vine.

Runs all (method x scenario x seed) combinations and saves raw results
to results/results.csv.

Contains:
    - METHODS   : list of 12 benchmarked methods
    - SCENARIOS : registry of 9 scenarios
    - select_d_lat  : automatic latent-dimension selection via PCA
    - run_method    : run one method on one (scenario, seed, d_lat)
    - run_all       : run all combinations

Usage
-----
python src/benchmark.py
"""

import time
import gc
import numpy as np
import pandas as pd
import torch

from .config import CFG, Config
from .vine_utils import (empirical_pit, fit_vine, vine_metrics, vine_bic,
                          vine_nparams, kendall_matrix)
from .train import (train_lsvine, train_ae, train_ae_selected, train_vae,
                     train_wae, train_infovae, DEVICE)
from .datasets import (make_student_dvine, make_mixed_rvine,
                        make_sp500_calibrated, make_era5_calibrated,
                        make_gaussian_factor, make_noisy_lowdim,
                        make_pure_gaussian_copula, make_block_factor_student_t,
                        split, split_real)

from sklearn.decomposition import PCA, FastICA, FactorAnalysis, KernelPCA

try:
    import pyvinecopulib as pv
    VINE_OK = True
except ImportError:
    VINE_OK = False


# =============================================================================
# Method registry
# =============================================================================
METHODS = [
    "LS-Vine",
    "AE-Vine-Selected",
    "PCA-Vine",
    "ICA-Vine",
    "FA-Vine",
    "KPCA-Vine",
    "AE-Vine",
    "VAE-Vine",
    "WAE-Vine",
    "InfoVAE-Vine",
    "Vine-Direct",
    "Vine-Truncated",
]


# =============================================================================
# Scenario registry
# =============================================================================
SCENARIOS = {
    "S1": dict(label="S1 - Student-t D-vine (d=10, nu=4)",
               builder=lambda s: make_student_dvine(10, 0.4, 4, 3500, s),
               split_fn=split, d_lat=4),
    "S2": dict(label="S2 - Student-t D-vine (d=20, nu=4)",
               builder=lambda s: make_student_dvine(20, 0.4, 4, 3500, s),
               split_fn=split, d_lat=5),
    "S3": dict(label="S3 - Mixed R-vine (d=12, heterogeneous)",
               builder=lambda s: make_mixed_rvine(12, 3500, s),
               split_fn=split, d_lat=4),
    "R1": dict(label="R1 - S&P500-calibrated (d=20)",
               builder=lambda s: make_sp500_calibrated(1500, 20, s),
               split_fn=split_real, d_lat=5),
    "R2": dict(label="R2 - ERA5-calibrated (d=15)",
               builder=lambda s: make_era5_calibrated(3000, 15, s),
               split_fn=split_real, d_lat=4),
    "S4": dict(label="S4 - Gaussian Factor (d=15)",
               builder=lambda s: make_gaussian_factor(15, 1500, 3, s),
               split_fn=split_real, d_lat=5),
    "S5": dict(label="S5 - Noisy Low-Dim (d=10, 30% noise)",
               builder=lambda s: make_noisy_lowdim(10, 1500, 0.30, s),
               split_fn=split_real, d_lat=4),
    "S6": dict(label="S6 - Pure Gaussian Copula (d=15)",
               builder=lambda s: make_pure_gaussian_copula(15, 1500, 0.5, s),
               split_fn=split_real, d_lat=5),
    "S7": dict(label="S7 - Block Factor Student-t (d=20, no shocks)",
               builder=lambda s: make_block_factor_student_t(
                   20, 4, 5, 0.7, 4, 2000, s),
               split_fn=split_real, d_lat=10),
}


# =============================================================================
# Automatic latent-dimension selection
# =============================================================================
def select_d_lat(X_train, var_threshold=0.90, d_min=2, d_max=None):
    """
    Choose d_lat as the smallest dimension explaining >= var_threshold
    of the variance of X_train (via PCA). Reproducible and justifiable.
    """
    d_max  = d_max or max(d_min, X_train.shape[1] // 2)
    n_comp = min(d_max, X_train.shape[1], X_train.shape[0])
    pca = PCA(n_components=n_comp).fit(X_train)
    cum = np.cumsum(pca.explained_variance_ratio_)
    d_lat = int(np.searchsorted(cum, var_threshold) + 1)
    return int(np.clip(d_lat, d_min, d_max))


# =============================================================================
# Helper: memory monitoring
# =============================================================================
def _mem_mb():
    if DEVICE.type == "cuda":
        return torch.cuda.memory_allocated() / 1e6
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        return 0.0


# =============================================================================
# run_method: one method x one (scenario, seed, d_lat)
# =============================================================================
def run_method(name, X_tr, X_vl, X_ts, d_lat, cfg=CFG, seed=42):
    """Run a single method and return a flat dict of metrics."""
    result = dict(method=name, d_lat=d_lat, seed=seed,
                  ll=np.nan, nll=np.nan, aic=np.nan, bic=np.nan,
                  n_params=np.nan, t_train=np.nan, t_sample=np.nan,
                  mem_mb=np.nan, rmse_corr=np.nan, tau_err=np.nan,
                  dep_rec=np.nan, stability=np.nan)

    d = X_tr.shape[1]
    U_tr_emp = empirical_pit(X_tr)
    U_ts_emp = empirical_pit(X_ts)

    tau_orig  = kendall_matrix(X_ts)
    corr_orig = np.corrcoef(X_ts.T)

    t0   = time.time()
    mem0 = _mem_mb()

    try:
        # ---------------------------------------------------------------------
        # LS-Vine
        # ---------------------------------------------------------------------
        if name == "LS-Vine":
            model, vine, hist, gmu, gstd = train_lsvine(
                X_tr, X_vl, d_lat, cfg, verbose=True, seed=seed)
            result["t_train"] = time.time() - t0
            if vine is None:
                return result

            Xts_t = torch.tensor(X_ts, dtype=torch.float32).to(DEVICE)
            model.eval()
            with torch.no_grad():
                Zts, Xhat = model(Xts_t)
                Uts     = empirical_pit(
                    Zts.float().cpu().numpy()).astype(np.float64)
                Xhat_np = Xhat.float().cpu().numpy()

            ll, aic = vine_metrics(vine, Uts)
            bic     = vine_bic(vine, Uts)
            n_prm   = vine_nparams(vine)

            ts0 = time.time()
            try:
                vine.simulate(500, seeds=[seed])
                result["t_sample"] = time.time() - ts0
            except Exception:
                pass

            tau_rec  = kendall_matrix(Xhat_np)
            corr_rec = np.corrcoef(Xhat_np.T)
            result.update(
                ll=ll, nll=-ll, aic=aic, bic=bic, n_params=n_prm,
                dep_rec=np.linalg.norm(tau_orig - tau_rec, "fro"),
                tau_err=np.abs(tau_orig - tau_rec)[
                    np.triu_indices(d, 1)].mean(),
                rmse_corr=np.sqrt(np.mean((corr_orig - corr_rec) ** 2)))
            del model

        # ---------------------------------------------------------------------
        # PCA / ICA / FA
        # ---------------------------------------------------------------------
        elif name in ["PCA-Vine", "ICA-Vine", "FA-Vine"]:
            if name == "PCA-Vine":
                proj = PCA(n_components=d_lat, random_state=seed).fit(X_tr)
            elif name == "ICA-Vine":
                proj = FastICA(n_components=d_lat, random_state=seed,
                               whiten="unit-variance").fit(X_tr)
            else:
                proj = FactorAnalysis(n_components=d_lat,
                                      random_state=seed).fit(X_tr)

            Ztr = proj.transform(X_tr); Zts = proj.transform(X_ts)
            Utr = empirical_pit(Ztr).astype(np.float64)
            Uts = empirical_pit(Zts).astype(np.float64)
            vine = fit_vine(Utr, cfg.trunc)
            result["t_train"] = time.time() - t0

            ll, aic = vine_metrics(vine, Uts)
            bic     = vine_bic(vine, Uts)
            ts0 = time.time()
            try:
                vine.simulate(500, seeds=[seed])
                result["t_sample"] = time.time() - ts0
            except Exception:
                pass

            if name != "FA-Vine":
                Xhat = proj.inverse_transform(Zts)
            else:
                Xhat = Zts @ proj.components_ + proj.mean_
            tau_rec  = kendall_matrix(Xhat)
            corr_rec = np.corrcoef(Xhat.T)
            result.update(
                ll=ll, nll=-ll, aic=aic, bic=bic,
                n_params=vine_nparams(vine),
                dep_rec=np.linalg.norm(tau_orig - tau_rec, "fro"),
                tau_err=np.abs(tau_orig - tau_rec)[
                    np.triu_indices(d, 1)].mean(),
                rmse_corr=np.sqrt(np.mean((corr_orig - corr_rec) ** 2)))

        # ---------------------------------------------------------------------
        # KPCA
        # ---------------------------------------------------------------------
        elif name == "KPCA-Vine":
            kpca = KernelPCA(n_components=d_lat, kernel="rbf", gamma=1.0 / d,
                             fit_inverse_transform=True,
                             random_state=seed).fit(X_tr)
            Ztr = kpca.transform(X_tr); Zts = kpca.transform(X_ts)
            Utr = empirical_pit(Ztr).astype(np.float64)
            Uts = empirical_pit(Zts).astype(np.float64)
            vine = fit_vine(Utr, cfg.trunc)
            result["t_train"] = time.time() - t0

            ll, aic = vine_metrics(vine, Uts)
            bic     = vine_bic(vine, Uts)
            ts0 = time.time()
            try:
                vine.simulate(500, seeds=[seed])
                result["t_sample"] = time.time() - ts0
            except Exception:
                pass

            Xhat = kpca.inverse_transform(Zts)
            tau_rec  = kendall_matrix(Xhat)
            corr_rec = np.corrcoef(Xhat.T)
            result.update(
                ll=ll, nll=-ll, aic=aic, bic=bic,
                n_params=vine_nparams(vine),
                dep_rec=np.linalg.norm(tau_orig - tau_rec, "fro"),
                tau_err=np.abs(tau_orig - tau_rec)[
                    np.triu_indices(d, 1)].mean(),
                rmse_corr=np.sqrt(np.mean((corr_orig - corr_rec) ** 2)))

        # ---------------------------------------------------------------------
        # AE-Vine
        # ---------------------------------------------------------------------
        elif name == "AE-Vine":
            model = train_ae(X_tr, X_vl, d_lat, cfg, seed)
            result["t_train"] = time.time() - t0
            model.eval()
            Xtr_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
            Xts_t = torch.tensor(X_ts, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                Ztr_t, _      = model(Xtr_t)
                Zts_t, Xhat_t = model(Xts_t)
                Ztr_np  = Ztr_t.float().cpu().numpy()
                Zts_np  = Zts_t.float().cpu().numpy()
                Xhat_np = Xhat_t.float().cpu().numpy()

            Utr = empirical_pit(Ztr_np).astype(np.float64)
            Uts = empirical_pit(Zts_np).astype(np.float64)
            vine = fit_vine(Utr, cfg.trunc)
            ll, aic = vine_metrics(vine, Uts)
            bic     = vine_bic(vine, Uts)
            ts0 = time.time()
            try:
                vine.simulate(500, seeds=[seed])
                result["t_sample"] = time.time() - ts0
            except Exception:
                pass

            tau_rec  = kendall_matrix(Xhat_np)
            corr_rec = np.corrcoef(Xhat_np.T)
            result.update(
                ll=ll, nll=-ll, aic=aic, bic=bic,
                n_params=vine_nparams(vine),
                dep_rec=np.linalg.norm(tau_orig - tau_rec, "fro"),
                tau_err=np.abs(tau_orig - tau_rec)[
                    np.triu_indices(d, 1)].mean(),
                rmse_corr=np.sqrt(np.mean((corr_orig - corr_rec) ** 2)))
            del model

        # ---------------------------------------------------------------------
        # AE-Vine-Selected (ablation baseline)
        # ---------------------------------------------------------------------
        elif name == "AE-Vine-Selected":
            model, vine, gmu, gstd = train_ae_selected(
                X_tr, X_vl, d_lat, cfg, seed, verbose=True)
            result["t_train"] = time.time() - t0
            if vine is None:
                return result

            Xts_t = torch.tensor(X_ts, dtype=torch.float32).to(DEVICE)
            model.eval()
            with torch.no_grad():
                Zts, Xhat = model(Xts_t)
                Uts     = empirical_pit(
                    Zts.float().cpu().numpy()).astype(np.float64)
                Xhat_np = Xhat.float().cpu().numpy()
            ll, aic = vine_metrics(vine, Uts)
            bic     = vine_bic(vine, Uts)
            ts0 = time.time()
            try:
                vine.simulate(500, seeds=[seed])
                result["t_sample"] = time.time() - ts0
            except Exception:
                pass

            tau_rec  = kendall_matrix(Xhat_np)
            corr_rec = np.corrcoef(Xhat_np.T)
            result.update(
                ll=ll, nll=-ll, aic=aic, bic=bic,
                n_params=vine_nparams(vine),
                dep_rec=np.linalg.norm(tau_orig - tau_rec, "fro"),
                tau_err=np.abs(tau_orig - tau_rec)[
                    np.triu_indices(d, 1)].mean(),
                rmse_corr=np.sqrt(np.mean((corr_orig - corr_rec) ** 2)))
            del model

        # ---------------------------------------------------------------------
        # VAE-Vine (uses posterior mean mu for PIT + reconstruction)
        # ---------------------------------------------------------------------
        elif name == "VAE-Vine":
            model = train_vae(X_tr, X_vl, d_lat, cfg, seed)
            result["t_train"] = time.time() - t0
            model.eval()
            Xtr_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
            Xts_t = torch.tensor(X_ts, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                _, _, mu_tr, lv_tr = model(Xtr_t)
                _, _, mu_ts, _     = model(Xts_t)
                mu_tr_np = mu_tr.float().cpu().numpy()
                mu_ts_np = mu_ts.float().cpu().numpy()
                lv_tr_np = lv_tr.float().cpu().numpy()
                Xhat_t   = model.decoder(mu_ts)
                Xhat_np  = Xhat_t.float().cpu().numpy()

            kl_per_dim  = -0.5 * (1 + lv_tr_np - mu_tr_np**2
                                   - np.exp(lv_tr_np)).mean(0)
            active_dims = np.where(kl_per_dim > 0.01)[0]
            if len(active_dims) == 0:
                active_dims = np.arange(mu_tr_np.shape[1])
            Ztr_act = mu_tr_np[:, active_dims]
            Zts_act = mu_ts_np[:, active_dims]

            Utr = empirical_pit(Ztr_act).astype(np.float64)
            Uts = empirical_pit(Zts_act).astype(np.float64)
            vine = fit_vine(Utr, cfg.trunc)
            ll, aic = vine_metrics(vine, Uts)
            bic     = vine_bic(vine, Uts)
            ts0 = time.time()
            try:
                vine.simulate(500, seeds=[seed])
                result["t_sample"] = time.time() - ts0
            except Exception:
                pass

            tau_rec  = kendall_matrix(Xhat_np)
            corr_rec = np.corrcoef(Xhat_np.T)
            result.update(
                ll=ll, nll=-ll, aic=aic, bic=bic,
                n_params=vine_nparams(vine),
                dep_rec=np.linalg.norm(tau_orig - tau_rec, "fro"),
                tau_err=np.abs(tau_orig - tau_rec)[
                    np.triu_indices(d, 1)].mean(),
                rmse_corr=np.sqrt(np.mean((corr_orig - corr_rec) ** 2)))
            del model

        # ---------------------------------------------------------------------
        # WAE-Vine
        # ---------------------------------------------------------------------
        elif name == "WAE-Vine":
            model = train_wae(X_tr, X_vl, d_lat, cfg, seed)
            result["t_train"] = time.time() - t0
            model.eval()
            Xtr_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
            Xts_t = torch.tensor(X_ts, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                Ztr_t, _      = model(Xtr_t)
                Zts_t, Xhat_t = model(Xts_t)
                Ztr_np  = Ztr_t.float().cpu().numpy()
                Zts_np  = Zts_t.float().cpu().numpy()
                Xhat_np = Xhat_t.float().cpu().numpy()

            Utr = empirical_pit(Ztr_np).astype(np.float64)
            Uts = empirical_pit(Zts_np).astype(np.float64)
            vine = fit_vine(Utr, cfg.trunc)
            ll, aic = vine_metrics(vine, Uts)
            bic     = vine_bic(vine, Uts)
            ts0 = time.time()
            try:
                vine.simulate(500, seeds=[seed])
                result["t_sample"] = time.time() - ts0
            except Exception:
                pass

            tau_rec  = kendall_matrix(Xhat_np)
            corr_rec = np.corrcoef(Xhat_np.T)
            result.update(
                ll=ll, nll=-ll, aic=aic, bic=bic,
                n_params=vine_nparams(vine),
                dep_rec=np.linalg.norm(tau_orig - tau_rec, "fro"),
                tau_err=np.abs(tau_orig - tau_rec)[
                    np.triu_indices(d, 1)].mean(),
                rmse_corr=np.sqrt(np.mean((corr_orig - corr_rec) ** 2)))
            del model

        # ---------------------------------------------------------------------
        # InfoVAE-Vine
        # ---------------------------------------------------------------------
        elif name == "InfoVAE-Vine":
            model = train_infovae(X_tr, X_vl, d_lat, cfg, seed)
            result["t_train"] = time.time() - t0
            model.eval()
            Xtr_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
            Xts_t = torch.tensor(X_ts, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                _, _, mu_tr, _ = model(Xtr_t)
                _, _, mu_ts, _ = model(Xts_t)
                Xhat_t   = model.decoder(mu_ts)
                mu_tr_np = mu_tr.float().cpu().numpy()
                mu_ts_np = mu_ts.float().cpu().numpy()
                Xhat_np  = Xhat_t.float().cpu().numpy()

            Utr = empirical_pit(mu_tr_np).astype(np.float64)
            Uts = empirical_pit(mu_ts_np).astype(np.float64)
            vine = fit_vine(Utr, cfg.trunc)
            ll, aic = vine_metrics(vine, Uts)
            bic     = vine_bic(vine, Uts)
            ts0 = time.time()
            try:
                vine.simulate(500, seeds=[seed])
                result["t_sample"] = time.time() - ts0
            except Exception:
                pass

            tau_rec  = kendall_matrix(Xhat_np)
            corr_rec = np.corrcoef(Xhat_np.T)
            result.update(
                ll=ll, nll=-ll, aic=aic, bic=bic,
                n_params=vine_nparams(vine),
                dep_rec=np.linalg.norm(tau_orig - tau_rec, "fro"),
                tau_err=np.abs(tau_orig - tau_rec)[
                    np.triu_indices(d, 1)].mean(),
                rmse_corr=np.sqrt(np.mean((corr_orig - corr_rec) ** 2)))
            del model

        # ---------------------------------------------------------------------
        # Vine-Direct
        # ---------------------------------------------------------------------
        elif name == "Vine-Direct":
            vine = fit_vine(U_tr_emp, cfg.trunc)
            result["t_train"] = time.time() - t0
            ll, aic = vine_metrics(vine, U_ts_emp)
            bic     = vine_bic(vine, U_ts_emp)
            ts0 = time.time()
            try:
                vine.simulate(500, seeds=[seed])
                result["t_sample"] = time.time() - ts0
            except Exception:
                pass
            result.update(ll=ll, nll=-ll, aic=aic, bic=bic,
                          n_params=vine_nparams(vine))

        # ---------------------------------------------------------------------
        # Vine-Truncated
        # ---------------------------------------------------------------------
        elif name == "Vine-Truncated":
            vine = fit_vine(U_tr_emp, cfg.trunc_short)
            result["t_train"] = time.time() - t0
            ll, aic = vine_metrics(vine, U_ts_emp)
            bic     = vine_bic(vine, U_ts_emp)
            ts0 = time.time()
            try:
                vine.simulate(500, seeds=[seed])
                result["t_sample"] = time.time() - ts0
            except Exception:
                pass
            result.update(ll=ll, nll=-ll, aic=aic, bic=bic,
                          n_params=vine_nparams(vine))

    except Exception as e:
        print(f"  [{name}] FAILED: {e}")
        import traceback
        traceback.print_exc()

    result["mem_mb"] = _mem_mb() - mem0
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


# =============================================================================
# run_all: run all (method x scenario x seed)
# =============================================================================
def run_all(cfg=CFG, methods=None, scenarios=None, seeds=None,
            auto_d_lat=True):
    """Run all (method x scenario x seed) combinations."""
    methods   = methods   or METHODS
    scenarios = scenarios or SCENARIOS
    seeds     = seeds     or cfg.seeds
    rows = []

    for sc_key, sc in scenarios.items():
        print(f"\n{'='*65}")
        print(f"  {sc['label']}")
        print(f"{'='*65}")

        for seed in seeds:
            print(f"\n  -- seed={seed} --")
            ds = sc["split_fn"](sc["builder"](seed))

            if auto_d_lat:
                d_lat_used = select_d_lat(ds["X_train"], cfg.d_lat_var_target)
            else:
                d_lat_used = sc["d_lat"]
            print(f"    d_lat = {d_lat_used}")

            for meth in methods:
                print(f"\n  [{meth}]")
                r = run_method(meth, ds["X_train"], ds["X_val"],
                               ds["X_test"], d_lat_used, cfg, seed)
                r["scenario"]          = sc_key
                r["scenario_label"]    = sc["label"]
                r["n_train"]           = len(ds["X_train"])
                r["d"]                 = ds["X_train"].shape[1]
                r["d_lat_manual_ref"]  = sc["d_lat"]
                rows.append(r)
                print(f"    LL={r['ll']:+.4f}  AIC={r['aic']:.2f}  "
                      f"DepRec={r.get('dep_rec', np.nan):.4f}  "
                      f"t_train={r['t_train']:.1f}s")

    return pd.DataFrame(rows)


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  LS-Vine Benchmark")
    print("=" * 65)
    print(f"  Methods   : {len(METHODS)}")
    print(f"  Scenarios : {len(SCENARIOS)}")
    print(f"  Seeds     : {len(CFG.seeds)}")
    print("=" * 65)

    df = run_all(CFG)
    df.to_csv("results/results.csv", index=False)
    print("\nSaved: results/results.csv")
