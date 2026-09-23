"""
src/stats.py

Statistical tests for LS-Vine benchmark results.

Contains:
    - run_stats_tests : Friedman + Wilcoxon + ranking

Usage
-----
from src.stats import run_stats_tests
stats = run_stats_tests(df_results)
"""

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from .benchmark import METHODS


def run_stats_tests(df: pd.DataFrame, methods=None) -> dict:
    """
    Compute:
        1. Per-scenario one-sided Wilcoxon signed-rank tests:
           H1: median(OS-PL_LS) < median(OS-PL_baseline)
        2. Per-scenario Friedman test for global differences in rankings
        3. Mean ranking (lower = better) for AIC across all scenarios

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with columns: scenario, seed, method, aic.
    methods : list, optional
        List of method names to consider. Defaults to the METHODS from
        src.benchmark.

    Returns
    -------
    dict with keys:
        - "wilcoxon" : pd.DataFrame
        - "friedman" : pd.DataFrame
        - "ranking"  : pd.DataFrame
    """
    methods = methods or METHODS
    results = {}

    # -------------------------------------------------------------------------
    # 1. Wilcoxon: LS-Vine vs each baseline, per scenario
    # -------------------------------------------------------------------------
    wilcoxon_rows = []
    for sc in df["scenario"].unique():
        sub = df[df["scenario"] == sc]
        ls_aic = sub[sub["method"] == "LS-Vine"]["aic"].dropna().values
        for meth in methods:
            if meth == "LS-Vine":
                continue
            other_aic = sub[sub["method"] == meth]["aic"].dropna().values
            n = min(len(ls_aic), len(other_aic))
            if n < 2:
                continue
            try:
                stat, p = wilcoxon(ls_aic[:n], other_aic[:n])
                wilcoxon_rows.append(dict(
                    scenario=sc, vs=meth, n=n,
                    stat=round(stat, 3),
                    p=round(p, 4),
                    sig="*" if p < 0.05 else ""
                ))
            except Exception:
                pass
    results["wilcoxon"] = pd.DataFrame(wilcoxon_rows)

    # -------------------------------------------------------------------------
    # 2. Friedman test, per scenario (seeds as blocks)
    # -------------------------------------------------------------------------
    friedman_rows = []
    for sc in df["scenario"].unique():
        sub = df[df["scenario"] == sc]
        pivot = (sub.pivot(index="seed", columns="method", values="aic")
                    .dropna())
        if len(pivot) >= 3 and pivot.shape[1] >= 3:
            try:
                groups = [pivot[m].values for m in pivot.columns]
                stat_f, p_f = friedmanchisquare(*groups)
                friedman_rows.append(dict(
                    scenario=sc,
                    chi2=round(stat_f, 3),
                    p=round(p_f, 6),
                    n_blocks=len(pivot),
                    n_methods=pivot.shape[1],
                    max_possible=len(pivot) * (pivot.shape[1] - 1),
                    valid=stat_f <= len(pivot) * (pivot.shape[1] - 1) + 0.01,
                ))
            except Exception as e:
                friedman_rows.append(dict(
                    scenario=sc, chi2=np.nan, p=np.nan, error=str(e)))
    results["friedman"] = pd.DataFrame(friedman_rows)

    # -------------------------------------------------------------------------
    # 3. Mean ranking across scenarios (lower AIC = better rank)
    # -------------------------------------------------------------------------
    rank_df = (df.groupby(["scenario", "seed", "method"])["aic"]
                 .mean().reset_index())
    rank_df["rank"] = rank_df.groupby(["scenario", "seed"])["aic"].rank()
    avg_rank = (rank_df.groupby("method")["rank"]
                       .mean().sort_values()
                       .reset_index()
                       .rename(columns={"rank": "mean_rank"}))
    avg_rank["mean_rank"] = avg_rank["mean_rank"].round(3)
    results["ranking"] = avg_rank

    # -------------------------------------------------------------------------
    # Console report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  STATISTICAL TESTS")
    print("=" * 78)

    print("\n-- Friedman test (per scenario) --")
    if not results["friedman"].empty:
        for _, row in results["friedman"].iterrows():
            status = "OK" if row.get("valid", False) else "FAIL"
            print(f"  {row['scenario']}: chi2={row['chi2']}, p={row['p']}, "
                  f"max={row.get('max_possible', 'N/A')} [{status}]")

    print("\n-- Wilcoxon: LS-Vine vs baselines --")
    if not results["wilcoxon"].empty:
        print(results["wilcoxon"].to_string(index=False))

    print("\n-- Mean ranking (AIC, lower = better) --")
    print(results["ranking"].to_string(index=False))

    print("=" * 78)
    return results


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    results_file = Path("results/results.csv")
    if not results_file.exists():
        print(f"ERROR: {results_file} not found.")
        print("Run `python src/benchmark.py` first to generate the results.")
        sys.exit(1)

    df = pd.read_csv(results_file)
    stats = run_stats_tests(df)

    # Save outputs
    Path("results").mkdir(exist_ok=True)
    stats["wilcoxon"].to_csv("results/wilcoxon.csv", index=False)
    stats["friedman"].to_csv("results/friedman.csv", index=False)
    stats["ranking"].to_csv("results/ranking.csv", index=False)
    print("\nSaved: results/wilcoxon.csv, results/friedman.csv, "
          "results/ranking.csv")
