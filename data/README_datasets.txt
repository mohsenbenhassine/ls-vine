================================================================================
LS-Vine Benchmark — Datasets Directory
================================================================================

This directory contains the 9 datasets used in the LS-Vine benchmark:

    S1_student_dvine_d10.csv         S1 — Student-t D-vine (d=10, nu=4, n=3500)
    S2_student_dvine_d20.csv         S2 — Student-t D-vine (d=20, nu=4, n=3500)
    S3_mixed_rvine_d12.csv           S3 — Mixed R-vine (d=12, n=3500)
    R1_sp500_calibrated_d20.csv      R1 — S&P500-calibrated (d=20, n=1500)
    R2_era5_calibrated_d15.csv       R2 — ERA5-calibrated (d=15, n=3000)
    S4_gaussian_factor_d15.csv       S4 — Gaussian Factor Model (d=15, n=1500)
    S5_noisy_lowdim_d10.csv          S5 — Noisy Low-Dim (d=10, 30% noise, n=1500)
    S6_pure_gaussian_d15.csv         S6 — Pure Gaussian Copula (d=15, n=1500)
    S7_block_factor_d20.csv          S7 — Block Factor Student-t (d=20, n=2000)

================================================================================
DATA FORMAT
================================================================================

Each CSV file contains:
    - One row per sample (no index column)
    - One column per variable, named x1, x2, ..., xd
    - First line = header row with column names

The values are continuous, representing the raw synthetic observations BEFORE
any rank transformation. The rank transformation (Empirical PIT) is applied
internally by the benchmark scripts.

================================================================================
GENERATION
================================================================================

All datasets are generated reproducibly from fixed random seeds using the
script `reproduce_datasets.py` at the root of this repository:

    python reproduce_datasets.py

Seeds:
    S1, S2, S3, S4, S5, S6, S7:  seed = 42
    R1:                           seed = 42
    R2:                           seed = 43

================================================================================
CITATION
================================================================================

If you use these datasets in your research, please cite:

    Ben Hassine, M. & Mili, L. (2026). LS-Vine: Learning Vine-Compatible Latent
    Representations for Multivariate Dependence Modeling.
    PeerJ Computer Science.

================================================================================
LICENSE
================================================================================

These datasets are released under the MIT License. See the LICENSE file at
the root of this repository.
================================================================================
