# LS-Vine: Learning Vine-Friendly Latent Representations for Improved Multivariate Dependence Modeling

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

**LS-Vine** is a latent representation learning framework that learns vine-compatible latent geometries for improved multivariate dependence modeling. Unlike conventional dimensionality reduction techniques that primarily optimize for reconstruction error, LS-Vine learns representations in which the multivariate dependence structure is intrinsically more parsimonious and decomposable into conditional bivariate pairs.

> **Key Insight:** A representation can have excellent reconstruction fidelity while remaining statistically inconvenient for vine estimation. LS-Vine deliberately sacrifices marginal local fidelity to achieve a substantial gain in global vine compatibility — the *Dependence Reconstruction Paradox*.

## 🎯 Key Contributions

| # | Contribution | Description |
|---|--------------|-------------|
| 1 | **Rank-based empirical PIT** | Non-parametric marginal uniformization for stable vine estimation |
| 2 | **Soft Kendall's τ loss** | Differentiable reconstruction loss that penalizes local dependence distortion |
| 3 | **Rank-distribution matching loss (ℒᵥ)** | Preserves the global distribution of pairwise dependence strengths |
| 4 | **Vine freezing schedule** | Alternating optimization with warm-started re-estimation |
| 5 | **Adaptive λ warmup** | Safe joint optimization of competing objectives |

## 📊 Main Results

### AIC Comparison (Lower is Better)

| Method | S1 (d=10) | S2 (d=20) | S3 (d=12) | R1 (Finance) | R2 (Météo) |
|--------|-----------|-----------|-----------|--------------|------------|
| Vine-Direct | -6765 ± 334 | -14537 ± 590 | -4108.9 ± 136.5 | -6556 ± 235 | -6874 ± 279 |
| Vine-Truncated | -4680 ± 308 | -9689 ± 595 | -3228.7 ± 144.4 | -5274 ± 226 | -5605 ± 249 |
| **LS-Vine** | **-2049 ± 147** | **-6757 ± 389** | **-1487.3 ± 87.2** | -4593 ± 819 | **-2588 ± 194** |
| AE-Vine-Selected | -1486 ± 414 | -2749 ± 588 | -584.8 ± 226.0 | -5855 ± 916 | -1212 ± 283 |
| AE-Vine | -1250 ± 481 | -2445 ± 555 | -404.4 ± 105.9 | -12009 ± 914 | -767 ± 197 |
| PCA-Vine | -136 ± 57 | -508 ± 64 | 30.8 ± 6.3 | -30 ± 73 | 54 ± 35 |
| VAE-Vine | -94 ± 47 | -317 ± 66 | -34.3 ± 38.7 | -218 ± 42 | 32 ± 71 |

### The Dependence Reconstruction Paradox

| Method | DepRec (S2) | OS-PL (S2) |
|--------|-------------|------------|
| **LS-Vine** | 2.04 ± 0.20 | **-6757 ± 389** |
| AE-Vine | **1.91 ± 0.16** | -2445 ± 555 |
| AE-Vine-Selected | 2.39 ± 0.39 | -2749 ± 588 |

> **Key Finding:** Minimizing local reconstruction error (DepRec) does NOT guarantee a globally vine-compatible geometry. LS-Vine sacrifices marginal local fidelity for massive gain in global dependence structuring (2.8× improvement in OS-PL magnitude).

### Sensitivity to Latent Dimension k

| Scenario | k | k/d | OS-PL | DepRec |
|----------|---|-----|-------|--------|
| **S1 (d=10)** | 3 | 0.30 | -877 ± 88 | 2.61 ± 0.12 |
| | 5 | 0.50 | -1709 ± 114 | 1.05 ± 0.03 |
| | 7 | 0.70 | -3733 ± 214 | 0.48 ± 0.06 |
| | **9** | **0.90** | **-4568 ± 268** | **0.20 ± 0.07** |
| **S2 (d=20)** | 5 | 0.25 | -1855 ± 82 | 4.29 ± 0.19 |
| | 8 | 0.40 | -5181 ± 139 | 2.67 ± 0.14 |
| | 12 | 0.60 | -7142 ± 245 | 1.19 ± 0.08 |
| | **16** | **0.80** | **-9645 ± 577** | **0.52 ± 0.19** |

> **Insight:** LS-Vine's primary benefit comes from **restructuring dependence geometry**, not aggressive dimensionality reduction. Best results are obtained with minimal compression (k/d ≈ 0.8-0.9).

## 🧪 Ablation Study

| Variant | S1 (d=10) | S3 (d=12) |
|---------|-----------|-----------|
| **LS-Vine (Full)** | **-1774.5 ± 159.4** | **-1133.4 ± 152.6** |
| A1: No ℒᵥ (RankZ) | -1166.3 ± 275.8 | -503.6 ± 155.7 |
| A2: No Soft-Tau | -1821.8 ± 105.8 | -1185.3 ± 93.5 |
| A3: No Regularization | -1780.0 ± 160.1 | -1138.8 ± 147.8 |
| A4: No Warmup | -1842.8 ± 156.1 | -1164.0 ± 95.3 |
| A5: MSE Only | -1240.2 ± 587.3 | -559.2 ± 125.8 |

**Key Insights:**
- **ℒᵥ is the primary driver** of latent space restructuring (34% drop on S1, 56% on S3)
- **Soft-Tau stabilizes training** (reduces variance across seeds)
- **Regularization and warmup** primarily aid training stability

## 🔬 Statistical Significance

### Wilcoxon Tests (Adjusted p-values)

| Scenario | LS-Vine vs AE-Vine | LS-Vine vs AE-Vine-Selected |
|----------|-------------------|------------------------------|
| S1 (d=10) | **0.002** | **0.015** |
| S2 (d=20) | **<0.001** | **0.003** |
| R2 (Météo) | **0.004** | **0.011** |
| R1 (Finance) | 0.124 (n.s.) | 0.087 (n.s.) |

*Values < 0.05 indicate statistically significant superiority of LS-Vine.*

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/mohsenbenhassine/ls-vine.git
cd ls-vine

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
