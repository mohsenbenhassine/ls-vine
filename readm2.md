# LS-Vine: Learning Vine-Compatible Latent Representations for Multivariate Dependence Modeling

**Mohsen Ben Hassine** (corresponding author) — <mohsenmbh851@gmail.com>  
**Lamine Mili** — <lamine.mili@vt.edu>

**GitHub repository:** https://github.com/mohsenbenhassine/ls-vine

---

## Table of Contents

1. [Title](#title)
2. [Description](#description)
3. [Dataset Information](#dataset-information)
4. [Code Information](#code-information)
5. [Usage Instructions](#usage-instructions)
6. [Requirements](#requirements)
7. [Methodology](#methodology)
8. [Citations](#citations)
9. [License & Contribution Guidelines](#license--contribution-guidelines)

---

## 1. Title

**LS-Vine: Learning Vine-Compatible Latent Representations for Multivariate Dependence Modeling**

LS-Vine is a latent representation learning framework designed to reorganize multivariate data into a geometry that is compatible with vine copula modeling.

---

## 2. Description

LS-Vine is a latent representation learning framework that reorganizes multivariate data into a **vine-compatible geometry**. Unlike conventional autoencoders, which primarily optimize reconstruction fidelity, LS-Vine explicitly guides the latent space toward a structure that standard vine algorithms can decompose efficiently and accurately.

The repository contains:

- The complete LS-Vine implementation, including the encoder-decoder architecture and differentiable dependence-aware training objective.
- Benchmark implementations for **11 reference methods**:
  - PCA-Vine
  - ICA-Vine
  - FA-Vine
  - KPCA-Vine
  - AE-Vine
  - AE-Vine-Selected
  - VAE-Vine
  - WAE-Vine
  - InfoVAE-Vine
  - Vine-Direct
  - Vine-Truncated
- The experimental code required to reproduce the **9 experimental scenarios**: S1–S7, R1, and R2.
- Ablation experiments A1–A5.
- Latent-dimension sensitivity experiments.

### Main scientific contributions

1. **Vine-compatible latent geometry:** a framework combining encoder-decoder representation learning with a differentiable dependence-aware objective.
2. **Differentiable soft Kendall's tau reconstruction loss:** a dependence-sensitive reconstruction component using pairwise subsampling.
3. **Rank-dependence distribution matching loss \(L_v\):** preserves the distribution of pairwise dependence strengths in the latent space.
4. **Comprehensive empirical evaluation:** experiments covering 9 scenarios and multiple dimensionality-reduction and vine-modeling baselines.

> **Research scope.** The datasets used in this repository are simulated or empirically calibrated synthetic datasets. No real human or animal data are used.

---

## 3. Dataset Information

### 3.1 Overview

The experiments use simulated and empirically calibrated synthetic datasets.

| Scenario | Description | \(d\) | \(n\) | Source |
|---|---|---:|---:|---|
| S1 | Student-t D-vine | 10 | 3500 | Simulated (`pyvinecopulib`) |
| S2 | Student-t D-vine | 20 | 3500 | Simulated (`pyvinecopulib`) |
| S3 | Mixed R-vine (heterogeneous families) | 12 | 3500 | Simulated |
| R1 | S&P500-calibrated financial returns | 20 | 1500 | Empirically calibrated |
| R2 | ERA5-calibrated meteorological data | 15 | 3000 | Empirically calibrated |

### 3.2 Boundary-condition scenarios

| Scenario | Description | \(d\) | \(n\) |
|---|---|---:|---:|
| S4 | Gaussian Factor Model | 15 | 1500 |
| S5 | Noisy Low-Dim (30% noise) | 10 | 1500 |
| S6 | Pure Gaussian Copula | 15 | 1500 |
| S7 | Block Factor Student-t (no localized shocks) | 20 | 2000 |

### 3.3 Data format

Datasets are stored as CSV files with:

- one row per observation;
- one column per variable;
- columns named `x1, x2, ..., xd`.

The datasets can be reproduced using:

```bash
python reproduce_datasets.py
```

The generation procedures are designed to reproduce the experimental scenarios using the specified random seeds and simulation/calibration procedures.

---

## 4. Code Information

### 4.1 Repository structure

```text
ls-vine/
├── README.md
├── LICENSE
├── requirements.txt
├── reproduce_datasets.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── losses.py
│   ├── vine_utils.py
│   ├── train.py
│   ├── datasets.py
│   ├── benchmark.py
│   ├── stats.py
│   └── figures.py
├── notebooks/
│   ├── benchmark.ipynb
│   ├── ablation.ipynb
│   └── sensitivity.ipynb
├── data/
└── results/
    ├── results.csv
    ├── ablation.csv
    ├── sensitivity_S1.csv
    ├── sensitivity_S2.csv
    ├── figures/
    └── tables/
```

### 4.2 Main components

| Component | Purpose |
|---|---|
| `src/config.py` | Experiment configuration and hyperparameters |
| `src/models.py` | Encoder, decoder, LS-Vine and related neural models |
| `src/losses.py` | Reconstruction, dependence and regularization losses |
| `src/vine_utils.py` | Vine-copula fitting, evaluation and utility functions |
| `src/train.py` | Model training and validation procedures |
| `src/datasets.py` | Dataset loading, preprocessing and splitting |
| `src/benchmark.py` | Main benchmark experiments |
| `src/stats.py` | Statistical tests and comparative analysis |
| `src/figures.py` | Publication-oriented plots and figures |
| `reproduce_datasets.py` | Reproduction of experimental datasets |

The repository also includes Jupyter notebooks for the benchmark, ablation, and sensitivity analyses.

---

## 5. Usage Instructions

### 5.1 Clone the repository

```bash
git clone https://github.com/mohsenbenhassine/ls-vine.git
cd ls-vine
```

### 5.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 5.3 Reproduce the datasets

```bash
python reproduce_datasets.py
```

### 5.4 Run the complete benchmark

```bash
python src/benchmark.py
```

The complete benchmark comprises:

- 9 experimental scenarios;
- 12 evaluated methods, including LS-Vine;
- 10 random seeds;
- **1080 experiments** in total.

Approximate runtime:

```text
~4–6 hours on an NVIDIA T4 GPU
```

Runtime depends on hardware, CUDA configuration, software versions, and implementation details.

### 5.5 Run the ablation study

```bash
python src/ablation.py
```

The ablation study evaluates:

- S1 and S3;
- 6 model/loss variants;
- 10 random seeds.

Approximate runtime:

```text
~1–2 hours on an NVIDIA T4 GPU
```

### 5.6 Run the latent-dimension sensitivity study

```bash
python src/sensitivity.py
```

The sensitivity analysis evaluates:

- S1: \(k = 3, 5, 7, 9\)
- S2: \(k = 5, 8, 12, 16\)

Approximate runtime:

```text
~1 hour on an NVIDIA T4 GPU
```

### 5.7 Minimal Python example

The exact import paths may depend on the installed repository version. A minimal workflow is conceptually:

```python
from src.datasets import make_student_dvine
from src.train import train_lsvine

# Generate a Student-t D-vine dataset
X = make_student_dvine(
    d=10,
    n=3500,
    seed=42
)

# Train LS-Vine
model, history = train_lsvine(
    X,
    latent_dim=4,
    seed=42
)

print("Training completed.")
print(history)
```

For a fully reproducible experiment, use the configuration and benchmark entry points supplied by the repository rather than changing individual parameters interactively.

---

## 6. Requirements

### 6.1 Software

The reference environment uses:

| Package | Version |
|---|---|
| Python | >= 3.10 |
| PyTorch | 2.11.0+cu128 |
| pyvinecopulib | 1.0.0 |
| NumPy | 2.1.3 |
| SciPy | 1.16.3 |
| scikit-learn | 1.6.1 |
| pandas | 2.2.3 |
| matplotlib | 3.10.0 |
| seaborn | 0.13.2 |
| openpyxl | 3.1.5 |

Install the pinned project dependencies with:

```bash
pip install -r requirements.txt
```

### 6.2 Hardware

**Recommended:**

- NVIDIA T4 GPU
- CUDA-compatible PyTorch installation
- Sufficient disk space for datasets and experiment outputs

**Minimum:**

- 8 GB RAM
- CPU-only execution is possible, but the complete benchmark will take substantially longer.

The reported runtime estimates are hardware-dependent and should be regarded as approximate.

---

## 7. Methodology

### 7.1 Preprocessing

#### Rank transformation

Each marginal variable is transformed to the empirical probability scale using the empirical probability integral transform (PIT):

\[
u_{ij} = \frac{\operatorname{rank}(x_{ij})}{n+1}.
\]

This maps the observed variables approximately to the unit interval while preserving their rank structure.

### 7.2 Train/validation/test splitting

The default splits are:

- **S1, S2, S3, S7:** 2000 / 500 / 1000 observations for train / validation / test.
- **R1, R2, S4, S5, S6:** 60% / 20% / 20% for train / validation / test.

### 7.3 Latent dimension selection

The latent dimension \(k\) is selected using PCA with a target threshold of **90% explained variance**, followed by the sensitivity analysis over explicitly specified candidate values.

### 7.4 LS-Vine training objective

LS-Vine optimizes the following objective:

\[
L =
L_{\mathrm{rec}}
+
\lambda(t)L_v
+
\gamma L_{\mathrm{reg}}.
\]

The reconstruction loss is:

\[
L_{\mathrm{rec}}
=
L_{\mathrm{MSE}}
+
\alpha L_{\mathrm{soft}},
\]

where \(L_{\mathrm{soft}}\) is a differentiable soft Kendall's tau loss.

The dependence-distribution matching term is:

\[
L_v =
\text{Wasserstein}_{1D}
\left(
|\tau_X|,
|\tau_Z|
\right),
\]

where the empirical distributions of pairwise absolute Kendall's tau values are compared between the observed space and the latent space.

The regularization term is:

\[
L_{\mathrm{reg}}
=
\|\bar Z\|^2
+
\lambda_{\mathrm{var}}
\sum_j(\log s_j)^2.
\]

The dependence-loss coefficient uses exponential warmup:

\[
\lambda(t)
=
\lambda_{\max}
\left(
1-\exp\left(-\frac{t}{\tau_w}\right)
\right).
\]

### 7.5 Vine likelihood and early stopping

The actual vine log-likelihood is evaluated periodically on the validation set for early stopping.

Importantly:

> **The vine likelihood is not used to calculate the gradient.**

This separation prevents the non-differentiable vine fitting procedure from becoming part of the neural optimization objective.

### 7.6 Benchmark methods

The benchmark compares the following 12 methods:

1. LS-Vine
2. AE-Vine-Selected
3. PCA-Vine
4. ICA-Vine
5. FA-Vine
6. KPCA-Vine
7. AE-Vine
8. VAE-Vine
9. WAE-Vine
10. InfoVAE-Vine
11. Vine-Direct
12. Vine-Truncated

### 7.7 Evaluation metrics

The principal evaluation metrics are:

| Metric | Interpretation |
|---|---|
| LL | Log-likelihood; larger values indicate a higher likelihood under the fitted model |
| OS-PL | Out-of-sample predictive loss; smaller values are preferred |
| DepRec | Dependence-recovery error; smaller values indicate closer dependence recovery |
| Training time | Computational cost of model training |

### 7.8 Statistical analysis

The comparative analysis uses:

- **Friedman tests** by scenario;
- **one-sided Wilcoxon tests** for pairwise comparisons;
- **Holm-Bonferroni correction** for multiple comparisons.

Statistical conclusions should always be interpreted in the context of the corresponding scenario, sample size, random seeds, and multiple-testing correction.

---

## 8. Citations

If you use this repository or LS-Vine in academic work, please cite the corresponding paper.

### 8.1 Main LS-Vine paper

```bibtex
@article{benhassine2026lsvine,
  title   = {LS-Vine: Learning Vine-Compatible Latent Representations for Multivariate Dependence Modeling},
  author  = {Ben Hassine, Mohsen and Mili, Lamine},
  journal = {PeerJ Computer Science},
  year    = {2026}
}
```

### 8.2 Related work by the authors

- Ben Hassine, M., Mili, L., & Karra, K. (2016). arXiv:1612.07269.
- Ben Hassine, M., & Mili, L. (2025). *PeerJ Computer Science*, 11, e3228.
- Ben Hassine, M., Mili, L., & Karra, K. (2017). *International Journal of Advanced Computer Science and Applications (IJACSA)*, 8(7), 144–154.

### 8.3 Key methodological references

- Dissmann, J., E. C. Brechmann, C. Czado, and D. Kurowicka (2013). “Selecting and estimating regular vine copulae and application to financial returns.” *Computational Statistics & Data Analysis*, 59, 52–69.
- Nagler, T., & Czado, C. (2016). “Evading the curse of dimensionality in nonparametric density estimation with simplified vine copulas.” *Journal of Multivariate Analysis*, 151, 69–89.
- Bedford, T., & Cooke, R. M. (2001). “Probability density decomposition for conditionally dependent random variables modeled by vines.” *Annals of Mathematics and Artificial Intelligence*, 32, 245–268.

### 8.4 AI-assisted development and documentation

ChatGPT (OpenAI; GPT-4 / GPT-5; https://chat.openai.com) was used for:

- linguistic editing and grammatical correction of the manuscript;
- review and suggestions for debugging Python code;
- drafting and structuring project documentation, including this README;
- assistance with formatting LaTeX tables and references.

All scientific content, methodological decisions, experimental design, data generation, interpretation of results, and final conclusions remain the sole responsibility of the authors.

---

## 9. License & Contribution Guidelines

### 9.1 License

This project is released under the **MIT License**.

```text
MIT License

Copyright (c) 2026 Mohsen Ben Hassine, Lamine Mili
```

See the `LICENSE` file for the complete license text.

### 9.2 Contribution guidelines

Contributions are welcome when they improve reproducibility, correctness, documentation, or scientific usability of the project.

Before submitting a contribution:

1. Clearly describe the proposed change.
2. Preserve the reproducibility of existing experiments whenever possible.
3. Report software and hardware versions for computational changes.
4. Include tests or reproducibility checks for modifications affecting numerical results.
5. Do not silently change datasets, random seeds, evaluation metrics, or benchmark definitions.
6. Clearly distinguish methodological changes from implementation or documentation fixes.

For substantial scientific changes, please describe:

- the motivation;
- the affected experimental scenarios;
- the expected effect on results;
- the relevant statistical evaluation;
- any additional computational requirements.

Issues and pull requests can be submitted through the GitHub repository:

https://github.com/mohsenbenhassine/ls-vine

---

## Contact

**Mohsen Ben Hassine**  
Corresponding author  
Email: <mohsenmbh851@gmail.com>  
Affiliation: Faculté des Sciences de Tunis, Université de Tunis El Manar, Tunisia

**Lamine Mili**  
Email: <lamine.mili@vt.edu>

**Repository:** https://github.com/mohsenbenhassine/ls-vine

---

## Reproducibility Note

The experiments are intended to be reproducible from the repository using the provided dataset-generation, benchmark, ablation, and sensitivity-analysis scripts.

For publication-level reproduction, record at minimum:
https://github.com/mohsenbenhassine/ls-vine/blob/main/readm2.md
- Python version;
- package versions;
- CUDA/PyTorch configuration;
- GPU/CPU hardware;
- random seeds;
- dataset-generation configuration;
- experiment configuration;
- output files and commit/version of the repository.

Because neural-network training and vine fitting can be computationally sensitive to software and hardware environments, small numerical differences may occur across platforms even when the experimental protocol is unchanged.
