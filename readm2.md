# LS-Vine: Learning Vine-Compatible Latent Representations for Multivariate Dependence Modeling

## Title

**LS-Vine: Learning Vine-Compatible Latent Representations for Multivariate Dependence Modeling**

**Authors:**
- Mohsen Ben Hassine (corresponding author) — [mohsenmbh851@gmail.com]
- Lamine Mili — [lamine.mili@vt.edu]

**Repository:** [https://github.com/mohsenbenhassine/ls-vine]

---

## Description

LS-Vine is a latent representation learning framework that reorganizes multivariate data into a vine-compatible geometry. Unlike conventional autoencoders, which optimize for reconstruction fidelity, LS-Vine explicitly guides the latent space toward a structure that standard vine algorithms can decompose efficiently and accurately.

This repository contains:

- The complete implementation of LS-Vine (encoder-decoder architecture + differentiable dependence-aware training objective)
- The benchmark code for 11 baseline methods:
  - Linear baselines: PCA-Vine, ICA-Vine, FA-Vine, KPCA-Vine
  - Neural baselines: AE-Vine, AE-Vine-Selected, VAE-Vine, WAE-Vine, InfoVAE-Vine
  - Reference models: Vine-Direct, Vine-Truncated
- The code to reproduce all 9 experimental scenarios (S1-S7, R1, R2)
- The ablation study (A1-A5) and sensitivity analysis (latent dimension k)

The main scientific contributions of this work are:
1. A vine-compatible latent geometry framework combining encoder-decoder representation learning with a differentiable dependence-aware objective.
2. A differentiable soft Kendall's tau reconstruction loss with pairwise subsampling.
3. A rank-distribution matching loss L_v that preserves the distribution of pairwise dependence strengths in the latent space.
4. A comprehensive empirical evaluation across 9 scenarios and 11 baselines.

---

## Dataset Information

This project uses simulated and empirically calibrated synthetic datasets. No real human or animal data is used. All datasets are generated reproducibly from fixed random seeds.

### Primary scenarios (complex non-linear dependencies)

| Scenario | Description | d | n | Source |
|----------|-------------|---|---|--------|
| S1 | Student-t D-vine | 10 | 3500 | Simulated (pyvinecopulib) |
| S2 | Student-t D-vine | 20 | 3500 | Simulated (pyvinecopulib) |
| S3 | Mixed R-vine (heterogeneous families) | 12 | 3500 | Simulated (pyvinecopulib) |
| R1 | S&P500-calibrated financial returns | 20 | 1500 | Empirically calibrated synthetic |
| R2 | ERA5-calibrated meteorological data | 15 | 3000 | Empirically calibrated synthetic |

### Boundary conditions

| Scenario | Description | d | n |
|----------|-------------|---|---|
| S4 | Gaussian Factor Model | 15 | 1500 |
| S5 | Noisy Low-Dim (30% noise) | 10 | 1500 |
| S6 | Pure Gaussian Copula | 15 | 1500 |
| S7 | Block Factor Student-t (no localized shocks) | 20 | 2000 |

### Data format

Each dataset is saved as a CSV file with the following structure:
- One row per sample
- One column per variable, named x1, x2, ..., xd

### Where to find the data

- Generated automatically by running reproduce_datasets.py (see Usage Instructions)
- Pre-generated copies are available in the data/ directory
- Code repository: https://github.com/mohsenbenhassine/ls-vine

---

## Code Information

The codebase is organized as a modular Python package:
ls-vine/
├── README.md # This file
├── LICENSE # MIT License
├── requirements.txt # Python dependencies
├── reproduce_datasets.py # Script to regenerate all datasets
├── src/
│ ├── init.py
│ ├── config.py # Global hyperparameters
│ ├── models.py # Neural network architectures
│ ├── losses.py # L_rec, L_v, L_reg, soft Kendall's tau
│ ├── vine_utils.py # Vine fitting and metrics (pyvinecopulib)
│ ├── train.py # Training loops for all methods
│ ├── datasets.py # Dataset generators (S1-S7, R1, R2)
│ ├── benchmark.py # Main experiment runner
│ ├── stats.py # Statistical tests (Friedman, Wilcoxon)
│ └── figures.py # Figure generation
├── notebooks/
│ ├── benchmark.ipynb # Full benchmark
│ ├── ablation.ipynb # Ablation study (A1-A5)
│ └── sensitivity.ipynb # Sensitivity to latent dimension k
├── data/ # Simulated datasets (CSV)
└── results/
├── results.csv # Raw benchmark results
├── ablation.csv # Ablation results
├── sensitivity_S1.csv # Sensitivity results for S1
├── sensitivity_S2.csv # Sensitivity results for S2
├── figures/ # All figures (PNG)
└── tables/ # LaTeX tables


### Key files explained

| File | Purpose |
|------|---------|
| src/losses.py | Implements L_rec (MSE + soft Kendall's tau), L_v (rank-distribution matching), L_reg (latent regularization) |
| src/train.py | Training loops for LS-Vine, AE-Vine, VAE-Vine, WAE-Vine, InfoVAE-Vine, and AE-Vine-Selected |
| src/vine_utils.py | Wrappers around pyvinecopulib for fitting R-vines and computing LL / AIC / BIC |
| src/benchmark.py | Main script: runs the full benchmark (9 scenarios x 12 methods x 10 seeds) |
| reproduce_datasets.py | Regenerates all 9 datasets from fixed seeds |

---

## Usage Instructions

### Step 1: Clone the repository
git clone [https://github.com/mohsenbenhassine/ls-vine].git
cd ls-vine

### Step 2: Install dependencies
pip install -r requirements.txt

### Step 3: Reproduce the datasets
python reproduce_datasets.py

This generates all 9 datasets in data/ directory as CSV files.

### Step 4: Run the full benchmark
python src/benchmark.py

This runs:
- 9 scenarios x 12 methods x 10 seeds = 1080 experiments
- Saves results to results/results.csv
- Generates all figures in results/figures/
- Exports LaTeX tables to results/tables/

Estimated runtime: ~4-6 hours on a T4 GPU.

### Step 5: Run the ablation study
python src/ablation.py

This runs the ablation study on scenarios S1 and S3 (6 variants x 10 seeds).
Estimated runtime: ~1-2 hours on a T4 GPU.

### Step 6: Run the sensitivity analysis
python src/sensitivity.py

This sweeps the latent dimension k on S1 (k=3,5,7,9) and S2 (k=5,8,12,16).
Estimated runtime: ~1 hour on a T4 GPU.

### Reproducing a single experiment (minimal example)

```python
import torch
import numpy as np
from src.datasets import make_student_dvine, split
from src.train import train_lsvine
from src.vine_utils import fit_vine, vine_metrics, empirical_pit

# 1. Generate S1 data
X = make_student_dvine(d=10, rho=0.4, nu=4, n=3500, seed=42)
ds = split(X)

# 2. Train LS-Vine
model, vine, hist, mu, std = train_lsvine(
    ds["X_train"], ds["X_val"], d_lat=5, seed=42)

# 3. Evaluate on test set
Xts_t = torch.tensor(ds["X_test"], dtype=torch.float32).cuda()
model.eval()
with torch.no_grad():
    Zts, Xhat = model(Xts_t)
    Uts = empirical_pit(Zts.float().cpu().numpy()).astype(np.float64)

ll, aic = vine_metrics(vine, Uts)
print(f"Test LL : {ll:.4f}")
print(f"Test AIC: {aic:.2f}")
Requirements
The project was developed and tested with the following versions:

Python >= 3.10

PyTorch == 2.11.0+cu128 (with CUDA support for GPU acceleration)

pyvinecopulib == 1.0.0

numpy == 2.1.3

scipy == 1.16.3

scikit-learn == 1.6.1

pandas == 2.2.3

matplotlib == 3.10.0

seaborn == 0.13.2

openpyxl == 3.1.5

Installing dependencies
Install all dependencies in one command:
pip install -r requirements.txt
pyvinecopulib==1.0.0
torch==2.11.0+cu128
numpy==2.1.3
scipy==1.16.3
scikit-learn==1.6.1
pandas==2.2.3
matplotlib==3.10.0
seaborn==0.13.2
openpyxl==3.1.5
Hardware requirements
Recommended: NVIDIA T4 GPU (or equivalent)

Minimum: 8 GB RAM, CPU-only (significantly slower)

Methodology
Methodology
Data preprocessing
All datasets are generated using the following protocol:

Rank transformation (Empirical PIT) — Each variable is mapped to the unit interval [0, 1] using the empirical cumulative distribution function:

u_ij = rank(x_ij) / (n + 1)

This guarantees that all values lie strictly in (0, 1), avoiding boundary issues in vine fitting.

Train/Validation/Test split:

Primary scenarios (S1, S2, S3, S7): 2000 train / 500 validation / 1000 test

Calibrated scenarios (R1, R2, S4, S5, S6): 60% train / 20% validation / 20% test

Latent dimension selection — Selected automatically via PCA explained variance threshold (90%) on the training set, computed separately for each (scenario, seed) pair.

LS-Vine training
The complete training objective is:
L = L_rec + lambda(t) * L_v + gamma * L_reg
where:

L_rec = MSE + alpha * L_soft — reconstruction loss combining MSE with a soft Kendall's tau term that preserves local pairwise concordance.

L_v — rank-distribution matching loss comparing the empirical quantile functions of |tau| between input X and latent Z (Wasserstein-1D).
L_reg = ||Z_bar||^2 + lambda_var * sum (log s_j)^2 — latent regularization to prevent exploding variances and degenerate solutions.

lambda(t) = lambda_max * (1 - exp(-t / tau_w)) — exponential warmup schedule for the vine loss weight.

The true vine likelihood L_vine is evaluated periodically on the validation set for early stopping and checkpoint selection. It is not used for gradient computation.
Benchmark methods
12 methods are benchmarked:

Method	Type	Description
LS-Vine	Proposed	Encoder-decoder + dependence-aware training
AE-Vine-Selected	Ablation	Standard AE, checkpoint selected by validation AIC
PCA-Vine	Linear	PCA + vine
ICA-Vine	Linear	FastICA + vine
FA-Vine	Linear	Factor Analysis + vine
KPCA-Vine	Non-linear	Kernel PCA (RBF) + vine
AE-Vine	Neural	Standard autoencoder + vine
VAE-Vine	Neural	Variational autoencoder + vine
WAE-Vine	Neural	Wasserstein autoencoder + vine
InfoVAE-Vine	Neural	InfoVAE + vine
Vine-Direct	Reference	Full-dimensional vine on raw pseudo-observations
Vine-Truncated  Reference	Vine truncated at depth 1
Evaluation metrics
Log-likelihood (LL) — higher is better

Out-of-Sample Penalized Likelihood (OS-PL) = -2 * l_test + 2p — lower is better

Dependence Reconstruction Error (DepRec) = ||tau(X) - tau(X_hat)||_F — lower is better

Training time (seconds, measured on a T4 GPU)

Statistical tests
Friedman test (per scenario) — for global differences in rankings

One-sided Wilcoxon signed-rank tests with Holm-Bonferroni correction — for pairwise comparisons (LS-Vine vs each baseline)

Citations
If you use this code or data, please cite
@article{benhassine2026lsvine,
  title   = {LS-Vine: Learning Vine-Compatible Latent Representations for Multivariate Dependence Modeling},
  author  = {Ben Hassine, Mohsen and Mili, Lamine},
  journal = {PeerJ Computer Science},
  year    = {2026},

}
Related work by the authors
Ben Hassine, M., Mili, L., & Karra, K. (2016). A Copula Statistic for Measuring Nonlinear Multivariate Dependence. arXiv preprint arXiv:1612.07269.

Ben Hassine, M., & Mili, L. (2025). Empirical copula-based data augmentation for mixed-type datasets: a robust approach for synthetic data generation. PeerJ Computer Science, 11, e3228. https://doi.org/10.7717/peerj-cs.3228
Ben Hassine, M., Mili, L., & Karra, K. (2017). A Copula Statistic for Measuring Nonlinear Dependence with Application to Feature Selection in Machine Learning. International Journal of Advanced Computer Science and Applications, 8(7), 144-154. https://doi.org/10.14569/IJACSA.2017.080720

Key methodological references
Dissmann, J., Brechmann, E. C., Czado, C., & Kurowicka, D. (2013). Selecting and estimating regular vine copulae and application to financial returns. Computational Statistics & Data Analysis, 59, 52-69.

Nagler, T., & Czado, C. (2016). Evading the curse of dimensionality in nonparametric density estimation with simplified vine copulas. Journal of Multivariate Analysis, 151, 69-89.

Bedford, T., & Cooke, R. M. (2001). Probability density decomposition for conditionally dependent random variables modeled by vines. Annals of Mathematics and Artificial Intelligence, 32, 245-268.
License & Contribution Guidelines
License
This project is licensed under the MIT License.
MIT License

Copyright (c) 2026 [Mohsen Ben Hassine, Lamine Mili]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
Contact
For scientific or technical questions, contact the corresponding author:

Mohsen Ben Hassine

Email: [mohsenmbh851@gmail.com]

Affiliation: Faculté des Sciences de Tunis, Université de Tunis El Manar, Tunisia

Acknowledgments
The authors thank the Editor and reviewers of PeerJ Computer Science for their constructive feedback. 
Use of Artificial Intelligence
ChatGPT (OpenAI, GPT-4 / GPT-5, https://chat.openai.com) was used in the preparation of this manuscript and code for the following purposes:

Language editing and grammar correction of the manuscript text

Review and debugging suggestions for the Python implementation

Drafting and structuring of documentation (including this README file)

Assistance in formatting LaTeX tables and reference lists

All scientific content, methodological choices, experimental design, data generation, result interpretation, and final conclusions are the sole responsibility of the authors. No scientific claims were generated by the AI tools. All AI-suggested text and code were reviewed, verified, tested, and where necessary corrected by the authors before inclusion.


Last updated: [Septembre 2026]

Version: 1.0.0





