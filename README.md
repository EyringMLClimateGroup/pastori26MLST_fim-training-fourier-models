# Fourier Models: Training and Fisher Information

Code for the numerical experiments of:

> L. Pastori, V. Eyring, M. Schwabe, *"Fisher Information, Training and Bias in Fourier Regression Models"*, [arXiv:2510.06945](https://arxiv.org/abs/2510.06945).
**Corresponding Author**: Lorenzo Pastori ([l.pastori.tn@gmail.com](l.pastori.tn@gmail.com))

## Description

Quantum neural networks (QNNs) with angle-encoded inputs, and more generally any model built from
a fixed harmonic basis, realize a **partial Fourier series**

```
f_theta(x) = sum_mu c_mu(theta) e_mu(x),      c_mu(theta) = sum_nu Gamma_{mu,nu} iota_nu(theta)
```

where `e_mu(x)` are input-space harmonic basis functions, `iota_nu(theta)` are parameter-space
harmonic basis functions, and the **structure-constants matrix Gamma** encodes how the trainable
parameters couple to the accessible input frequencies. Writing the singular value decomposition
`Gamma = U @ diag(S) @ V^T` isolates the **correlation spectrum** `S`, whose decay/purity
`tr(S^4)` governs the model's expressivity.

This repository implements Fourier models directly from `Gamma` (dense reference
implementation), from a QNN circuit layout (mapped to the subset of `Gamma` it can express via
qubit backward light-cones), and from **tensor-network** (matrix product state / tensor train,
and tree tensor network) representations of `U`/`V^T` that avoid the exponential storage cost of
the dense bases as the number of input features `N` and parameters `M` grow. For all of these it
computes the **Fisher Information Matrix (FIM)** `F_jk(theta) = E_x[df/dtheta_j * df/dtheta_k]`,
either by automatic differentiation or in closed form from `Gamma`'s spectrum, and the resulting
**effective dimension (ED)**, a FIM-spectrum-based capacity measure. The code is used to study how
ED and model **bias** (constructed by orthogonalizing `V^T` against a reference parameter point)
jointly determine training performance when models are fit to a target function by minimizing MSE.

## Dependencies

- [JAX](https://github.com/google/jax) (`jax`, `jax.numpy`) — autodiff, JIT compilation and
  `vmap`/`fori_loop`-based tensor-network contractions
- [Optax](https://github.com/google-deepmind/optax) — gradient-based training loop
- [PennyLane](https://pennylane.ai/) (`pennylane.numpy`) — array backend used in the experiment
  scripts (drop-in NumPy with autodiff support)
- NumPy / SciPy — dense linear algebra, SVD, random orthogonal matrix generation
- scikit-learn (`sklearn.utils.shuffle`) — data shuffling during training and construction of
  block-sparse orthogonal matrices
- Jupyter notebooks (`.ipynb`) — used throughout to load the pickled results and produce the
  paper's plots

No `requirements.txt`/environment file is provided; the experiment scripts also hard-code an HPC
working-directory path (`path_base`, `results_folder`) that must be edited to a local path before
running.

## Repository overview

- **`useful_functions/`** — the core library shared by all experiments:
  - `model_constructor_functions.py`, `tensorized_model_constructor_functions.py`,
    `fully_tensorized_model_constructor_functions.py`: build the Fourier model from a dense
    `Gamma`, a partially tensorized `Gamma` (tensor-train `V^T`, dense `U`), or a fully tensorized
    `Gamma` (tensor-train `V^T` and MPO+TTN `U`), respectively.
  - `tensor_network_functions_np.py` / `tensor_network_functions_jax.py`: tensor-train
    construction, orthogonalization, contraction, and FIM evaluation (NumPy reference and
    JAX/JIT implementations).
  - `FIM_functions_jax.py`: black-box, autodiff-based FIM estimator (used for QNNs and other
    models without an explicit `Gamma`).
  - `analytical_FIM_functions_np.py`: closed-form FIM/effective-dimension from `Gamma`'s SVD.
  - `ortho_matrices_functions.py`: random (Haar and block-sparse) orthogonal matrix generation
    for `U`/`V`.
  - `compute_basis_functions_masks_qnns.py`: maps a QNN circuit layout to the subset of the joint
    input x parameter Fourier basis it can express, via each qubit's backward light-cone.
  - `training_functions_jax.py`: JAX/Optax MSE training loop used in all training experiments.

- **`basis_functions_analysis_QNN/`** — scans QNN measurement-layer connectivities and relates
  the resulting basis-function coverage to correlation-spectrum purity and effective dimension.

- **`scaling_effdim_*/`** (dense) and **`scaling_effdim_TN_*/`** / **`scaling_effdim_bond_TN_Vh/`**
  (tensor-network) — sweep one control parameter at a time (local parameter-basis dimension
  `d_tilde`, number of parameters `M`, correlation-spectrum decay exponent, input-space dimension,
  `V^T`'s sparsity, or tensor-train bond dimension `chi`) and measure how the normalized effective
  dimension scales with it.

- **`scaling_training_with_MaxFreq/`**, **`scaling_training_with_Nparams_Dloc3/`** — train biased
  and unbiased models while sweeping the maximum input frequency or the number of parameters, to
  study how the ED/bias interplay affects the achievable MSE.

- **`train_and_FIM_Dloc3/`**, **`train_and_FIM_TN_Dloc3/`**, **`train_and_FIM_fullyTN_Dloc3/`**,
  **`train_and_FIM_2D_TN_Dloc3/`** — the main training + FIM experiments (dense, partially
  tensorized, fully tensorized, and 2-input-feature tensorized models): construct full/cutoff
  biased model pairs sharing a data-generating function, compute their normalized FIM spectra, and
  train both to compare `MSE_min` against effective dimension.

Each experiment directory pairs one or more `.py` scripts (which pickle results to disk) with a
`.ipynb` notebook that loads those results and reproduces the corresponding paper figures.


## List of figures

- Figs. 2(a) and 2(b) have been generated using the scripts in **`scaling_effdim_decay_spectrS/`** and **`scaling_effdim_decay_dimInSpace/`**, respectively.

- Figs. 4 and 5 have been generated using the scripts in **`train_and_FIM_Dloc3/`**.

- Fig. 6 has been generated using the scripts in **`train_and_FIM_fullyTN_Dloc3/`**.

- Figs. 7(a) and 7(b) have been generated using the scripts in **`scaling_training_with_Nparams_Dloc3/`**.

- Figs. 7(c) and 7(d) have been generated using the scripts in **`scaling_training_with_MaxFreq/`**.

- Figs. D1 and D2 have been generated using the scripts in **`train_and_FIM_TN_Dloc3/`**.

- Fig. 3 in the Supplementary Material has been generated using the scripts in **`basis_functions_analysis_QNN/`**.

- Fig. 4 in the Supplementary Material has been generated using the scripts in **`scaling_effdim_sparsity_Vh/`**.

- All remaining figures in the Supplementary Material have been generated by the same scripts with changed parameter selection as reported in the corresponding figure caption.
