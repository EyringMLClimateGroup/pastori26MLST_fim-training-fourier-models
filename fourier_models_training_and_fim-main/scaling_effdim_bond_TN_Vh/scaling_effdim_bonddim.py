"""Scaling of the (normalized) effective dimension (ED) with the tensor-train bond
dimension `bond_dim` used to represent V, the parameter-side singular-vector matrix
of the SVD Gamma = U @ diag(S) @ V^T of the structure constants (partially tensorized
model: V is a matrix product state / tensor train, while the correlation spectrum S
is still specified directly, as in the non-TN scripts).

For each bond dimension in `bond_dim_vec`, draws `no_matrix_realiz` random orthogonal
tensor-train realizations of V (fixed correlation spectrum S with decay exponent
`dec_exp`), evaluates the normalized FIM at `no_samples` random parameter draws per
realization, and computes: the correlation-spectrum purity tr(S^4) (constant across
realizations here, since S is fixed) and the normalized effective dimension (Eq. 12
of the paper) via the Monte-Carlo estimator `eff_dim_liminf`. Results (bond dimension,
tr(S^4), FIM purity, normalized ED) are pickled to `results_folder` for later plotting.

Other key control parameters set in the "Experiments specs" section (bond_dim itself
being the swept variable):
- no_samples: number of random parameter samples used to Monte-Carlo estimate the
  normalized effective dimension for each model draw.
- no_matrix_realiz: number of random tensor-train realizations of V drawn per bond
  dimension.
- dim_in: input-basis dimension D (dimension of the correlation spectrum S).
- dec_exp: decay exponent of the correlation spectrum S (fixed across the bond-dim
  scan).
- no_params: number of model parameters M (= K, the parameter-basis dimension).
- local_dim_param: local per-parameter trigonometric basis dimension d_tilde (Dloc).
- params_min / params_max: bounds of the uniform distribution used to sample
  parameter values theta.
"""

# Importing necessary packages
import sys
import os
import importlib
import pickle

import pennylane.numpy as np

import jax
from jax import numpy as jnp


import scipy



# Current path for importing custom functions
path_base = '/home/b/b309245/FIM_Training_Bias_RegressionModels/fourier_models_training_and_fim/'
sys.path.insert(0, path_base + 'useful_functions')

import model_constructor_functions
importlib.reload(model_constructor_functions)

import ortho_matrices_functions
importlib.reload(ortho_matrices_functions)

import tensor_network_functions_np
importlib.reload(tensor_network_functions_np)

import tensor_network_functions_jax
importlib.reload(tensor_network_functions_jax)
import tensor_network_functions_jax as TN_fns_jax

import FIM_functions_jax
importlib.reload(FIM_functions_jax)

import analytical_FIM_functions_np
importlib.reload(analytical_FIM_functions_np)

import training_functions_jax
importlib.reload(training_functions_jax)





### ---------------------------------------------------------------------------------------- ###
## ----------------------------------- Experiments specs ------------------------------------ ##
### ---------------------------------------------------------------------------------------- ###

# Folder in which to save results
results_folder = '/work/bd1179/b309245/fourier_models_train_and_FIM/scaling_effdim_bonddim_Vh/'

# No. of random parameter samples for evaluating normalized eff. dim.
no_samples = 200
no_par_samples_name = str(no_samples)

# No. of random TT realizations per bond dim
no_matrix_realiz = 20
no_V_samples_name = str(no_matrix_realiz)

# Vector of bond dimensions
#bond_dim_vec = np.asarray([20, 40, 60, 80, 120, 160, 240, 320])
#bond_dim_vec_names = ['20', '40', '60', '80', '120', '160', '240', '320']
bond_dim_vec = np.asarray([60, 80, 120, 160, 240, 320])
bond_dim_vec_names = ['60', '80', '120', '160', '240', '320']
#bond_dim_vec = np.asarray([480, 640])
#bond_dim_vec_names = ['480', '640']

# Input space dimension
dim_in = 60
name_dim_in = '60'

# Decay factor of correlation spectrum
dec_exp = 0.0
name_dec_exp = '0p0'

# No. of parameters
no_params = 30
name_no_params = str(no_params)

# Local dimension of parameter functions space
local_dim_param = 5
name_dim_par_loc = str(local_dim_param)

### Bounds for the (uniformly distributed) parameters
params_min = - np.pi
params_max = + np.pi





### ---------------------------------------------------------------------------------------- ###
## --------------------- Define global variables for model and FIM calc. -------------------- ##
### ---------------------------------------------------------------------------------------- ###

### Define S (input-param. correlation spectrum)
inds = np.arange(dim_in)
Svals0 = np.ones(dim_in)
Svals_np = Svals0 * np.exp(- dec_exp * inds)
Svals_np = Svals_np / np.sqrt(np.sum(Svals_np**2.0))
### Compute purity of S
purity_S = np.sum(Svals_np ** 4.0)

### Local derivative tensor
B_jnp = TN_fns_jax.derivative_tensor_jax(local_dim_param)





### ---------------------------------------------------------------------------------------- ###
## ---------------------------- Function for effective dimension ---------------------------- ##
### ---------------------------------------------------------------------------------------- ###

def eff_dim_liminf(FIMs):
    """Monte-Carlo estimator of the paper's normalized effective dimension (Eq. 12),
    evaluated as the cn -> infinity limit at large but finite cn = 1e12.

    Given a batch of normalized FIM samples F_hat(theta_i) (shape
    (nsamples, npars, npars), one per random parameter draw theta_i), computes
    logdet(I + cn * F_hat(theta_i)) for each sample and log-sum-exps them (via
    scipy.special.logsumexp) to average exp(0.5 * logdet(...)) over samples in a
    numerically stable way; dividing by log(cn) and rescaling implements the
    large-cn limit defining the effective dimension in Eq. 12. The caller further
    divides the returned value by the number of parameters to obtain the
    normalized ED in [0, 1].
    """
    cn = 1.0e12
    nsamples = FIMs.shape[0]
    npars = FIMs.shape[1]
    logdets = np.zeros(nsamples)
    for i in range(0,nsamples):
        cnF = cn * FIMs[i,:,:]
        IplusF = np.eye(npars) + cnF
        logdets[i] = np.linalg.slogdet(IplusF)[1]
    effdim = 2.0 * (scipy.special.logsumexp(0.5 * logdets) - np.log(nsamples)) / np.log(cn)
    return effdim





### ---------------------------------------------------------------------------------------- ###
## ------------------------------------- Loop over setups ----------------------------------- ##
### ---------------------------------------------------------------------------------------- ###

for ndi in range(len(bond_dim_vec)):
    bond_dim = bond_dim_vec[ndi]
    name_bond_dim = bond_dim_vec_names[ndi]

    name_end = ('_BondDim' + name_bond_dim + '_DimIn' + name_dim_in + '_DimLocPar' + name_dim_par_loc + 
                '_Nparams' + name_no_params + '_NsamplesPar' + no_par_samples_name + 
                '_NsamplesV' + no_V_samples_name + '_DecExp' + name_dec_exp)

    dims_Vtensors = TN_fns_jax.dimensions_tensor_train(dim_in, no_params, local_dim_param, bond_dim)
    maxdims_Vtensors = np.max(dims_Vtensors, axis=0)
    maxdims_Ivecs = np.asarray([1, 1, local_dim_param])

    Svals = np.zeros(maxdims_Vtensors[0])
    Svals[0:dim_in] = Svals_np
    Svals_jnp = jnp.asarray(Svals)

    bonddim_all = []
    purity_S_all = []
    norm_eff_dim_all = []

    bonddim_all_2 = []
    purity_S_all_2 = []
    purity_FIM_all_2 = []

    
    ### Loop over random model draws
    for nm in range(no_matrix_realiz):
        random_tensor_train = TN_fns_jax.generate_random_tensor_train_np_padded(maxdims_Vtensors, no_params)
        V_tensors_np = TN_fns_jax.orthogonalize_tensor_train_np_padded(random_tensor_train, dims_Vtensors, maxdims_Vtensors)
        V_tensors_jnp = jnp.asarray(V_tensors_np)
        
        @jax.jit
        def norm_FIM_jit(Ivecs_jnp):
            return TN_fns_jax.normalized_FIM_sample_jax(Ivecs_jnp, V_tensors_jnp, maxdims_Vtensors, maxdims_Ivecs, no_params, B_jnp, Svals_jnp)

        ### Loop over random parameter samples
        nFIMs = []
        for ns in range(no_samples):
            params = (params_max - params_min) * np.random.rand(no_params) + params_min
            params_jnp = jnp.asarray(params)
            loc_Ivecs_jnp = TN_fns_jax.local_basis_vectors_jaxjit(params_jnp, local_dim_param)

            ### Compute FIM
            nFIM = norm_FIM_jit(loc_Ivecs_jnp)
            nFIM = np.asarray(nFIM)
            nFIMs.append(nFIM)

            ### Compute FIM purity
            evals, _ = np.linalg.eig(nFIM)
            evals = np.real(evals)
            evals = evals / np.sum(evals)
            pur_FIM = np.sum(evals ** 2.0)

            bonddim_all_2.append(bond_dim)
            purity_S_all_2.append(purity_S)
            purity_FIM_all_2.append(pur_FIM)
            
        nFIMs = np.asarray(nFIMs)
        nED = eff_dim_liminf(nFIMs) / no_params
        
        bonddim_all.append(bond_dim)
        purity_S_all.append(purity_S)
        norm_eff_dim_all.append(nED)

    bonddim_all = np.asarray(bonddim_all)
    purity_S_all = np.asarray(purity_S_all)
    norm_eff_dim_all = np.asarray(norm_eff_dim_all)
    bonddim_all_2 = np.asarray(bonddim_all_2)
    purity_S_all_2 = np.asarray(purity_S_all_2)
    purity_FIM_all_2 = np.asarray(purity_FIM_all_2)

    dict_fim_pur = dict()
    dict_fim_pur['bonddim_all'] = bonddim_all_2
    dict_fim_pur['purity_S_all'] = purity_S_all_2
    dict_fim_pur['purity_FIM_all'] = purity_FIM_all_2

    dict_norm_ed = dict()
    dict_norm_ed['bonddim_all'] = bonddim_all
    dict_norm_ed['purity_S_all'] = purity_S_all
    dict_norm_ed['norm_eff_dim_all'] = norm_eff_dim_all
    
    filename = 'FIM_purity' + name_end + '.pkl'
    path_file = os.path.join(results_folder, filename)
    with open(path_file, 'wb') as f:
        pickle.dump(dict_fim_pur, f)

    filename = 'norm_eff_dim' + name_end + '.pkl'
    path_file = os.path.join(results_folder, filename)
    with open(path_file, 'wb') as f:
        pickle.dump(dict_norm_ed, f)
    
    print(' ******** Saved results')
    print(' ')