"""Compute the scaling of the normalized effective dimension (ED, Eq. 12) with the local
per-parameter basis dimension Dloc (`local_dim_param`, i.e. d_tilde) in a partially tensorized
model, at fixed tensor-train bond dimension.

For each value of Dloc in `local_dim_param_vec` (the swept control parameter), the script draws
`no_matrix_realiz` random Gamma-equivalent models (random correlation spectrum S with decay
exponent `dec_exp`, and random orthogonal V represented as a tensor train of bond dimension
`bond_dim`), evaluates the normalized FIM at `no_samples` random parameter draws per model via
`eff_dim_liminf`, and pickles the resulting normalized ED and correlation-spectrum purity tr(S^4)
(plus the per-sample normalized-FIM purity) for later plotting.

Other key fixed parameters set in the "Experiments specs" section (not swept here):
- `no_samples`: number of random parameter samples used to Monte-Carlo estimate the ED per model draw.
- `no_matrix_realiz`: number of random Gamma-equivalent model draws (random S/V realizations) per Dloc value.
- `dim_in`: input-basis dimension D of the correlation spectrum S.
- `bond_dim`: tensor-train bond dimension chi of the V^T representation.
- `dec_exp`: decay exponent of the correlation spectrum S (S_i ~ exp(-dec_exp * i)).
- `no_params`: number of trainable parameters M.
- `params_min`, `params_max`: bounds of the uniform distribution used to sample parameters theta.
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
results_folder = '/work/bd1179/b309245/fourier_models_train_and_FIM/scaling_effdim_TN_Dloc/'

# No. of random parameter samples for evaluating normalized eff. dim.
no_samples = 200
no_par_samples_name = str(no_samples)

# No. of random TT realizations per bond dim
no_matrix_realiz = 30
no_V_samples_name = str(no_matrix_realiz)

# Input space dimension
dim_in = 90
name_dim_in = str(dim_in)

# Vector of bond dimensions
bond_dim = 100
name_bond_dim = str(bond_dim)

### Bounds for the (uniformly distributed) parameters
params_min = - np.pi
params_max = + np.pi

# Decay exponent
dec_exp = 0.0
name_dec_exp = '0p0'

# Vector of no. of parameters
no_params = 60
name_no_params = str(no_params)

# Vector of local dimension of parameter functions space
local_dim_param_vec = [3, 5, 31]
#local_dim_param_vec = [7, 9, 29]
#local_dim_param_vec = [11, 13, 27]
#local_dim_param_vec = [15, 17, 25]
#local_dim_param_vec = [19, 21, 23]





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





### ---------------------------------------------------------------------------------------- ###
## ---------------------------- Function for effective dimension ---------------------------- ##
### ---------------------------------------------------------------------------------------- ###

def eff_dim_liminf(FIMs):
    """Monte-Carlo estimator of the normalized effective dimension (Eq. 12 of the paper), taken
    in the large-c_n limit (cn=1e12). `FIMs` is a batch of normalized FIM samples F_hat(theta)
    (one per random parameter draw theta); the estimator averages det(I + cn*F_hat(theta)) over
    the batch (via logsumexp for numerical stability) and rescales by log(cn)."""
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

for local_dim_param in local_dim_param_vec:
    name_dim_par_loc = str(local_dim_param)
    
    name_end = ('_BondDim' + name_bond_dim + '_DimIn' + name_dim_in + '_Nparams' + name_no_params +
                '_DecExp' + name_dec_exp  + '_NsamplesPar' + no_par_samples_name + 
                '_NsamplesV' + no_V_samples_name + '_DimLocPar' + name_dim_par_loc)

    ### Local derivative tensor
    B_jnp = TN_fns_jax.derivative_tensor_jax(local_dim_param)

    dims_Vtensors = TN_fns_jax.dimensions_tensor_train(dim_in, no_params, local_dim_param, bond_dim)
    maxdims_Vtensors = np.max(dims_Vtensors, axis=0)
    maxdims_Ivecs = np.asarray([1, 1, local_dim_param])

    Svals = np.zeros(maxdims_Vtensors[0])
    Svals[0:dim_in] = Svals_np
    Svals_jnp = jnp.asarray(Svals)

    loc_dim_all = []
    purity_S_all = []
    norm_eff_dim_all = []

    loc_dim_all_2 = []
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

            loc_dim_all_2.append(local_dim_param)
            purity_S_all_2.append(purity_S)
            purity_FIM_all_2.append(pur_FIM)
            
        nFIMs = np.asarray(nFIMs)
        nED = eff_dim_liminf(nFIMs) / no_params
        
        loc_dim_all.append(local_dim_param)
        purity_S_all.append(purity_S)
        norm_eff_dim_all.append(nED)

    loc_dim_all = np.asarray(loc_dim_all)
    purity_S_all = np.asarray(purity_S_all)
    norm_eff_dim_all = np.asarray(norm_eff_dim_all)
    loc_dim_all_2 = np.asarray(loc_dim_all_2)
    purity_S_all_2 = np.asarray(purity_S_all_2)
    purity_FIM_all_2 = np.asarray(purity_FIM_all_2)

    dict_fim_pur = dict()
    dict_fim_pur['loc_dim_all'] = loc_dim_all_2
    dict_fim_pur['purity_S_all'] = purity_S_all_2
    dict_fim_pur['purity_FIM_all'] = purity_FIM_all_2

    dict_norm_ed = dict()
    dict_norm_ed['loc_dim_all'] = loc_dim_all
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