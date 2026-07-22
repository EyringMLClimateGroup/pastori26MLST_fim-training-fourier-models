"""Compute the scaling of the normalized effective dimension (ED, Eq. 12) with the decay exponent
`dec_exp` of the correlation spectrum S (singular values of the structure constants Gamma,
S_i ~ exp(-dec_exp * i)) in a partially tensorized model, at fixed architecture. This directly
scans tr(S^4), the correlation-spectrum purity that controls ED, at fixed architecture.

For each value of `dec_exp` in `dec_exp_vec` (the swept control parameter), the script draws
`no_matrix_realiz` random Gamma-equivalent models (random correlation spectrum S with that decay
exponent, and random orthogonal V represented as a tensor train of bond dimension `bond_dim`),
evaluates the normalized FIM at `no_samples` random parameter draws per model via
`eff_dim_liminf`, and pickles the resulting normalized ED and correlation-spectrum purity tr(S^4)
(plus the per-sample normalized-FIM purity) for later plotting.

Other key fixed parameters set in the "Experiments specs" section (not swept here):
- `no_samples`: number of random parameter samples used to Monte-Carlo estimate the ED per model draw.
- `no_matrix_realiz`: number of random Gamma-equivalent model draws (random S/V realizations) per `dec_exp` value.
- `bond_dim`: tensor-train bond dimension chi of the V^T representation.
- `dim_in`: input-basis dimension D of the correlation spectrum S.
- `no_params`: number of trainable parameters M.
- `local_dim_param`: local per-parameter basis dimension Dloc (d_tilde).
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
results_folder = '/work/bd1179/b309245/fourier_models_train_and_FIM/scaling_effdim_TN_decay_spectrS/'

# No. of random parameter samples for evaluating normalized eff. dim.
no_samples = 200
no_par_samples_name = str(no_samples)

# No. of random TT realizations per bond dim
no_matrix_realiz = 30
no_V_samples_name = str(no_matrix_realiz)

# Vector of bond dimensions
bond_dim = 100
name_bond_dim = str(bond_dim)

# Input space dimension
dim_in = 80
name_dim_in = str(dim_in)

# No. of parameters
no_params = 40
name_no_params = str(no_params)

# Local dimension of parameter functions space
local_dim_param = 5
name_dim_par_loc = str(local_dim_param)

### Bounds for the (uniformly distributed) parameters
params_min = - np.pi
params_max = + np.pi

# Vector of decay exponents
dec_exp_vec = np.asarray([0.001, 0.002, 0.004, 0.006, 0.008, 0.01])
dec_exp_vec_names = ['0p001', '0p002', '0p004', '0p006', '0p008', '0p01']
#dec_exp_vec = np.asarray([0.02, 0.04, 0.06, 0.08, 0.1, 0.2])
#dec_exp_vec_names = ['0p02', '0p04', '0p06', '0p08', '0p1', '0p2']
#dec_exp_vec = np.asarray([0.4, 0.6, 0.8, 1.0, 2.0, 4.0])
#dec_exp_vec_names = ['0p4', '0p6', '0p8', '1p0', '2p0', '4p0']





### ---------------------------------------------------------------------------------------- ###
## --------------------- Define global variables for model and FIM calc. -------------------- ##
### ---------------------------------------------------------------------------------------- ###

dims_Vtensors = TN_fns_jax.dimensions_tensor_train(dim_in, no_params, local_dim_param, bond_dim)
maxdims_Vtensors = np.max(dims_Vtensors, axis=0)
maxdims_Ivecs = np.asarray([1, 1, local_dim_param])

### Local derivative tensor
B_jnp = TN_fns_jax.derivative_tensor_jax(local_dim_param)





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

for ndec in range(len(dec_exp_vec)):
    dec_exp = dec_exp_vec[ndec]
    name_dec_exp = dec_exp_vec_names[ndec]

    name_end = ('_BondDim' + name_bond_dim + '_DimIn' + name_dim_in + '_DimLocPar' + name_dim_par_loc + 
                '_Nparams' + name_no_params + '_NsamplesPar' + no_par_samples_name + 
                '_NsamplesV' + no_V_samples_name + '_DecExp' + name_dec_exp)

    ### Define S (input-param. correlation spectrum)
    inds = np.arange(dim_in)
    Svals0 = np.ones(dim_in)
    Svals_np = Svals0 * np.exp(- dec_exp * inds)
    Svals_np = Svals_np / np.sqrt(np.sum(Svals_np**2.0))
    ### Compute purity of S
    purity_S = np.sum(Svals_np ** 4.0)

    Svals = np.zeros(maxdims_Vtensors[0])
    Svals[0:dim_in] = Svals_np
    Svals_jnp = jnp.asarray(Svals)

    decay_exp_all = []
    purity_S_all = []
    norm_eff_dim_all = []

    decay_exp_all_2 = []
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

            decay_exp_all_2.append(dec_exp)
            purity_S_all_2.append(purity_S)
            purity_FIM_all_2.append(pur_FIM)
            
        nFIMs = np.asarray(nFIMs)
        nED = eff_dim_liminf(nFIMs) / no_params
        
        decay_exp_all.append(dec_exp)
        purity_S_all.append(purity_S)
        norm_eff_dim_all.append(nED)

    decay_exp_all = np.asarray(decay_exp_all)
    purity_S_all = np.asarray(purity_S_all)
    norm_eff_dim_all = np.asarray(norm_eff_dim_all)
    decay_exp_all_2 = np.asarray(decay_exp_all_2)
    purity_S_all_2 = np.asarray(purity_S_all_2)
    purity_FIM_all_2 = np.asarray(purity_FIM_all_2)

    dict_fim_pur = dict()
    dict_fim_pur['decay_exp_all'] = decay_exp_all_2
    dict_fim_pur['purity_S_all'] = purity_S_all_2
    dict_fim_pur['purity_FIM_all'] = purity_FIM_all_2

    dict_norm_ed = dict()
    dict_norm_ed['decay_exp_all'] = decay_exp_all
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