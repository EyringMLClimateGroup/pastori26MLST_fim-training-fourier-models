"""Scaling of the (normalized) effective dimension (ED) with the decay exponent
`dec_exp` of the correlation spectrum S (the singular values of the structure
constants Gamma = U @ diag(S) @ V^T, ordered decreasing), in a non-tensorized
(dense V) model.

For each decay exponent in `dec_exp_vec`, builds S = exp(-dec_exp * arange(dim_in))
(normalized to unit norm) and draws `no_matrix_realiz` random dense orthogonal
realizations of V (via `scipy.stats.ortho_group`), evaluates the normalized FIM at
`no_samples` random parameter draws per realization, and computes: the
correlation-spectrum purity tr(S^4) and the normalized effective dimension (Eq. 12
of the paper) via the Monte-Carlo estimator `eff_dim_liminf`. Results (decay
exponent, tr(S^4), FIM purity, normalized ED) are pickled to `results_folder` for
later plotting.

Other key control parameters set in the "Experiments specs" section (dec_exp itself
being the swept variable, via dec_exp_vec):
- no_samples: number of random parameter samples used to Monte-Carlo estimate the
  normalized effective dimension for each model draw.
- no_matrix_realiz: number of random dense orthogonal realizations of V drawn per
  decay exponent.
- dim_in: input-basis dimension D (dimension of the correlation spectrum S).
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

from jax import numpy as jnp


import scipy



# Current path for importing custom functions
path_base = '/home/b/b309245/FIM_Training_Bias_RegressionModels/fourier_models_training_and_fim/'
sys.path.insert(0, path_base + 'useful_functions')

import model_constructor_functions
importlib.reload(model_constructor_functions)
import model_constructor_functions as model_fns

import ortho_matrices_functions
importlib.reload(ortho_matrices_functions)

import tensor_network_functions_np
importlib.reload(tensor_network_functions_np)

import FIM_functions_jax
importlib.reload(FIM_functions_jax)

import analytical_FIM_functions_np
importlib.reload(analytical_FIM_functions_np)
import analytical_FIM_functions_np as an_fim_fns

import training_functions_jax
importlib.reload(training_functions_jax)





### ---------------------------------------------------------------------------------------- ###
## ----------------------------------- Experiments specs ------------------------------------ ##
### ---------------------------------------------------------------------------------------- ###

# Folder in which to save results
results_folder = '/work/bd1179/b309245/fourier_models_train_and_FIM/scaling_effdim_decay_spectrS/'

# No. of random parameter samples for evaluating normalized eff. dim.
no_samples = 150
no_par_samples_name = str(no_samples)

# No. of random orthogonal matrix realizations per S decay exponent
no_matrix_realiz = 50
no_V_samples_name = str(no_matrix_realiz)

# Vector of decay exponents
dec_exp_vec = np.asarray([0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 4.0])
dec_exp_vec_names = ['0p01', '0p02', '0p04', '0p07', '0p1', '0p2', '0p4', '0p7', '1p0', '2p0', '4p0']

# Dimension of input function space
dim_in = 10
name_dim_in = str(dim_in)

# No. of parameters
no_params = 6
name_no_params = str(no_params)

# Local dimension of parameter functions space
local_dim_param = 3
dim_basis_params = local_dim_param ** no_params
name_dim_par_loc = str(local_dim_param)

### Bounds for the (uniformly distributed) parameters
params_min = - np.pi
params_max = + np.pi





### ---------------------------------------------------------------------------------------- ###
## -------------------- Basis functions for input and params. dependence -------------------- ##
### ---------------------------------------------------------------------------------------- ###

all_freqs_p, all_offsets_p, all_norms_p = model_fns.params_basis_functions(local_dim_param, no_params)





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

for ndec in range(len(dec_exp_vec)):
    dec_exp = dec_exp_vec[ndec]
    dec_exp_name = dec_exp_vec_names[ndec]

    name_end = ('_DimIn' + name_dim_in + '_DimLocPar' + name_dim_par_loc + '_Nparams' + name_no_params + 
                '_NsamplesPar' + no_par_samples_name + '_NsamplesV' + no_V_samples_name + '_DecExpS' + dec_exp_name)

    ### Define S (input-param. correlation spectrum)
    inds = np.arange(dim_in)
    Svals0 = np.ones(dim_in)
    Svals = Svals0 * np.exp(- dec_exp * inds)
    Svals = Svals / np.sqrt(np.sum(Svals**2.0))

    ### Compute purity of S
    purity_S = np.sum(Svals ** 4.0)

    decay_exp_all = []
    purity_S_all = []
    norm_eff_dim_all = []

    decay_exp_all_2 = []
    purity_S_all_2 = []
    purity_FIM_all_2 = []

    
    ### Loop over random model draws
    for nm in range(no_matrix_realiz):
        Vfull = scipy.stats.ortho_group.rvs(dim=dim_basis_params, size=1)
        Vh = Vfull[0:dim_in, :]

        
        ### Loop over random parameter samples
        nFIMs = []
        for ns in range(no_samples):
            params = (params_max - params_min) * np.random.rand(no_params) + params_min
            basis_funcs_th = np.prod(all_norms_p * jnp.cos(all_freqs_p * params + all_offsets_p), axis=1)  ### (local_dim_param, )
            basis_funcs_th = np.expand_dims(basis_funcs_th, axis=1)  ### (local_dim_param, 1)

            ### Compute FIM
            nFIM = an_fim_fns.normalized_FIM_sample_fullmatrices(basis_funcs_th, no_params, local_dim_param, Svals, Vh)
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