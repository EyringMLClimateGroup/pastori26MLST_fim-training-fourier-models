"""Scaling of the (normalized) effective dimension (ED) with the input-basis
dimension D (`dim_in`), i.e. the dimension of the correlation spectrum S in the SVD
Gamma = U @ diag(S) @ V^T of the structure constants, in a non-tensorized (dense V)
model.

For each D in `dim_in_vec`, the decay exponent of S is first solved for (via
`scipy.optimize.minimize` on helper `purityS`) so that the correlation-spectrum
purity tr(S^4) matches a fixed target `purity0` regardless of D; `no_matrix_realiz`
random dense orthogonal realizations of V (via `scipy.stats.ortho_group`) are then
drawn, the normalized FIM is evaluated at `no_samples` random parameter draws per
realization, and the normalized effective dimension (Eq. 12 of the paper) is computed
via the Monte-Carlo estimator `eff_dim_liminf`. Results (input dimension, tr(S^4),
FIM purity, normalized ED) are pickled to `results_folder` for later plotting.

Other key control parameters set in the "Experiments specs" section (dim_in itself
being the swept variable, via dim_in_vec):
- no_samples: number of random parameter samples used to Monte-Carlo estimate the
  normalized effective dimension for each model draw.
- no_matrix_realiz: number of random dense orthogonal realizations of V drawn per
  input dimension.
- purity0: fixed target value of the correlation-spectrum purity tr(S^4) that the
  decay exponent is tuned to reproduce at each input dimension, so ED vs. D is
  compared at constant spectrum purity.
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
results_folder = '/work/bd1179/b309245/fourier_models_train_and_FIM/scaling_effdim_dimInSpace/'

# No. of random parameter samples for evaluating normalized eff. dim.
no_samples = 150
no_par_samples_name = str(no_samples)

# No. of random orthogonal matrix realizations per input dim
no_matrix_realiz = 50
no_V_samples_name = str(no_matrix_realiz)

# Vector of decay exponents
#dim_in_vec = np.asarray([3, 5, 7, 10, 20, 30, 50])
#dim_in_vec = np.asarray([70, 100, 200])
#dim_in_vec = np.asarray([300, 500])
dim_in_vec = np.asarray([30, 50])

# Fixed purity of S
purity0 = 0.333
purity0_name = '0p333'

# No. of parameters
no_params = 7
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

def purityS(dec_exp, dim_in):
    inds = np.arange(dim_in)
    Svals0 = np.ones(dim_in)
    Svals = Svals0 * np.exp(- dec_exp * inds)
    Svals = Svals / np.sqrt(np.sum(Svals**2.0))
    purity_S = np.sum(Svals ** 4.0)
    return purity_S

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

for ndi in range(len(dim_in_vec)):
    dim_in = dim_in_vec[ndi]
    name_dim_in = str(dim_in)

    name_end = ('_DimIn' + name_dim_in + '_DimLocPar' + name_dim_par_loc + '_Nparams' + name_no_params + 
                '_NsamplesPar' + no_par_samples_name + '_NsamplesV' + no_V_samples_name + '_S4trace' + purity0_name)

    def loss(d):
        return (purityS(d, dim_in) - purity0)**2.0

    res = scipy.optimize.minimize(loss, 0.0, method='Nelder-Mead', bounds=[(0.0, None)]) 
    dec_exp = res.x[0]

    ### Define S (input-param. correlation spectrum)
    inds = np.arange(dim_in)
    Svals0 = np.ones(dim_in)
    Svals = Svals0 * np.exp(- dec_exp * inds)
    Svals = Svals / np.sqrt(np.sum(Svals**2.0))

    ### Compute purity of S
    purity_S = np.sum(Svals ** 4.0)

    dim_in_all = []
    purity_S_all = []
    norm_eff_dim_all = []

    dim_in_all_2 = []
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

            dim_in_all_2.append(dim_in)
            purity_S_all_2.append(purity_S)
            purity_FIM_all_2.append(pur_FIM)
            
        nFIMs = np.asarray(nFIMs)
        nED = eff_dim_liminf(nFIMs) / no_params
        
        dim_in_all.append(dim_in)
        purity_S_all.append(purity_S)
        norm_eff_dim_all.append(nED)

    dim_in_all = np.asarray(dim_in_all)
    purity_S_all = np.asarray(purity_S_all)
    norm_eff_dim_all = np.asarray(norm_eff_dim_all)
    dim_in_all_2 = np.asarray(dim_in_all_2)
    purity_S_all_2 = np.asarray(purity_S_all_2)
    purity_FIM_all_2 = np.asarray(purity_FIM_all_2)

    dict_fim_pur = dict()
    dict_fim_pur['dim_in_all'] = dim_in_all_2
    dict_fim_pur['purity_S_all'] = purity_S_all_2
    dict_fim_pur['purity_FIM_all'] = purity_FIM_all_2

    dict_norm_ed = dict()
    dict_norm_ed['dim_in_all'] = dim_in_all
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