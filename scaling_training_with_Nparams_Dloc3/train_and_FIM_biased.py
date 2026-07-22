"""Scaling of training performance and effective dimension with Nparams at fixed Dloc=3 (biased data generation).

Scans the control parameter Nparams (`no_params`, over the values in `no_params_vec`, e.g.
[5, 10, 15, 20, 25, 30, 35]) at fixed local per-parameter basis dimension Dloc = 3
(`dim_basis_single_param`, corresponding to the local basis (1, cos(theta), sin(theta))) and fixed
`max_frequency`. For a partially tensorized partial Fourier series model f_theta(x) =
sum_mu c_mu(theta) e_mu(x) with c_mu(theta) = sum_nu Gamma_{mu,nu} iota_nu(theta),
Gamma = U @ diag(S) @ V^T (dense U on the input side, V^T a tensor train of bond dimension
`bond_dim` on the parameter side).

The data-generating function used here is BIASED with respect to the full and cutoff models: no
separate `Fourier_model_data` is drawn independently -- instead, the target y(x) is directly
`Fourier_model_full(params0, x)`, i.e. the full model itself evaluated at a random parameter
configuration `params0`. Both the full and cutoff models share the same random orthogonal matrix
`U` and the same parameter-side tensor train `V_tensors_jnp`, the latter generated via
`tensorized_model_constructor_functions.generate_V_orthogonal_to_params(params0, cutoff, ...)` so
that the truncated ("cutoff") correlation spectrum still reproduces y(x) exactly at `params0`.
Thus y(x) lies exactly in the expressible function space of both models for theta0 = params0.

For each of `no_rand_model_tests` random draws of (U, V, params0) at a given Nparams value, this
script builds the "full" model (correlation spectrum Svals_full = all ones, i.e. all D singular
values comparable -> high effective dimension) and the "cutoff" model (Svals_cut equal to
Svals_full up to index `cutoff`, then decayed as exp(-decay_exp * i) beyond it -> lower effective
dimension), computes the normalized FIM spectrum and normalized effective dimension (Eq. 12 of the
paper) of both models via Monte-Carlo sampling of `no_params_samples` random parameter points,
trains both models against MSE for `no_train_tests` random parameter initializations, and pickles
Delta_{f-c}MSE_min = MSE_full_min - MSE_cut_min together with the FIM spectra and effective
dimensions of both models.

Other key control parameters (see 'Experiments specs' section below):
    no_rand_model_tests -- number of random (U, V, params0) model draws per Nparams value
    no_train_tests       -- number of random parameter initializations trained per model draw
    no_params_samples    -- number of random parameter samples for Monte-Carlo FIM/eff. dim. estimation
    learning_rate        -- Adam learning rate used for training
    batch_size           -- minibatch size used for training
    no_epochs            -- number of training epochs
    bond_dim             -- bond dimension of the tensor-train parametrization of V^T
    max_frequency        -- fixed maximum Fourier input frequency (held constant across the Nparams scan)
    cutoff               -- index after which the cutoff model's correlation spectrum starts decaying
    decay_exp            -- exponential decay rate of the cutoff model's spectrum beyond `cutoff`
"""

# Importing necessary packages
import sys
import os
import copy
import importlib
import pickle

import pennylane.numpy as np

import jax
from jax import numpy as jnp
import optax


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

import training_functions_jax
importlib.reload(training_functions_jax)
import training_functions_jax as jax_train_fns

import tensorized_model_constructor_functions
importlib.reload(tensorized_model_constructor_functions)
import tensorized_model_constructor_functions as TNmodel_fns

import tensor_network_functions_jax
importlib.reload(tensor_network_functions_jax)
import tensor_network_functions_jax as TN_fns_jax





### ---------------------------------------------------------------------------------------- ###
## ----------------------------------- Experiments specs ------------------------------------ ##
### ---------------------------------------------------------------------------------------- ###

# Folder in which to save results
results_folder = '/work/bd1179/b309245/fourier_models_train_and_FIM/scaling_training_with_Nparams_Dloc3/'

### 'biased_data_gen': the data generating funct. is encompassed by both full and cutoff models
name_data_gen = 'biased_data_gen'

# No. of random model draws
no_rand_model_tests = 30

# No. of training tests per model
no_train_tests = 30

# No. of param. samples for eff. dim.
no_params_samples = 200

# Learning rate
learning_rate = 0.02
learning_rate_name = '0p02'

# Batch size for training
batch_size = 5
batch_size_name = str(batch_size)

# No. of training epochs
no_epochs = 250

# Used for setting the no. of training data
# no_training_data_per_feature: 10, 15, 20, 30, 40, 60
no_training_data_per_feature = 30

# Used for setting the no. of validation data
no_validation_data_per_feature = no_training_data_per_feature

# Bond dimension
bond_dim = 50
name_bond_dim = str(bond_dim)

### Cutoff index for correlations among Fourier components:
### the higher the cutoff, the more correlations the cut-off model will exhibit,
### and the higher the effective dimension will be.
### The SVs of the model's structure constants will start decaying after 'cutoff' values, i.e.,
### S_full = S0[0:]  for the full model
### S_cut[0:cutoff] = S0[0:cutoff]; S_cut[cutoff:] = decay_factor * S0[cutoff:]  for the cut model
cutoff = 3
cutoff_name = str(cutoff)

### Decay exponent for cutoff model: 
### S_cut[cutoff+i-1] = np.exp(- decay_exp * i) for i in [1, dim_basis_inputs-cutoff]
decay_exp = 3.0
decay_exp_name = '3p0'

### No. of parameters
no_params_vec = [5, 10, 15, 20, 25, 30, 35]
#no_params_vec = [40, 50, 60, 70]
#no_params_vec = [80, 90, 100]





### ---------------------------------------------------------------------------------------- ###
## ------------------------------------ Basic model specs ----------------------------------- ##
### ---------------------------------------------------------------------------------------- ###

###
### Model frequencies defined on a 'square lattice' in d dimensions,
### with d=no_of_features, where the lattice points have integer frequencies
### W = (w_1,...,w_d) with w_j=[0,max_freq-1]
###

### No. of features (dimension of input vectors)
no_of_features = 1
name_no_features = str(no_of_features)

### Maximal Fourier frequency
max_frequency = 25
name_max_freq = str(max_frequency)

### Bounds for the (uniformly distributed) inputs
input_min = - np.pi
input_max = + np.pi

### No. of frequencies
no_of_frequencies = max_frequency**no_of_features

### No. of basis states inputs
local_dim_basis_inputs = 2 * no_of_frequencies - 1
dim_basis_inputs = local_dim_basis_inputs ** no_of_features

### Bounds for the (uniformly distributed) parameters
params_min = - np.pi
params_max = + np.pi

### No. basis states per parameter
dim_basis_single_param = 3  ### (1, cos(th), sin(th))
name_dim_basis_param = str(dim_basis_single_param)

# No. of training data
no_training_data = no_training_data_per_feature**no_of_features

# No. of validation data
no_validation_data = no_validation_data_per_feature**no_of_features

### Decay factors for 'cutoff' model
decay_factors = np.exp(- decay_exp * (np.arange(0, dim_basis_inputs-cutoff) + 1.0))





### ---------------------------------------------------------------------------------------- ###
## ----------------------------------- General model specs ---------------------------------- ##
### ---------------------------------------------------------------------------------------- ###

all_freqs, all_offsets, all_norms = model_fns.input_basis_functions(max_frequency, no_of_features)

### Local derivative tensor for FIM calculation
B_jnp = TN_fns_jax.derivative_tensor_jax(dim_basis_single_param)

jnp_all_freqs = jnp.asarray(all_freqs)
jnp_all_offsets = jnp.asarray(all_offsets)
jnp_all_norms = jnp.asarray(all_norms)





### ---------------------------------------------------------------------------------------- ###
## ----------------------------------- Inputs for training ---------------------------------- ##
### ---------------------------------------------------------------------------------------- ###

train_inputs = np.zeros((no_training_data, no_of_features))
xi_vals = np.arange(input_min, input_max, (input_max-input_min)/no_training_data_per_feature)
if no_of_features==1:
    train_inputs = xi_vals
    train_inputs = np.expand_dims(train_inputs, axis=1)
else:
    for i in range(0,no_of_features):
        xi = 1.0
        for j in range(0,no_of_features):
            xi = np.kron(xi, xi_vals**(j==i))
        train_inputs[:, i] = xi

val_inputs = np.zeros((no_validation_data, no_of_features))
xi_vals = np.arange(input_min, input_max, (input_max-input_min)/no_validation_data_per_feature)
if no_of_features==1:
    val_inputs = xi_vals
    val_inputs = np.expand_dims(val_inputs, axis=1)
else:
    for i in range(0,no_of_features):
        xi = 1.0
        for j in range(0,no_of_features):
            xi = np.kron(xi, xi_vals**(j==i))
        val_inputs[:, i] = xi
val_inputs = val_inputs + (input_max - input_min) / no_training_data_per_feature / 2.0  ### shift slighly to make it different than trainset

jnp_train_inputs = jnp.asarray(train_inputs)
jnp_val_inputs = jnp.asarray(val_inputs)

# Loss function used (currently only MSE supported)
loss = jax_train_fns.mse_loss





### ---------------------------------------------------------------------------------------- ###
## ---------------------------- Function for effective dimension ---------------------------- ##
### ---------------------------------------------------------------------------------------- ###

def eff_dim_liminf(FIMs):
    """Monte-Carlo estimator of the paper's normalized effective dimension (Eq. 12).

    Given a batch of normalized FIM samples `FIMs` (shape (nsamples, npars, npars)), evaluates
    the effective-dimension functional in the large-c_n limit (here approximated with the finite
    but very large constant cn=1e12) by averaging det(I + cn*FIM) over the sampled parameter
    points (via a numerically stable logsumexp over 0.5*logdet), and returns the un-normalized
    effective dimension estimate (the caller divides by the number of parameters to obtain the
    [0,1]-normalized effective dimension ED).
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
## ------------------ Loop over different decay factors for truncated model ----------------- ##
### ---------------------------------------------------------------------------------------- ###

for no_params in no_params_vec:
    name_no_params = str(no_params)

    dims_Vtensors = TN_fns_jax.dimensions_tensor_train(dim_basis_inputs, no_params, dim_basis_single_param, bond_dim)
    maxdims_Vtensors = np.max(dims_Vtensors, axis=0)
    maxdims_Ivecs = np.asarray([1, 1, dim_basis_single_param])

    def Fourier_model_constructor(inputs, params, U_jnp, Svals_jnp, V_tensors_jnp):
        return TNmodel_fns.tensorized_Fourier_model_constructor_jax(inputs, params, dim_basis_single_param, 
                                                                    no_params, dim_basis_inputs,
                                                                    U_jnp, Svals_jnp, V_tensors_jnp, 
                                                                    maxdims_Vtensors, maxdims_Ivecs,
                                                                    jnp_all_freqs, jnp_all_offsets, jnp_all_norms)


    ####################### Loop over random model draws and random data gen. functions #######################
    for nm in range(no_rand_model_tests):
        name_model = str(nm)
        name_end = ('_BondDim' + name_bond_dim + '_Nfeatures' + name_no_features + '_MaxFreq' + name_max_freq + 
                    '_Nparams' + name_no_params + '_DimBasisParam' + name_dim_basis_param + '_' + name_data_gen + 
                    '_NtrainPerFeat' + str(no_training_data_per_feature) + '_batch' + batch_size_name + 
                    '_lr' + learning_rate_name + '_cutoff' + cutoff_name + '_decayexp' + decay_exp_name + 
                    '_modeldraw' + name_model)


        ################# Define data-generating function that can be learnt from both models #################
        params0 = (params_max - params_min) * np.random.rand(no_params) + params_min
        params0_jnp = jnp.asarray(params0)
        
        ### Define random U matrix
        U = scipy.stats.ortho_group.rvs(dim=dim_basis_inputs, size=1)
        U_jnp = jnp.asarray(U)
        
        ### Define random V tensor train orthogonalized w.r.t. parameters params0
        V_tensors_jnp = TNmodel_fns.generate_V_orthogonal_to_params(params0_jnp, cutoff, dim_basis_inputs, no_params, dim_basis_single_param, 
                                                                    dims_Vtensors, maxdims_Vtensors, maxdims_Ivecs)
        
        ### Define correlation spectrum for full model
        Svals_full = np.ones(dim_basis_inputs)
        Svals_full_jnp = jnp.asarray(Svals_full)
        
        ### Define correlation spectrum for cut model
        Svals_cut = copy.deepcopy(Svals_full)
        Svals_cut[cutoff:] = decay_factors * Svals_cut[cutoff:]
        Svals_cut_jnp = jnp.asarray(Svals_cut)

        ### Pad Svals vectors with 0s for computing FIM
        Svals_full_jnp_pad = jnp.zeros(maxdims_Vtensors[0])
        Svals_full_jnp_pad = Svals_full_jnp_pad.at[0:dim_basis_inputs].set(Svals_full_jnp)
        Svals_cut_jnp_pad = jnp.zeros(maxdims_Vtensors[0])
        Svals_cut_jnp_pad = Svals_cut_jnp_pad.at[0:dim_basis_inputs].set(Svals_cut_jnp)


        ############################# Define full and cut models for JAX backprop #############################
        def Fourier_model_full(params, inputs):
            return Fourier_model_constructor(inputs, params, U_jnp, Svals_full_jnp, V_tensors_jnp)
        
        def Fourier_model_cut(params, inputs):
            return Fourier_model_constructor(inputs, params, U_jnp, Svals_cut_jnp, V_tensors_jnp)
        
        
        ####################################### Define training outputs #######################################
        train_outputs = Fourier_model_full(params0, jnp_train_inputs)
        val_outputs = Fourier_model_full(params0, jnp_val_inputs)
        jnp_train_outputs = jnp.asarray(train_outputs)
        jnp_val_outputs = jnp.asarray(val_outputs)


        ################################# Define routine for FIM calculation ##################################
        @jax.jit
        def norm_FIM_jit(Ivecs_jnp, Svals_jnp):
            return TN_fns_jax.normalized_FIM_sample_jax(Ivecs_jnp, V_tensors_jnp, maxdims_Vtensors, maxdims_Ivecs, no_params, B_jnp, Svals_jnp)

        
        ########################### Define training routine for full and cut models ###########################
        ### Optimizer chosen
        opt = optax.adam(learning_rate=learning_rate, eps=1e-07)
        
        ### Define JIT compiled training loop by wrapping training routine
        @jax.jit
        def train_model_full_jit(params):
            args_opt = (opt, loss, no_epochs, batch_size)
            opt_params, loss_history = jax_train_fns.train_model_noprint(args_opt, Fourier_model_full, params, jnp_train_inputs, 
                                                                         jnp_train_outputs, jnp_val_inputs, jnp_val_outputs)
            return opt_params, loss_history
        
        ### Define JIT compiled training loop by wrapping training routine
        @jax.jit
        def train_model_cut_jit(params):
            args_opt = (opt, loss, no_epochs, batch_size)
            opt_params, loss_history = jax_train_fns.train_model_noprint(args_opt, Fourier_model_cut, params, jnp_train_inputs, 
                                                                         jnp_train_outputs, jnp_val_inputs, jnp_val_outputs)
            return opt_params, loss_history


        ######################################### Calculate eff. dim. #########################################
        nFIMs_full = []
        nFIMs_cut = []
        nFIMspectra_full = []
        nFIMspectra_cut = []
        for nps in range(no_params_samples):
            params_sample = (params_max - params_min) * np.random.rand(no_params) + params_min
            params_sample = jnp.asarray(params_sample)
            loc_Ivecs_jnp = TN_fns_jax.local_basis_vectors_jaxjit(params_sample, dim_basis_single_param)
            ### Compute FIM
            FIM_full = norm_FIM_jit(loc_Ivecs_jnp, Svals_full_jnp_pad)
            FIM_cut = norm_FIM_jit(loc_Ivecs_jnp, Svals_cut_jnp_pad)
            FIM_full = np.asarray(FIM_full)
            FIM_cut = np.asarray(FIM_cut)
            evalsF, _ = np.linalg.eig(FIM_full)
            III = np.argsort(np.real(evalsF))
            III = III[::-1]
            evalsF = evalsF[III]
            nFIMspectra_full.append(np.real(evalsF))
            evalsF, _ = np.linalg.eig(FIM_cut)
            III = np.argsort(np.real(evalsF))
            III = III[::-1]
            evalsF = evalsF[III]
            nFIMspectra_cut.append(np.real(evalsF))
            nFIMs_full.append(FIM_full)
            nFIMs_cut.append(FIM_cut)
        nFIMspectra_full = np.asarray(nFIMspectra_full)
        nFIMspectra_cut = np.asarray(nFIMspectra_cut)
        nFIMs_full = np.asarray(nFIMs_full)
        nFIMs_cut = np.asarray(nFIMs_cut)
        nED_full = eff_dim_liminf(nFIMs_full) / no_params
        nED_cut = eff_dim_liminf(nFIMs_cut) / no_params
    

        ########################## Loop over different random params. initialization ##########################
        list_train_mses_full = []
        list_train_mses_cut = []
        list_delta_mse = []
        for nt in range(no_train_tests):
            params_sample = (params_max - params_min) * np.random.rand(no_params) + params_min
            params_sample = jnp.asarray(params_sample)


            ######################################## Train both models ########################################
            opt_params_full, loss_history_full = train_model_full_jit(params_sample)
            opt_params_cut, loss_history_cut = train_model_cut_jit(params_sample)

            train_mse_full = np.squeeze(loss_history_full['train_loss'])
            list_train_mses_full.append(train_mse_full)
            train_mse_cut = np.squeeze(loss_history_cut['train_loss'])
            list_train_mses_cut.append(train_mse_cut)
            min_MSE_full = np.min(train_mse_full)
            min_MSE_cut = np.min(train_mse_cut)
            delta_minMSE_full_m_cut = min_MSE_full - min_MSE_cut
            list_delta_mse.append(delta_minMSE_full_m_cut)

        train_mses_full = np.asarray(list_train_mses_full)
        train_mses_cut = np.asarray(list_train_mses_cut)
        list_delta_mse = np.asarray(list_delta_mse)

        dict_model = dict()
        dict_model['delta_minMSE_full_m_cut'] = list_delta_mse
        dict_model['MSE_full_mean'] = np.mean(train_mses_full, axis=0)
        dict_model['MSE_full_std'] = np.std(train_mses_full, axis=0)
        dict_model['MSE_full_min'] = np.min(train_mses_full, axis=0)
        dict_model['MSE_full_max'] = np.max(train_mses_full, axis=0)
        dict_model['MSE_cut_mean'] = np.mean(train_mses_cut, axis=0)
        dict_model['MSE_cut_std'] = np.std(train_mses_cut, axis=0)
        dict_model['MSE_cut_min'] = np.min(train_mses_cut, axis=0)
        dict_model['MSE_cut_max'] = np.max(train_mses_cut, axis=0)
        dict_model['mean_normFIM_spectra_full'] = np.mean(nFIMspectra_full, axis=0)
        dict_model['std_normFIM_spectra_full'] = np.std(nFIMspectra_full, axis=0)
        dict_model['min_normFIM_spectra_full'] = np.min(nFIMspectra_full, axis=0)
        dict_model['max_normFIM_spectra_full'] = np.max(nFIMspectra_full, axis=0)
        dict_model['norm_eff_dim_full'] = nED_full
        dict_model['mean_normFIM_spectra_cut'] = np.mean(nFIMspectra_cut, axis=0)
        dict_model['std_normFIM_spectra_cut'] = np.std(nFIMspectra_cut, axis=0)
        dict_model['min_normFIM_spectra_cut'] = np.min(nFIMspectra_cut, axis=0)
        dict_model['max_normFIM_spectra_cut'] = np.max(nFIMspectra_cut, axis=0)
        dict_model['norm_eff_dim_cut'] = nED_cut
    
        filename = 'dict_results' + name_end + '.pkl'
        path_file = os.path.join(results_folder, filename)
        with open(path_file, 'wb') as f:
            pickle.dump(dict_model, f)
            
        print(' ******** Saved results')
        print(' ')