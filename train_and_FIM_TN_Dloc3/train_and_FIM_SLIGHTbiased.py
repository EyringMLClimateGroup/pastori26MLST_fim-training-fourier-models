"""Script for training partially tensorized partially biased models with both high and low ED.

- The bond dimension of the MPS representing V in the SVD of the structure constants is set by bond_dim.
- The ED is controlled by decay_exp: larger decay_exp means smaller ED.
- The number of input features is set by no_of_features.
- The number of parameters is set by no_params.
- The local dimension of parameter space is set by dim_basis_single_param.
- Builds the same fully-biased V-tensor-train as train_and_FIM_biased.py (via
  generate_V_orthogonal_to_params, orthogonal to params0), then perturbs it with Gaussian
  noise of strength eps via tensorized_model_constructor_functions.perturb_V_tensors to get
  pertV_tensors_jnp, which defines the actual data-generating function; eps plays the role
  of the partial-bias parameter delta_data (paper Eq. 21) and is scanned over eps_vec, so the
  data-generating function is only approximately (rather than exactly) encompassed by the
  unperturbed full/cut models, interpolating between the biased and unbiased regimes.
- For both the full and cutoff model (which share U/V and differ only in their correlation
  spectrum Svals_full vs. truncated Svals_cut), computes the normalized FIM / effective
  dimension (via eff_dim_liminf) and trains by MSE minimization over no_train_tests random
  initializations, saving delta_minMSE_full_m_cut = MSE_full_min - MSE_cut_min, plus
  bias diagnostics est_bias_full/est_bias_cut, per model draw.
- Key control params (see 'Experiments specs' below): no_of_features, no_params,
  dim_basis_single_param, bond_dim, max_frequency, cutoff, decay_exp_vec, eps_vec,
  no_rand_model_tests, no_train_tests, learning_rate, batch_size, no_epochs.
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
results_folder = '/work/bd1179/b309245/fourier_models_train_and_FIM/train_and_FIM_TN_Dloc3/'

### 'biased_data_gen': the data generating funct. is encompassed by both full and cutoff models
name_data_gen = 'SLIGHTbiased_data_gen'

### Epsilon perturb. value for generating slightly biased data generating function
#eps_vec = [0.0001, 0.0002, 0.0004, 0.0007, 0.001, 0.002, 0.004, 0.0053]
#name_eps_vec = ['0p0001', '0p0002', '0p0004', '0p0007', '0p001', '0p002', '0p004', '0p0053']
eps_vec = [0.002, 0.004, 0.0053]
name_eps_vec = ['0p002', '0p004', '0p0053']
#eps_vec = [0.0001, 0.0002, 0.0004, 0.0007, 0.001, 0.002, 0.004, 0.0053, 
#           0.007, 0.0083, 0.01, 0.0125, 0.0165, 0.02, 0.025, 0.03, 
#           0.035, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.2, 
#           0.3, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0]
#name_eps_vec = ['0p0001', '0p0002', '0p0004', '0p0007', '0p001', '0p002', '0p004', '0p0053', 
#                '0p007', '0p0083', '0p01', '0p0125', '0p0165', '0p02', '0p025', '0p03', 
#                '0p035', '0p04', '0p05', '0p06', '0p07', '0p08', '0p1', '0p2', 
#                '0p3', '0p4', '0p6', '0p8', '1p0', '2p0', '3p0', '4p0']

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
no_training_data_per_feature = 30

# Used for setting the no. of validation data
no_validation_data_per_feature = 30

# Bond dimension
bond_dim = 60
name_bond_dim = str(bond_dim)

### Cutoff index for correlations among Fourier components:
### the higher the cutoff, the more correlations the cut-off model will exhibit,
### and the higher the effective dimension will be.
### The SVs of the model's structure constants will start decaying after 'cutoff' values, i.e.,
### S_full = S0[0:]  for the full model
### S_cut[0:cutoff] = S0[0:cutoff]; S_cut[cutoff:] = decay_factor * S0[cutoff:]  for the cut model
cutoff = 7
cutoff_name = str(cutoff)

### Decay exponent for cutoff model: 
### S_cut[cutoff+i-1] = np.exp(- decay_exp * i) for i in [1, dim_basis_inputs-cutoff]
decay_exp_vec = [0.05, 0.07, 0.1, 0.2, 0.333]
decay_exp_name_vec = ['0p05', '0p07', '0p1', '0p2', '0p333']
#decay_exp_vec = [0.5, 0.7, 1.0, 2.0, 3.33]
#decay_exp_name_vec = ['0p5', '0p7', '1p0', '2p0', '3p33']





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
max_frequency = 17
name_max_freq = str(max_frequency)

### Bounds for the (uniformly distributed) inputs
input_min = - np.pi
input_max = + np.pi

### No. of frequencies
no_of_frequencies = max_frequency**no_of_features

### No. of basis states inputs
local_dim_basis_inputs = 2 * no_of_frequencies - 1
dim_basis_inputs = local_dim_basis_inputs ** no_of_features

### No. of parameters
no_params = 32
name_no_params = str(no_params)

### Bounds for the (uniformly distributed) parameters
params_min = - np.pi
params_max = + np.pi

### No. basis states per parameter
dim_basis_single_param = 3  ### (1, cos(th), sin(th))
name_dim_basis_param = str(dim_basis_single_param)
dim_basis_params = dim_basis_single_param**no_params

# No. of training data
no_training_data = no_training_data_per_feature**no_of_features

# No. of validation data
no_validation_data = no_validation_data_per_feature**no_of_features





### ---------------------------------------------------------------------------------------- ###
## -------------------------------- General model constructor ------------------------------- ##
### ---------------------------------------------------------------------------------------- ###

all_freqs, all_offsets, all_norms = model_fns.input_basis_functions(max_frequency, no_of_features)

dims_Vtensors = TN_fns_jax.dimensions_tensor_train(dim_basis_inputs, no_params, dim_basis_single_param, bond_dim)
maxdims_Vtensors = np.max(dims_Vtensors, axis=0)
maxdims_Ivecs = np.asarray([1, 1, dim_basis_single_param])

### Local derivative tensor for FIM calculation
B_jnp = TN_fns_jax.derivative_tensor_jax(dim_basis_single_param)

jnp_all_freqs = jnp.asarray(all_freqs)
jnp_all_offsets = jnp.asarray(all_offsets)
jnp_all_norms = jnp.asarray(all_norms)

def Fourier_model_constructor(inputs, params, U_jnp, Svals_jnp, V_tensors_jnp):
    return TNmodel_fns.tensorized_Fourier_model_constructor_jax(inputs, params, dim_basis_single_param, 
                                                                no_params, dim_basis_inputs,
                                                                U_jnp, Svals_jnp, V_tensors_jnp, 
                                                                maxdims_Vtensors, maxdims_Ivecs,
                                                                jnp_all_freqs, jnp_all_offsets, jnp_all_norms)





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
    """Monte-Carlo estimate of the normalized effective dimension (paper Eq. 12) in the
    large-c_n limit (cn=1e12), averaged over the sampled-parameter batch of normalized FIMs."""
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

for neps in range(len(eps_vec)):
    eps = eps_vec[neps]
    name_eps = name_eps_vec[neps]
    
    for ndec in range(len(decay_exp_vec)):
        decay_exp = decay_exp_vec[ndec]
        decay_exp_name = decay_exp_name_vec[ndec]
    
        ### Decay factors for 'cutoff' model
        decay_factors = np.exp(- decay_exp * (np.arange(0, dim_basis_inputs-cutoff) + 1.0))
    
    
        ####################### Loop over random model draws and random data gen. functions #######################
        for nm in range(no_rand_model_tests):
            name_model = str(nm)
            name_end = ('_BondDim' + name_bond_dim + '_Nfeatures' + name_no_features + '_MaxFreq' + name_max_freq + 
                        '_Nparams' + name_no_params + '_DimBasisParam' + name_dim_basis_param + '_' + name_data_gen + 
                        '_Eps' + name_eps + '_batch' + batch_size_name + '_lr' + learning_rate_name + 
                        '_cutoff' + cutoff_name + '_decayexp' + decay_exp_name + '_modeldraw' + name_model)
    
    
            ################# Define data-generating function that can be learnt from both models #################
            params0 = (params_max - params_min) * np.random.rand(no_params) + params_min
            params0_jnp = jnp.asarray(params0)
            Ivecs0_jnp = TN_fns_jax.local_basis_vectors_jaxjit(params0_jnp, dim_basis_single_param)
            
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

            ### Perturbed V tensors for data generating function
            pertV_tensors_jnp = TNmodel_fns.perturb_V_tensors(eps, V_tensors_jnp, no_params, dim_basis_single_param, dims_Vtensors)
            VI0 = TN_fns_jax.contract_tensortrain_with_localvectors_jaxjit(V_tensors_jnp, Ivecs0_jnp, no_params, maxdims_Vtensors, maxdims_Ivecs)
            VI0 = np.squeeze(np.asarray(VI0[0:dim_basis_inputs]))
            pVI0 = TN_fns_jax.contract_tensortrain_with_localvectors_jaxjit(pertV_tensors_jnp, Ivecs0_jnp, no_params, maxdims_Vtensors, maxdims_Ivecs)
            pVI0 = np.squeeze(np.asarray(pVI0[0:dim_basis_inputs]))

            ### Estimate models' biases
            est_bias_full = np.sum(np.abs(Svals_full*VI0 - Svals_full*pVI0))
            est_bias_cut = np.sum(np.abs(Svals_cut*VI0 - Svals_full*pVI0))
    
    
            ############################# Define full and cut models for JAX backprop #############################
            def Fourier_model_data(params, inputs):
                return Fourier_model_constructor(inputs, params, U_jnp, Svals_full_jnp, pertV_tensors_jnp)
            
            def Fourier_model_full(params, inputs):
                return Fourier_model_constructor(inputs, params, U_jnp, Svals_full_jnp, V_tensors_jnp)
            
            def Fourier_model_cut(params, inputs):
                return Fourier_model_constructor(inputs, params, U_jnp, Svals_cut_jnp, V_tensors_jnp)
            
            
            ####################################### Define training outputs #######################################
            train_outputs = Fourier_model_data(params0, jnp_train_inputs)
            val_outputs = Fourier_model_data(params0, jnp_val_inputs)
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

            dict_biases = dict()
            dict_biases['model_bias_full'] = est_bias_full
            dict_biases['model_bias_cut'] = est_bias_cut
            
            filename = 'model_biases' + name_end + '.pkl'
            path_file = os.path.join(results_folder, filename)
            with open(path_file, 'wb') as f:
                pickle.dump(dict_biases, f)
                
            print(' ******** Saved results')
            print(' ')