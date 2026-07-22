"""Script for training fully tensorized unbiased models with both high and low ED.

- The bond dimension of the TNs representing the SVD of the structure constants is set by bond_dim.
- The ED is controlled by decay_exp: larger decay_exp means smaller ED.
- The number of input features is set by no_of_features.
- The number of parameters is set by no_params.
- The local dimension of parameter space is set by dim_basis_single_param.

Unbiased model construction:
- An independently-drawn MPO+TTN isometric mapping ('U') together with an independently-drawn,
  right-orthogonalized tensor-train ('V', Vdata_tensors_jnp) and a flat correlation spectrum
  (Svals_data_jnp) define the data-generating function, unrelated to the trained models.
- The full and cutoff models instead share their own independently-drawn MPO+TTN mapping and
  tensor-train V (V_tensors_jnp), differing only in the model's correlation spectrum S: flat
  (Svals_full) for the full model vs. decayed after index cutoff, at rate decay_exp
  (Svals_cut), for the cutoff model.

Workflow per model draw:
- Compute the normalized FIM and effective dimension (via the local eff_dim_liminf helper)
  for both the full and cutoff models, averaged over sampled parameter configurations.
- Train both models by MSE minimization from several random parameter initializations.
- Save Delta_{f-c}MSE_min = MSE_full_min - MSE_cut_min (expected negative here: higher ED
  trains better for unbiased data) together with training curves and FIM spectra to a pickle
  file.

Experiments specs (key control params):
- no_of_features: number of input features (dimension of input vectors).
- no_params: number of variational parameters theta.
- dim_basis_single_param: local per-parameter basis dimension d_tilde (here 3: 1, cos(th), sin(th)).
- bond_dim: bond dimension of the MPO/TTN (U) and tensor-train (V) containers.
- dim_corr_space: dimension of the correlation space that input features are mapped into.
- max_frequency: maximal Fourier frequency.
- cutoff: index after which the correlation-spectrum values start decaying in the cutoff model.
- decay_exp: decay exponent for the cutoff model's spectrum past cutoff.
- no_rand_model_tests: number of random full/cutoff model-pair draws.
- ini_counter_draws: starting index used to label the model draws.
- no_train_tests: number of independent training runs (random parameter inits) per model.
- learning_rate: Adam optimizer learning rate.
- batch_size: mini-batch size used during training.
- no_epochs: number of training epochs.
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

import tensor_network_functions_jax
importlib.reload(tensor_network_functions_jax)
import tensor_network_functions_jax as TN_fns_jax

import fully_tensorized_model_constructor_functions
importlib.reload(fully_tensorized_model_constructor_functions)
import fully_tensorized_model_constructor_functions as fTNmodel_fns





### ---------------------------------------------------------------------------------------- ###
## ----------------------------------- Experiments specs ------------------------------------ ##
### ---------------------------------------------------------------------------------------- ###

# Folder in which to save results
results_folder = '/work/bd1179/b309245/fourier_models_train_and_FIM/train_and_FIM_fullyTN_Dloc3/'

### 'biased_data_gen': the data generating funct. is encompassed by both full and cutoff models
name_data_gen = 'UNbiased_data_gen'

# No. of random model draws
no_rand_model_tests = 1
# Initial counter for model draws
ini_counter_draws = 6

# No. of training tests per model
no_train_tests = 7

# No. of param. samples for eff. dim.
no_params_samples = 200

# Learning rate
learning_rate = 0.02
learning_rate_name = '0p02'

# Batch size for training
batch_size = 12
batch_size_name = str(batch_size)

# No. of training epochs
no_epochs = 100

# Used for setting the no. of training data
no_training_data_per_feature = 6

# Used for setting the no. of validation data
no_validation_data_per_feature = 6

# Bond dimension
bond_dim = 30
name_bond_dim = str(bond_dim)

### Dimension of correlation space to which input functions are mapped
dim_corr_space = 30
name_corr_space = str(dim_corr_space)

### Cutoff index for correlations among Fourier components:
### the higher the cutoff, the more correlations the cut-off model will exhibit,
### and the higher the effective dimension will be.
### The SVs of the model's structure constants will start decaying after 'cutoff' values, i.e.,
### S_full = S0[0:]  for the full model
### S_cut[0:cutoff] = S0[0:cutoff]; S_cut[cutoff:] = decay_factor * S0[cutoff:]  for the cut model
cutoff = 2
cutoff_name = str(cutoff)

### Decay exponent for cutoff model: 
### S_cut[cutoff+i-1] = np.exp(- decay_exp * i) for i in [1, dim_basis_inputs-cutoff]
decay_exp = 1.0
decay_exp_name = '1p0'





### ---------------------------------------------------------------------------------------- ###
## ------------------------------------ Basic model specs ----------------------------------- ##
### ---------------------------------------------------------------------------------------- ###

### No. of features (dimension of input vectors)
no_of_features = 4
name_no_features = str(no_of_features)

### Maximal Fourier frequency
max_frequency = 3
name_max_freq = str(max_frequency)

### No. of basis states inputs
local_dim_basis_inputs = 2 * max_frequency + 1

### No. of parameters
no_params = 24
name_no_params = str(no_params)

### No. basis states per parameter
dim_basis_single_param = 3  ### (1, cos(th), sin(th))
name_dim_basis_param = str(dim_basis_single_param)

### Bounds for the (uniformly distributed) inputs
input_min = - np.pi
input_max = + np.pi

### Bounds for the (uniformly distributed) parameters
params_min = - np.pi
params_max = + np.pi

# No. of training data
no_training_data = no_training_data_per_feature**no_of_features

# No. of validation data
no_validation_data = no_validation_data_per_feature**no_of_features





### ---------------------------------------------------------------------------------------- ###
## ----------------- Specifications for TN containers for model contruction ----------------- ##
### ---------------------------------------------------------------------------------------- ###

### Max. input dimension for tensor containers in JAX
max_phys_dim = np.max((local_dim_basis_inputs, bond_dim, dim_corr_space))

### No. of layers of isometries in TTN
no_ttn_layers = int(np.log2(no_of_features))

### Dimensions of orthogonal MPO containers
mpo_dims = fTNmodel_fns.dimensions_shallow_orthogonal_MPO(no_of_features, local_dim_basis_inputs)
maxdims_mpo = np.max(mpo_dims, axis=0)

### Dimension of isometric TTN containers
ttn_dims = fTNmodel_fns.dimensions_isometric_TTN_tensors(no_of_features, local_dim_basis_inputs, bond_dim, dim_corr_space)
maxdims_ttn = np.max(ttn_dims, axis=0)

### Dimension of V tensor train containers
dims_Vtensors = TN_fns_jax.dimensions_tensor_train(dim_corr_space, no_params, dim_basis_single_param, bond_dim)
maxdims_Vtensors = np.max(dims_Vtensors, axis=0)
maxdims_Ivecs = np.asarray([1, 1, dim_basis_single_param])

### Local derivative tensor for FIM calculation
B_jnp = TN_fns_jax.derivative_tensor_jax(dim_basis_single_param)





### ---------------------------------------------------------------------------------------- ###
## ------------------------------------ Inputs for training --------------------------------- ##
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

### Decay factors for 'cutoff' model
decay_factors = np.exp(- decay_exp * (np.arange(0, dim_corr_space-cutoff) + 1.0))

####################### Loop over random model draws and random data gen. functions #######################
for nm in range(no_rand_model_tests):
    name_model = str(ini_counter_draws + nm)
    name_end = ('_BondDim' + name_bond_dim + '_Nfeatures' + name_no_features + '_MaxFreq' + name_max_freq + 
                '_Nparams' + name_no_params + '_DimBasisParam' + name_dim_basis_param + '_DimCorrSpace' + name_corr_space + 
                '_' + name_data_gen + '_batch' + batch_size_name + '_lr' + learning_rate_name + '_cutoff' + cutoff_name + 
                '_decayexp' + decay_exp_name + '_modeldraw' + name_model)


    ############ Define random data-generating function (not necessarily encompassed by models) ###########
    params0 = (params_max - params_min) * np.random.rand(no_params) + params_min
    params0_jnp = jnp.asarray(params0)

    ### Generate MPO orthogonal
    oMPO_tensors_data = fTNmodel_fns.generate_random_shallow_orthogonal_MPO_padded(no_of_features, local_dim_basis_inputs, max_phys_dim)
    oMPO_tensors_data_jnp = jnp.asarray(oMPO_tensors_data)
    
    ### Generate TTN isometry
    ttn_tensors_data = fTNmodel_fns.generate_random_isometric_TTN(no_of_features, local_dim_basis_inputs, bond_dim, dim_corr_space)
    ttn_tensors_data_jnp = jnp.asarray(ttn_tensors_data)
    
    ### Compile isometric mapping function
    @jax.jit
    def iso_map_input_data_jaxjit(x):
        return fTNmodel_fns.isometric_mapping_input_vector_jax(x, no_of_features, max_frequency, max_phys_dim, oMPO_tensors_data_jnp, 
                                                               ttn_tensors_data_jnp, maxdims_mpo, no_ttn_layers, dim_corr_space)

    ### Generate set of random right-orthogonal tensors
    random_tensor_train = TN_fns_jax.generate_random_tensor_train_np_padded(maxdims_Vtensors, no_params)
    Vdata_tensors_np = TN_fns_jax.orthogonalize_tensor_train_np_padded(random_tensor_train, dims_Vtensors, maxdims_Vtensors)
    Vdata_tensors_jnp = jnp.asarray(Vdata_tensors_np)
    
    ### Define correlation spectrum for data gen.
    Svals_data = np.ones(dim_corr_space)
    Svals_data_jnp = jnp.asarray(Svals_data)
        

    ############################## Define structure constants for both models #############################

    ### Generate MPO orthogonal
    ortho_mpo_tensors_padded = fTNmodel_fns.generate_random_shallow_orthogonal_MPO_padded(no_of_features, local_dim_basis_inputs, max_phys_dim)
    ortho_mpo_tensors_padded_jnp = jnp.asarray(ortho_mpo_tensors_padded)
    
    ### Generate TTN isometry
    ttn_tensors = fTNmodel_fns.generate_random_isometric_TTN(no_of_features, local_dim_basis_inputs, bond_dim, dim_corr_space)
    ttn_tensors_jnp = jnp.asarray(ttn_tensors)
    
    ### Compile isometric mapping function
    @jax.jit
    def iso_map_input_jaxjit(x):
        return fTNmodel_fns.isometric_mapping_input_vector_jax(x, no_of_features, max_frequency, max_phys_dim, ortho_mpo_tensors_padded_jnp, 
                                                               ttn_tensors_jnp, maxdims_mpo, no_ttn_layers, dim_corr_space)

    ### Generate set of random right-orthogonal tensors
    random_tensor_train = TN_fns_jax.generate_random_tensor_train_np_padded(maxdims_Vtensors, no_params)
    V_tensors_np = TN_fns_jax.orthogonalize_tensor_train_np_padded(random_tensor_train, dims_Vtensors, maxdims_Vtensors)
    V_tensors_jnp = jnp.asarray(V_tensors_np)
    
    ### Define correlation spectrum for full model
    Svals_full = np.ones(dim_corr_space)
    Svals_full_jnp = jnp.asarray(Svals_full)
    
    ### Define correlation spectrum for cut model
    Svals_cut = copy.deepcopy(Svals_full)
    Svals_cut[cutoff:] = decay_factors * Svals_cut[cutoff:]
    Svals_cut_jnp = jnp.asarray(Svals_cut)

        
    ############################# Define full and cut models for JAX backprop #############################
    @jax.jit
    def Fourier_model_data(params, inputs):
        return fTNmodel_fns.fully_tensorized_Fourier_model_constructor_jax(inputs, params, dim_basis_single_param, no_params, dim_corr_space,
                                                                           iso_map_input_data_jaxjit, Svals_data_jnp, Vdata_tensors_jnp, 
                                                                           maxdims_Vtensors, maxdims_Ivecs)
        
    @jax.jit
    def Fourier_model_full(params, inputs):
        return fTNmodel_fns.fully_tensorized_Fourier_model_constructor_jax(inputs, params, dim_basis_single_param, no_params, dim_corr_space,
                                                                           iso_map_input_jaxjit, Svals_full_jnp, V_tensors_jnp, 
                                                                           maxdims_Vtensors, maxdims_Ivecs)
    
    @jax.jit
    def Fourier_model_cut(params, inputs):
        return fTNmodel_fns.fully_tensorized_Fourier_model_constructor_jax(inputs, params, dim_basis_single_param, no_params, dim_corr_space,
                                                                           iso_map_input_jaxjit, Svals_cut_jnp, V_tensors_jnp, 
                                                                           maxdims_Vtensors, maxdims_Ivecs)


    ####################################### Define training outputs #######################################
    no_train_data = train_inputs.shape[0]
    no_val_data = val_inputs.shape[0]
    train_outputs = np.zeros(no_train_data)
    val_outputs = np.zeros(no_val_data)
    no_batches = int(no_train_data / batch_size)
    for j in range(no_batches):
        in_batch = jax.lax.dynamic_slice(jnp_train_inputs, [j*batch_size, 0], [batch_size, no_of_features])
        out_batch = Fourier_model_data(params0, in_batch)
        train_outputs[j*batch_size:(j+1)*batch_size] = np.asarray(out_batch)
    no_batches = int(no_val_data / batch_size)
    for j in range(no_batches):
        in_batch = jax.lax.dynamic_slice(jnp_val_inputs, [j*batch_size, 0], [batch_size, no_of_features])
        out_batch = Fourier_model_data(params0, in_batch)
        val_outputs[j*batch_size:(j+1)*batch_size] = np.asarray(out_batch)
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
        FIM_full = norm_FIM_jit(loc_Ivecs_jnp, Svals_full_jnp)
        FIM_cut = norm_FIM_jit(loc_Ivecs_jnp, Svals_cut_jnp)
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
    list_val_mses_full = []
    list_val_mses_cut = []
    list_delta_mse = []
    list_delta_mse_val = []
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
        val_mse_full = np.squeeze(loss_history_full['val_loss'])
        list_val_mses_full.append(val_mse_full)
        val_mse_cut = np.squeeze(loss_history_cut['val_loss'])
        list_val_mses_cut.append(val_mse_cut)
        min_MSE_full = np.min(train_mse_full)
        min_MSE_cut = np.min(train_mse_cut)
        min_valMSE_full = np.min(val_mse_full)
        min_valMSE_cut = np.min(val_mse_cut)
        delta_minMSE_full_m_cut = min_MSE_full - min_MSE_cut
        list_delta_mse.append(delta_minMSE_full_m_cut)
        delta_minMSEval_full_m_cut = min_valMSE_full - min_valMSE_cut
        list_delta_mse_val.append(delta_minMSEval_full_m_cut)

    train_mses_full = np.asarray(list_train_mses_full)
    train_mses_cut = np.asarray(list_train_mses_cut)
    list_delta_mse = np.asarray(list_delta_mse)
    val_mses_full = np.asarray(list_val_mses_full)
    val_mses_cut = np.asarray(list_val_mses_cut)
    list_delta_mse_val = np.asarray(list_delta_mse_val)

    dict_model = dict()
    dict_model['delta_minMSEval_full_m_cut'] = list_delta_mse_val
    dict_model['MSEval_full'] = val_mses_full
    dict_model['MSEval_cut'] = val_mses_cut
    dict_model['delta_minMSE_full_m_cut'] = list_delta_mse
    dict_model['MSE_full'] = train_mses_full
    dict_model['MSE_cut'] = train_mses_cut
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