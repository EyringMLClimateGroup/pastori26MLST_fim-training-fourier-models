"""
Construction of the plain (non-tensorized) partial Fourier series model
f_theta(x) = sum_mu c_mu(theta) e_mu(x), with c_mu(theta) = sum_nu Gamma_{mu,nu} iota_nu(theta)
(Eqs. (1)-(3) of the paper).

This module builds the harmonic input basis functions e_mu(x) and parameter basis functions
iota_nu(theta) explicitly as dense arrays (frequency, phase, and normalization for every basis
element), and evaluates the model by contracting these bases with a full, densely stored
structure-constants matrix 'coeffs_vectors' (i.e. Gamma, or Gamma's SVD factors reconstructed as
one matrix). It is meant for small problem sizes -- since the basis dimensions D = d**N (inputs)
and K = d_tilde**M (parameters) grow exponentially with the number of features N and parameters
M -- and is the reference/exact implementation against which the tensor-network model
constructors (see tensorized_model_constructor_functions.py and
fully_tensorized_model_constructor_functions.py) are checked.
"""

import numpy as np
from jax import numpy as jnp
import copy





###
### Generate the basis functions for input dependence. The outputs are the arrays:
### all_freqs:  (dim_basis_inputs, no_of_features)
### all_offsets:  (dim_basis_inputs, no_of_features)
### all_norms:  (dim_basis_inputs, no_of_features)
### which are combined to form the basis functions as follows:
### np.prod(all_norms[j,:] * np.cos(all_freqs[j,:] * x + all_offsets[j,:])) = e_j(x)
###
### For simplicity, the basis functions are taken to be harmonic functions
### normalized in the interval [-pi, pi].
###
def input_basis_functions(max_frequency, no_of_features):
    """
    Build the dense input basis functions e_mu(x) as a tensor product, over the
    'no_of_features' input components, of the local per-feature basis
    {1, sqrt(2)*cos(w*x), ..., sqrt(2)*cos((max_frequency-1)*x), sqrt(2)*sin(w*x), ...},
    i.e. the harmonic basis with frequencies w = 0, ..., max_frequency-1 (MaxFreq),
    orthonormalized on [-pi, pi]. The total number of basis functions is
    D = (2*max_frequency - 1)**no_of_features.

    Each basis function e_mu(x) is recovered from the returned arrays as
        e_mu(x) = prod_n( all_norms[mu, n] * cos(all_freqs[mu, n] * x_n + all_offsets[mu, n]) ).

    Parameters
    ----------
    max_frequency : int
        Number of distinct (non-negative) frequencies per feature, i.e. the maximum Fourier
        frequency accessible to the model is max_frequency - 1 (MaxFreq).
    no_of_features : int
        Number of input features N.

    Returns
    -------
    all_freqs : ndarray, shape (D, no_of_features)
        Frequency of each basis function for each feature.
    all_offsets : ndarray, shape (D, no_of_features)
        Phase offset of each basis function for each feature (0 for cosines, -pi/2 for sines).
    all_norms : ndarray, shape (D, no_of_features)
        Normalization factor of each basis function for each feature.
    """
    ### Define local basis (local meaning related to one feature in this case)
    all_freqs_cos = np.arange(0, max_frequency)
    all_offsets_cos = np.zeros(all_freqs_cos.shape)
    all_norms_cos = np.sqrt(2.0) * np.ones(all_freqs_cos.shape)
    all_norms_cos[0] = all_norms_cos[0] / np.sqrt(2.0)
    all_freqs_sin = copy.deepcopy(all_freqs_cos[1:])
    all_offsets_sin = - np.pi * np.ones(all_freqs_sin.shape) / 2.0
    all_norms_sin = np.sqrt(2.0) * np.ones(all_freqs_sin.shape)
    all_freqs_1feat = np.expand_dims(np.concatenate((all_freqs_cos, all_freqs_sin)), axis=1)
    all_offsets_1feat = np.expand_dims(np.concatenate((all_offsets_cos, all_offsets_sin)), axis=1)
    all_norms_1feat = np.expand_dims(np.concatenate((all_norms_cos, all_norms_sin)), axis=1)
    mask_1feat = np.ones(all_freqs_1feat.shape)

    ### Tensor the local basis for the required number of input features
    ###
    ### np.prod(all_norms[j,:] * np.cos(all_freqs[j,:] * x + all_offsets[j,:])) = e_j(x)
    ### all_freqs:  (dim_basis_inputs, no_of_features)
    ### all_offsets:  (dim_basis_inputs, no_of_features)
    ### all_norms:  (dim_basis_inputs, no_of_features)
    all_freqs = all_freqs_1feat
    all_offsets = all_offsets_1feat
    all_norms = all_norms_1feat
    if no_of_features>1:
        all_freqs_feats = []
        all_offsets_feats = []
        all_norms_feats = []
        #j = 0
        all_freqs_feat_j = copy.deepcopy(all_freqs_1feat)
        all_offsets_feat_j = copy.deepcopy(all_offsets_1feat)
        all_norms_feat_j = copy.deepcopy(all_norms_1feat)
        for k in range(1,no_of_features):
            all_freqs_feat_j = np.kron(all_freqs_feat_j, mask_1feat)
            all_offsets_feat_j = np.kron(all_offsets_feat_j, mask_1feat)
            all_norms_feat_j = np.kron(all_norms_feat_j, mask_1feat)
        all_freqs_feats.append(all_freqs_feat_j)
        all_offsets_feats.append(all_offsets_feat_j)
        all_norms_feats.append(all_norms_feat_j)
        #j > 0
        for j in range(1,no_of_features):
            all_freqs_feat_j = copy.deepcopy(mask_1feat)
            all_offsets_feat_j = copy.deepcopy(mask_1feat)
            all_norms_feat_j = copy.deepcopy(mask_1feat)
            for k in range(1,no_of_features):
                if k==j:
                    all_freqs_feat_j = np.kron(all_freqs_feat_j, all_freqs_1feat)
                    all_offsets_feat_j = np.kron(all_offsets_feat_j, all_offsets_1feat)
                    all_norms_feat_j = np.kron(all_norms_feat_j, all_norms_1feat)
                else:
                    all_freqs_feat_j = np.kron(all_freqs_feat_j, mask_1feat)
                    all_offsets_feat_j = np.kron(all_offsets_feat_j, mask_1feat)
                    all_norms_feat_j = np.kron(all_norms_feat_j, mask_1feat)
            all_freqs_feats.append(all_freqs_feat_j)
            all_offsets_feats.append(all_offsets_feat_j)
            all_norms_feats.append(all_norms_feat_j)
        all_freqs = np.hstack(all_freqs_feats)
        all_offsets = np.hstack(all_offsets_feats)
        all_norms = np.hstack(all_norms_feats)
    return all_freqs, all_offsets, all_norms



###
### Generate the basis functions for parameters dependence. The outputs are the arrays:
### all_freqs_p:  (dim_basis_params, no_params)
### all_offsets_p:  (dim_basis_params, no_params)
### all_norms_p:  (dim_basis_params, no_params)
### which are combined to form the basis functions as follows:
### np.prod(all_norms_p[j,:] * np.cos(all_freqs_p[j,:] * params + all_offsets_p[j,:])) = i_j(th1,...,thM)
###
### For simplicity, the basis functions are taken to be harmonic functions
### normalized in the interval [-pi, pi].
###
def params_basis_functions(local_dim_param, no_params):
    """
    Build the dense parameter-space basis functions iota_nu(theta) as a tensor product, over the
    'no_params' trainable parameters, of the local per-parameter harmonic basis
    {1, sqrt(2)*cos(theta), ..., sqrt(2)*cos(L*theta), sqrt(2)*sin(theta), ...} with
    L = (local_dim_param - 1) / 2, orthonormalized on [-pi, pi]. The total number of parameter
    basis functions is K = local_dim_param**no_params.

    Each basis function iota_nu(theta) is recovered from the returned arrays as
        iota_nu(theta) = prod_m( all_norms_p[nu, m] * cos(all_freqs_p[nu, m] * theta_m + all_offsets_p[nu, m]) ).

    Parameters
    ----------
    local_dim_param : int
        Local per-parameter basis dimension d_tilde (must be odd, 2L+1).
    no_params : int
        Number of trainable parameters M.

    Returns
    -------
    all_freqs_p : ndarray, shape (K, no_params)
        Frequency of each basis function for each parameter.
    all_offsets_p : ndarray, shape (K, no_params)
        Phase offset of each basis function for each parameter.
    all_norms_p : ndarray, shape (K, no_params)
        Normalization factor of each basis function for each parameter.
    """
    ### Define local basis (local meaning related to one parameter in this case)
    max_freq_params = int((local_dim_param - 1) / 2)
    all_freqs_cos_p = np.arange(0, max_freq_params + 1)
    all_offsets_cos_p = np.zeros(all_freqs_cos_p.shape)
    all_norms_cos_p = np.sqrt(2.0) * np.ones(all_freqs_cos_p.shape)
    all_norms_cos_p[0] = all_norms_cos_p[0] / np.sqrt(2.0)
    all_freqs_sin_p = copy.deepcopy(all_freqs_cos_p[1:])
    all_offsets_sin_p = - np.pi * np.ones(all_freqs_sin_p.shape) / 2.0
    all_norms_sin_p = np.sqrt(2.0) * np.ones(all_freqs_sin_p.shape)
    all_freqs_1p = np.expand_dims(np.concatenate((all_freqs_cos_p, all_freqs_sin_p)), axis=1)
    all_offsets_1p = np.expand_dims(np.concatenate((all_offsets_cos_p, all_offsets_sin_p)), axis=1)
    all_norms_1p = np.expand_dims(np.concatenate((all_norms_cos_p, all_norms_sin_p)), axis=1)
    mask_1p = np.ones(all_freqs_1p.shape)

    ### Tensor the local basis for the required number of parameters
    ###
    ### np.prod(all_norms[j,:] * np.cos(all_freqs[j,:] * params + all_offsets[j,:])) = i_j(th1,...,thM)
    ### all_freqs_p:  (dim_basis_params, no_params)
    ### all_offsets_p:  (dim_basis_params, no_params)
    ### all_norms_p:  (dim_basis_params, no_params)
    all_freqs_p = all_freqs_1p
    all_offsets_p = all_offsets_1p
    all_norms_p = all_norms_1p
    if no_params>1:
        all_freqs_pars = []
        all_offsets_pars = []
        all_norms_pars = []
        #j = 0
        all_freqs_feat_j = copy.deepcopy(all_freqs_1p)
        all_offsets_feat_j = copy.deepcopy(all_offsets_1p)
        all_norms_feat_j = copy.deepcopy(all_norms_1p)
        for k in range(1,no_params):
            all_freqs_feat_j = np.kron(all_freqs_feat_j, mask_1p)
            all_offsets_feat_j = np.kron(all_offsets_feat_j, mask_1p)
            all_norms_feat_j = np.kron(all_norms_feat_j, mask_1p)
        all_freqs_pars.append(all_freqs_feat_j)
        all_offsets_pars.append(all_offsets_feat_j)
        all_norms_pars.append(all_norms_feat_j)
        #j > 0
        for j in range(1,no_params):
            all_freqs_feat_j = copy.deepcopy(mask_1p)
            all_offsets_feat_j = copy.deepcopy(mask_1p)
            all_norms_feat_j = copy.deepcopy(mask_1p)
            for k in range(1,no_params):
                if k==j:
                    all_freqs_feat_j = np.kron(all_freqs_feat_j, all_freqs_1p)
                    all_offsets_feat_j = np.kron(all_offsets_feat_j, all_offsets_1p)
                    all_norms_feat_j = np.kron(all_norms_feat_j, all_norms_1p)
                else:
                    all_freqs_feat_j = np.kron(all_freqs_feat_j, mask_1p)
                    all_offsets_feat_j = np.kron(all_offsets_feat_j, mask_1p)
                    all_norms_feat_j = np.kron(all_norms_feat_j, mask_1p)
            all_freqs_pars.append(all_freqs_feat_j)
            all_offsets_pars.append(all_offsets_feat_j)
            all_norms_pars.append(all_norms_feat_j)
        all_freqs_p = np.hstack(all_freqs_pars)
        all_offsets_p = np.hstack(all_offsets_pars)
        all_norms_p = np.hstack(all_norms_pars)
    return all_freqs_p, all_offsets_p, all_norms_p



###
### Define the (Fourier) model based on the structure constants 'coeffs_vectors'
### and the supplied input and parameters basis functions.
###
### This function can be wrapped anf JIT compiled in the main as follows:
###
### jnp_all_freqs = jnp.asarray(all_freqs)
### jnp_all_offsets = jnp.asarray(all_offsets)
### jnp_all_norms = jnp.asarray(all_norms)
### jnp_all_freqs_p = jnp.asarray(all_freqs_p)
### jnp_all_offsets_p = jnp.asarray(all_offsets_p)
### jnp_all_norms_p = jnp.asarray(all_norms_p)
###
### @jax.jit
### def Fourier_model(params, inputs):
###     return Fourier_model_constructor_jax(inputs, params, coeffs_vectors
###                                          jnp_all_freqs, jnp_all_offsets, jnp_all_norms,
###                                          jnp_all_freqs_p, jnp_all_offsets_p, jnp_all_norms_p)
### 
def Fourier_model_constructor_jax(inputs, params, coeffs_vectors,
                                  jnp_all_freqs, jnp_all_offsets, jnp_all_norms,
                                  jnp_all_freqs_p, jnp_all_offsets_p, jnp_all_norms_p):
    """
    Evaluate the partial Fourier series model
        f_theta(x) = sum_mu c_mu(theta) e_mu(x),  c_mu(theta) = sum_nu Gamma_{mu,nu} iota_nu(theta)
    (Eqs. (1)-(3) of the paper) on a batch of inputs, given the dense structure-constants matrix
    Gamma ('coeffs_vectors') and the input/parameter basis functions produced by
    input_basis_functions and params_basis_functions.

    Parameters
    ----------
    inputs : ndarray, shape (batch_size, no_of_features)
        Batch of input points x.
    params : ndarray, shape (no_params,)
        Model parameters theta.
    coeffs_vectors : ndarray, shape (dim_basis_inputs, dim_basis_params)
        Structure-constants matrix Gamma (D x K).
    jnp_all_freqs, jnp_all_offsets, jnp_all_norms : ndarray, shape (dim_basis_inputs, no_of_features)
        Input basis functions e_mu(x), as returned by input_basis_functions.
    jnp_all_freqs_p, jnp_all_offsets_p, jnp_all_norms_p : ndarray, shape (dim_basis_params, no_params)
        Parameter basis functions iota_nu(theta), as returned by params_basis_functions.

    Returns
    -------
    outputs : ndarray, shape (batch_size,)
        Model predictions f_theta(x) for each input in the batch.
    """
    ### inputs:  (batch_size, no_of_features)
    ### params:  (no_params, )
    ### coeffs_vectors:  (dim_basis_inputs, dim_basis_params)

    ### Define basis functions for parameter space
    basis_funcs_th = jnp.prod(jnp_all_norms_p * jnp.cos(jnp_all_freqs_p * params + jnp_all_offsets_p), axis=1)  ### (dim_basis_params, )
    basis_funcs_th = jnp.expand_dims(basis_funcs_th, axis=1)  ### (dim_basis_params, 1)

    ### Calculate Fourier coeffs.
    coeffs = jnp.matmul(coeffs_vectors, basis_funcs_th)  ### (dim_basis_inputs, 1)

    ### Calculate input basis functions for all inputs
    batch_size = inputs.shape[0]
    all_Ws = jnp.tile(jnp_all_freqs, (batch_size, 1, 1))  ### (batch_size, dim_basis_inputs, no_of_features)
    all_Bs = jnp.tile(jnp_all_offsets, (batch_size, 1, 1))  ### (batch_size, dim_basis_inputs, no_of_features)
    all_Ns = jnp.tile(jnp_all_norms, (batch_size, 1, 1))  ### (batch_size, dim_basis_inputs, no_of_features)
    all_Ws = jnp.transpose(all_Ws, (1, 0, 2))  ### (dim_basis_inputs, batch_size, no_of_features)
    all_Bs = jnp.transpose(all_Bs, (1, 0, 2))  ### (dim_basis_inputs, batch_size, no_of_features)
    all_Ns = jnp.transpose(all_Ns, (1, 0, 2))  ### (dim_basis_inputs, batch_size, no_of_features)
    basis_funcs_ins = jnp.prod(all_Ns * jnp.cos(all_Ws * inputs + all_Bs), axis=2)  ### (dim_basis_inputs, batch_size)
    basis_funcs_ins = jnp.transpose(basis_funcs_ins)  ### (batch_size, dim_basis_inputs)

    ### Outputs
    outputs = jnp.squeeze(jnp.matmul(basis_funcs_ins, coeffs))  ### (batch_size, )

    return outputs