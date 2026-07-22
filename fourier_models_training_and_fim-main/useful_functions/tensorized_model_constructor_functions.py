"""
Construction of a partially tensorized Fourier model, in which the parameter-to-correlation-space
map V (from the SVD Gamma = U @ diag(S) @ V^T, Eq. (22)) is represented as a tensor train (matrix
product state) of bond dimension chi over the M parameter legs, while the input side (U and the
input basis functions e_mu(x)) is kept dense/explicit. This lets the number of trainable
parameters M grow while avoiding the exponential storage cost K = d_tilde**M of the parameter
basis, at the price of an approximation controlled by the bond dimension chi.

Besides evaluating the model, this module provides the routines used to control the model's
*bias* towards a given target parameter configuration theta*: 'generate_V_orthogonal_to_params'
builds a biased model whose V-tensor-train is exactly orthogonal (up to a cutoff) to
theta*, and 'perturb_V_tensors' interpolates away from that bias by adding Gaussian noise of
strength epsilon (delta_data in the paper) and re-orthogonalizing, producing the partially
biased models used in the bias-scan training experiments (Fig. 5 of the paper).
"""

import numpy as np
import jax
from jax import numpy as jnp
import copy
import importlib

import ortho_matrices_functions
importlib.reload(ortho_matrices_functions)
import ortho_matrices_functions as ortho_fns

import tensor_network_functions_jax
importlib.reload(tensor_network_functions_jax)
import tensor_network_functions_jax as TN_fns_jax






###
### Define the (Fourier) model based on the tensorized structure constants defined by
### 'U_jnp', 'Svals_jnp', 'V_tensors_jnp'. In this case, only 'V_tensors_jnp' is tensorized.
###
### This function can be wrapped anf JIT compiled in the main as follows:
###
### local_dim_param = ...
### no_params = ...
### dim_basis_inputs = ...
### U_jnp = ...
### Svals_jnp = ...
### V_tensors_jnp = ...
### maxdims_Vtens = ...
### maxdims_Itens = ...
### jnp_all_freqs = jnp.asarray(all_freqs)
### jnp_all_offsets = jnp.asarray(all_offsets)
### jnp_all_norms = jnp.asarray(all_norms)
###
### @jax.jit
### def Fourier_model(params, inputs):
###     return tensorized_Fourier_model_constructor_jax(inputs, params, local_dim_param, no_params, dim_basis_inputs,
###                                                     U_jnp, Svals_jnp, V_tensors_jnp, maxdims_Vtens, maxdims_Itens,
###                                                     jnp_all_freqs, jnp_all_offsets, jnp_all_norms)
### 
def tensorized_Fourier_model_constructor_jax(inputs, params_jnp, local_dim_param, no_params, dim_basis_inputs,
                                             U_jnp, Svals_jnp, V_tensors_jnp, maxdims_Vtens, maxdims_Itens,
                                             jnp_all_freqs, jnp_all_offsets, jnp_all_norms):
    """
    Evaluate the Fourier model f_theta(x) = e(x)^T U diag(S) V^T iota(theta) (Eq. (22) of the
    paper) where the parameter basis contraction V^T iota(theta) is performed as a tensor-train
    contraction (bond dimension chi, tensors 'V_tensors_jnp') instead of building the full
    K = local_dim_param**no_params parameter basis explicitly. U (the input-side orthogonal
    matrix) and the input basis functions e_mu(x) are still handled densely.

    Parameters
    ----------
    inputs : ndarray, shape (batch_size, no_of_features)
        Batch of input points x.
    params_jnp : ndarray, shape (no_params,)
        Model parameters theta.
    local_dim_param : int
        Local per-parameter basis dimension d_tilde.
    no_params : int
        Number of trainable parameters M.
    dim_basis_inputs : int
        Dimension D of the input basis / correlation space.
    U_jnp : ndarray, shape (dim_basis_inputs, dim_basis_inputs)
        Left-singular-vector (orthogonal) matrix U of Gamma.
    Svals_jnp : ndarray, shape (dim_basis_inputs,)
        Correlation spectrum (singular values of Gamma).
    V_tensors_jnp : ndarray
        Tensor-train (bond dimension chi) representing V^T, one tensor per parameter leg.
    maxdims_Vtens, maxdims_Itens : ndarray
        Padded auxiliary/physical leg dimensions of the V tensor train and of the local
        parameter-basis vectors, used for JAX-compatible fixed-shape contraction.
    jnp_all_freqs, jnp_all_offsets, jnp_all_norms : ndarray, shape (dim_basis_inputs, no_of_features)
        Input basis functions e_mu(x), as returned by input_basis_functions.

    Returns
    -------
    outputs : ndarray, shape (batch_size,)
        Model predictions f_theta(x) for each input in the batch.
    """
    ### inputs:  (batch_size, no_of_features)
    ### params:  (no_params, )

    ### Construct local tensors for local parameter basis @params
    loc_Ivecs_jnp = TN_fns_jax.local_basis_vectors_jaxjit(params_jnp, local_dim_param)

    ### Compute VI = (V^T) * Ivec,  (dim_basis_inputs, 1)
    VI = TN_fns_jax.contract_tensortrain_with_localvectors_jaxjit(V_tensors_jnp, loc_Ivecs_jnp, no_params, 
                                                                  maxdims_Vtens, maxdims_Itens)
    VI = jax.lax.dynamic_slice(VI, [0, 0], [dim_basis_inputs, 1])

    ### Compute SVI = S * (V^T) * Ivec,  (dim_basis_inputs, 1)
    SVI = VI * jnp.expand_dims(Svals_jnp, axis=1)

    ### Compute Fourier coeffs:  coeffs = U * S * (V^T) * Ivec,  (dim_basis_inputs, 1)
    coeffs = jnp.matmul(U_jnp, SVI)

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





###
### Construct model's V matrix partially orthogonalized w.r.t. a set of parameters.
### Used for constructing biased models. 
###
def generate_V_orthogonal_to_params(params0_jnp, cutoff, dim_in, no_params, local_dim_param,
                                    dims_Vtensors, maxdims_Vtensors, maxdims_Ivecs):
    """
    Construct a random tensor-train V (right-orthogonal, bond dimension chi) whose first-site
    tensor is built so that 'cutoff' of its dim_in output columns are exactly orthogonal to the
    parameter basis vector iota(theta0) evaluated at a chosen reference configuration theta0
    ('params0_jnp'), while the remaining columns are unconstrained.

    This produces the *biased* Fourier models used in the paper's training experiments: taking
    Svals to be supported only on the 'cutoff' orthogonal directions makes the model's
    correlation spectrum exactly reproduce a target function evaluated at theta0 (the model is
    then guaranteed to contain the data-generating function y(x) in its expressible space,
    Sec. on biased vs. unbiased models).

    Parameters
    ----------
    params0_jnp : ndarray, shape (no_params,)
        Reference parameter configuration theta0 (e.g. the target/data-generating parameters)
        that the model is biased towards.
    cutoff : int
        Number of columns of the constructed V-matrix kept orthogonal to iota(theta0); together
        with dim_in - cutoff unconstrained columns.
    dim_in : int
        Dimension D of the input/correlation space (output dimension of the first tensor-train site).
    no_params : int
        Number of trainable parameters M.
    local_dim_param : int
        Local per-parameter basis dimension d_tilde.
    dims_Vtensors, maxdims_Vtensors : ndarray
        Actual and padded (auxiliary, physical) leg dimensions of the V tensor train.
    maxdims_Ivecs : ndarray
        Padded dimensions of the local parameter-basis vectors.

    Returns
    -------
    Vtensors_jnp : ndarray
        Tensor train representing the (partially bias-constructed) V matrix.
    """
    ### Construct local tensors for local parameter basis @params0
    Ivecs0_jnp = TN_fns_jax.local_basis_vectors_jaxjit(params0_jnp, local_dim_param)
    
    ### Generate set of random right-orthogonal tensors
    random_tensor_train = TN_fns_jax.generate_random_tensor_train_np_padded(maxdims_Vtensors, no_params)
    Vtensors_np = TN_fns_jax.orthogonalize_tensor_train_np_padded(random_tensor_train, dims_Vtensors, maxdims_Vtensors)
    Vtensors_jnp = jnp.asarray(Vtensors_np)
    
    ### Extract local basis vectors and tensors for all but first site
    Ivecs0_2toM = jax.lax.dynamic_slice(Ivecs0_jnp, [1, 0, 0, 0], [no_params-1, 1, 1, local_dim_param])
    Vtensors_2toM = jax.lax.dynamic_slice(Vtensors_jnp, [1, 0, 0, 0], [no_params-1, maxdims_Vtensors[0], maxdims_Vtensors[1], local_dim_param])
    
    ### Extract local basis vector for first site
    Ivec0_1 = jax.lax.dynamic_slice(Ivecs0_jnp, [0, 0, 0, 0], [1, 1, 1, local_dim_param])  # (1, 1, 1, local_dim_param)
    Ivec0_1 = jnp.expand_dims(jnp.squeeze(Ivec0_1), axis=1)  # (local_dim_param, 1)
    
    ### Compute A_2toM = contraction(Vtensors_2toM, Ivecs0_2toM) vector
    dim_right_1 = dims_Vtensors[0, 1]
    A_2toM = TN_fns_jax.contract_tensortrain_with_localvectors_jaxjit(Vtensors_2toM, Ivecs0_2toM, no_params-1, maxdims_Vtensors, maxdims_Ivecs)
    A_2toM = jax.lax.dynamic_slice(A_2toM, [0, 0], [dim_right_1, 1])  # (dim_right_1, 1)
    
    ### Compute AI_2toM = tensor_product(A_2toM, Ivecs0_2toM) vector:  (dim_right_1 * local_dim_param, 1)
    AI_2toM = jnp.kron(A_2toM, Ivec0_1)
    
    ### Generate orthogonal matrix matrix whose columns are also orthogonal to AI_2toM
    Wtens1_mat = ortho_fns.gram_schmidt_ortho(dim_in - cutoff, dim_right_1*local_dim_param, ortho_vecs=np.asarray(AI_2toM))  # (dim_right_1 * local_dim_param, dim_in - cutoff)
    
    ### Generate orthogonal matrix matrix whose columns are orthogonal to Wtens1_mat
    Qtens1_mat = ortho_fns.gram_schmidt_ortho(cutoff, dim_right_1*local_dim_param, ortho_vecs=Wtens1_mat)  # (dim_right_1 * local_dim_param, cutoff)
    
    ### Stack matrices horizontally
    Vtens1_mat = np.hstack((Qtens1_mat, Wtens1_mat))  # (dim_right_1 * local_dim_param, dim_in)
    Vtens1_mat = np.transpose(Vtens1_mat)  # (dim_in, dim_right_1 * local_dim_param)
    
    ### Reshape matrix to tensor
    Vtens1 = np.reshape(Vtens1_mat, (dim_in, dim_right_1, local_dim_param))
    Vtens1_jnp = jnp.asarray(Vtens1)
    
    ### Store in Vtensors_jnp
    Vtensors_jnp = Vtensors_jnp.at[0, 0:dim_in, 0:dim_right_1, :].set(Vtens1_jnp)
    
    return Vtensors_jnp






###
### Perurb model's V tensors with Gaussian noise with strength eps.
### Used for constructing partially biased models. 
###
def perturb_V_tensors(eps, V_tensors_jnp, no_params, local_dim_param, dims_Vtensors):
    """
    Perturb every tensor of a right-orthogonal V tensor train with i.i.d. Gaussian noise of
    strength 'eps' and re-orthogonalize (site by site, from the last to the first leg), while
    preserving right-orthogonality.

    Used to interpolate a fully biased model (built with generate_V_orthogonal_to_params,
    eps = 0) continuously towards an unbiased one as eps grows: 'eps' plays the role of the
    partial-bias parameter delta_data used to scan the bias/effective-dimension interplay in the
    training experiments (Fig. 5 of the paper).

    Parameters
    ----------
    eps : float
        Standard deviation of the Gaussian perturbation added to each tensor-train entry
        (delta_data in the paper; eps=0 leaves the model exactly biased).
    V_tensors_jnp : ndarray
        Tensor train representing V^T (bond dimension chi) to be perturbed.
    no_params : int
        Number of trainable parameters M (number of tensor-train sites).
    local_dim_param : int
        Local per-parameter basis dimension d_tilde.
    dims_Vtensors : ndarray
        Actual (unpadded) auxiliary/physical leg dimensions of each tensor-train site.

    Returns
    -------
    pert_V_tensors_jnp : ndarray
        Perturbed and re-orthogonalized tensor train.
    """
    ### Loop over physical legs, perturb and orthogonalize
    pert_V_tensors_jnp = copy.deepcopy(V_tensors_jnp)
    for m in range(0, no_params):
        i = no_params - (m + 1)
        dim_left = dims_Vtensors[i, 0]
        dim_right = dims_Vtensors[i, 1]
        Vi = jnp.squeeze(jax.lax.dynamic_slice(pert_V_tensors_jnp, [i, 0, 0, 0], [1, dim_left, dim_right, local_dim_param]))
        Vi_np = np.asarray(Vi)  # (dim_left, dim_right, local_dim_param)
        matVi_np = np.reshape(Vi_np, (dim_left, dim_right*local_dim_param))  # (dim_left, dim_right*local_dim_param)
        Ge = eps * np.random.randn(dim_left, dim_right*local_dim_param)  # (dim_left, dim_right*local_dim_param)
        matWe_i_np = matVi_np + Ge  # (dim_left, dim_right*local_dim_param)
        matWe_i_np = np.transpose(matWe_i_np)  # (dim_right*local_dim_param, dim_left)
        matVe_i_np = ortho_fns.seeded_gram_schmidt_ortho(dim_left, dim_right*local_dim_param, matWe_i_np)  # (dim_right*local_dim_param, dim_left)
        matVe_i_np = np.transpose(matVe_i_np)  # (dim_left, dim_right*local_dim_param)
        Ve_i_np = np.reshape(matVe_i_np, (dim_left, dim_right, local_dim_param))
        Ve_i = jnp.asarray(Ve_i_np)
        ### Store in V_tensors_jnp
        pert_V_tensors_jnp = pert_V_tensors_jnp.at[i, 0:dim_left, 0:dim_right, :].set(Ve_i)
    return pert_V_tensors_jnp