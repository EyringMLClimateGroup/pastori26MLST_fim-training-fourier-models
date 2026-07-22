"""
JAX/JIT-compiled tensor-train (matrix product state) utilities used to represent the
parameter-side map V^T (from the SVD Gamma = U @ diag(S) @ V^T, Eq. (22) of the paper) as a
tensor network of bond dimension chi across the M parameter legs, together with the routines to
build, orthogonalize and contract such tensor trains, and to compute the Fisher Information
Matrix (FIM) directly from this representation. Local per-parameter tensors have physical leg
dimension d_tilde (local_basis_dim); all arrays are padded to fixed maximal auxiliary/physical
dimensions so they can be used inside jax.lax.fori_loop / jax.jit with static shapes. This is the
JAX counterpart of tensor_network_functions_np.py, used in all the tensorized ('TN') model
constructors and scaling experiments.
"""

import numpy as np
import jax
from jax import numpy as jnp




###
### Computes the dimensions of tensors in a tensor train.
###
def dimensions_tensor_train(dim_in, no_legs, local_basis_dim, bond_dim):
    """
    Compute the (dim_left, dim_right, local_basis_dim) leg dimensions of each site of a tensor
    train with 'no_legs' sites and requested bond dimension 'bond_dim' (chi), representing a
    matrix of input dimension dim_in and output dimension local_basis_dim**no_legs. Internal
    bond dimensions grow as local_basis_dim**m from both ends and are capped at 'bond_dim',
    matching the maximal bond dimension chi used to control the tensor-network approximation of
    Gamma's parameter-side factor V^T (Eq. (22) of the paper).

    Parameters
    ----------
    dim_in : int
        Left (input-space) dimension of the represented matrix.
    no_legs : int
        Number of tensor-train sites (e.g. number of parameters M).
    local_basis_dim : int
        Local physical leg dimension per site (d_tilde).
    bond_dim : int
        Requested (maximal) bond dimension chi.

    Returns
    -------
    tensor_train_dims : ndarray, shape (no_legs, 3)
        Rows are (dim_left, dim_right, local_basis_dim) for each site.
    """
    # Check for which nn local_basis_dim**nn > bond_dim
    nn = 0
    while (local_basis_dim**nn<bond_dim and nn<no_legs):
        nn = nn + 1
    if nn==no_legs:
        bond_dim = local_basis_dim ** no_legs
    dim_left = dim_in
    dim_right = bond_dim
    tensor_train_dims = np.zeros((no_legs, 3), dtype=np.int32)
    for m in range(no_legs - nn):
        min_dim = np.min((dim_left*local_basis_dim, dim_right))
        if min_dim>bond_dim:
            min_dim = bond_dim
        tensor_dims = np.zeros(3, dtype=np.int32)
        tensor_dims[0] = dim_left
        tensor_dims[1] = min_dim
        tensor_dims[2] = local_basis_dim
        tensor_train_dims[m, :] = tensor_dims
        dim_left = min_dim
    for m in range(nn):
        dim_right = local_basis_dim ** (nn - (m + 1))
        min_dim = np.min((dim_left*local_basis_dim, dim_right))
        if min_dim>bond_dim:
            min_dim = bond_dim
        tensor_dims = np.zeros(3, dtype=np.int32)
        tensor_dims[0] = dim_left
        tensor_dims[1] = min_dim
        tensor_dims[2] = local_basis_dim
        tensor_train_dims[no_legs - nn + m, :] = tensor_dims
        dim_left = min_dim
    return tensor_train_dims



###
### Create random tensor train as numpy array with dimensions
###
### (no_legs, 
###  np.max(dimensions_tensor_train(...), axis=0)[0]
###  np.max(dimensions_tensor_train(...), axis=0)[1]
###  np.max(dimensions_tensor_train(...), axis=0)[2]).
###
def generate_random_tensor_train_np_padded(maxdims_tensortrain, no_legs):
    """
    Generate a tensor train of 'no_legs' sites with i.i.d. standard-normal random entries,
    padded to uniform (dim_t_0, dim_t_1, dim_t_2) leg dimensions (the maxima of
    dimensions_tensor_train across all sites) for compatibility with JAX's fixed-shape
    contraction routines. Serves as the unnormalized seed that
    orthogonalize_tensor_train_np_padded turns into a right-orthogonal tensor train
    representing a random V^T of bond dimension chi.

    Parameters
    ----------
    maxdims_tensortrain : array-like, length 3
        Maximal (dim_left, dim_right, local_basis_dim) leg dimensions, e.g. from
        np.max(dimensions_tensor_train(...), axis=0).
    no_legs : int
        Number of tensor-train sites.

    Returns
    -------
    random_tensor_train : ndarray, shape (no_legs, dim_t_0, dim_t_1, dim_t_2)
    """
    dim_t_0 = maxdims_tensortrain[0]
    dim_t_1 = maxdims_tensortrain[1]
    dim_t_2 = maxdims_tensortrain[2]
    random_tensor_train = np.zeros((no_legs, dim_t_0, dim_t_1, dim_t_2))
    for m in range(no_legs):
        Am = np.random.randn(dim_t_0, dim_t_1, dim_t_2)
        random_tensor_train[m, :, :, :] = Am
    return random_tensor_train



###
### Right-orthogonalization of tensor train (np).
### Output is a list of tensors, that is a numpy array with dimensions
###
### (no_legs, 
###  np.max(dimensions_tensor_train(...), axis=0)[0]
###  np.max(dimensions_tensor_train(...), axis=0)[1]
###  np.max(dimensions_tensor_train(...), axis=0)[2]).
###
### The actual dimensions of the right-orthogonal tensors are specified by 'dims_tensortrain'.
### 
def orthogonalize_tensor_train_np_padded(random_tensor_train, dims_tensortrain, maxdims_tensortrain):
    """
    Right-orthogonalize a tensor train (e.g. from generate_random_tensor_train_np_padded) via a
    sweep of SVDs from the last site to the first, so that each site tensor Vi is right-isometric
    (sum over right-auxiliary and physical legs of Vi Vi^dagger = identity). This produces a
    valid tensor-train representation of the parameter-side map V^T with bond dimension bounded
    by 'bond_dim' (encoded in maxdims_tensortrain), as required for V to be part of an orthogonal
    (or isometric) SVD factor of Gamma.

    Parameters
    ----------
    random_tensor_train : ndarray
        Unnormalized (random) tensor train, e.g. from generate_random_tensor_train_np_padded.
    dims_tensortrain : ndarray, shape (no_legs, 3)
        Actual (unpadded) (dim_left, dim_right, local_basis_dim) dimensions per site, e.g. from
        dimensions_tensor_train.
    maxdims_tensortrain : array-like, length 3
        Padded maximal leg dimensions shared by all sites.

    Returns
    -------
    ortho_tensor_train : ndarray
        Right-orthogonalized tensor train, same padded shape as the input.
    """
    no_legs = random_tensor_train.shape[0]
    local_basis_dim = random_tensor_train.shape[3]
    dim_t_0 = maxdims_tensortrain[0]
    dim_t_1 = maxdims_tensortrain[1]
    dim_t_2 = maxdims_tensortrain[2]
    ortho_tensor_train = np.zeros((no_legs, dim_t_0, dim_t_1, dim_t_2))
    Uip1 = np.ones((1, 1))
    for m in range(0, no_legs):
        i = no_legs - (m + 1)
        dim_left = dims_tensortrain[i, 0]
        dim_right = dims_tensortrain[i, 1]
        Ai = random_tensor_train[i, 0:dim_left, 0:dim_right, 0:local_basis_dim]
        tAi = np.tensordot(Ai, Uip1, axes=(1, 0))
        tAi = np.transpose(tAi, axes=(0, 2, 1))
        tAi = np.reshape(tAi, (dim_left, dim_right*local_basis_dim))
        Ui, Si, Vhi = np.linalg.svd(tAi, full_matrices=True)
        min_dim = np.min((dim_left, dim_right*local_basis_dim))
        Ui = Ui[:, 0:min_dim]
        Vhi = Vhi[0:min_dim, :]
        Vi = np.reshape(Vhi, (dim_left, dim_right, local_basis_dim))
        ortho_tensor_train[i, 0:dim_left, 0:dim_right, 0:local_basis_dim] = Vi
        Uip1 = Ui
    return ortho_tensor_train




###
### Contracts physical legs of tensor trains from the right, while
### also contracting the right-most auxiliary leg.
### JAX implementation.
###
### 'tensor_train_jnp_X' should be jnp arrays with size
### (no_legs, max_dim_1, max_dim_2, local_phys_dim).
### The last tensor dimension is interpreted as the physical leg, while 
### max_dim_1 and max_dim_2 are the left and right auxiliary legs, respectively.
###
def contract_tensortrains_from_right_jaxjit(tensor_train_jnp_1, tensor_train_jnp_2, no_legs,
                                            maxdims_tensortrain_1, maxdims_tensortrain_2):
    """
    JAX/JIT (jax.lax.fori_loop-based) contraction of two tensor trains of 'no_legs' sites against
    each other over both their physical legs and their right auxiliary legs, working from the
    last site towards the first. This is the JIT-compiled counterpart of
    tensor_network_functions_np.contract_tensortrains_from_right, used to compute overlaps such
    as Mjk = <V_j|V_k> that enter the tensor-network FIM computation (FIM_sample_jax,
    expected_trace_FIM_jax).

    Parameters
    ----------
    tensor_train_jnp_1, tensor_train_jnp_2 : ndarray
        Padded tensor trains of 'no_legs' sites with shape
        (no_legs, max_dim_left, max_dim_right, max_phys_dim).
    no_legs : int
        Number of tensor-train sites.
    maxdims_tensortrain_1, maxdims_tensortrain_2 : array-like, length 3
        Padded (dim_left, dim_right, physical) leg dimensions of each tensor train.

    Returns
    -------
    res : ndarray, shape (dim_left_1, dim_left_2)
        Result of contracting both tensor trains down to their left-most auxiliary legs.
    """
    dL1 = maxdims_tensortrain_1[0]
    dR1 = maxdims_tensortrain_1[1]
    dP1 = maxdims_tensortrain_1[2]
    dL2 = maxdims_tensortrain_2[0]
    dR2 = maxdims_tensortrain_2[1]
    dP2 = maxdims_tensortrain_2[2]

    @jax.jit
    def contract_phys_indices_jit(j, args):
        res, tensor_train_jnp_1, tensor_train_jnp_2 = args
        i = no_legs - j
        T1_i = jnp.squeeze(jax.lax.dynamic_slice(tensor_train_jnp_1, [i, 0, 0, 0], [1, dL1, dR1, dP1]), axis=0)
        T2_i = jnp.squeeze(jax.lax.dynamic_slice(tensor_train_jnp_2, [i, 0, 0, 0], [1, dL2, dR2, dP2]), axis=0)
        res_i = jnp.tensordot(T1_i, res, axes=(1, 0))  ## rank-3 with (axis 0: left-aux., axis 1: phys., axis 2: right-aux)
        res_i = jnp.transpose(res_i, axes=(0, 2, 1))  ## rank-3 with (axis 0: left-aux., axis 1: right-aux, axis 2: phys.)
        # Contract physical and right-aux. legs of last tensor
        res = jnp.tensordot(res_i, T2_i, axes=([1, 2], [1, 2]))  ## (axis 0: left-aux. from 1), (axis 1: left-aux. from 2)
        return (res, tensor_train_jnp_1, tensor_train_jnp_2)

    # Contract physical and right-aux. legs of last tensor
    i = no_legs - 1
    T1_i = jnp.squeeze(jax.lax.dynamic_slice(tensor_train_jnp_1, [i, 0, 0, 0], [1, dL1, dR1, dP1]), axis=0)
    T2_i = jnp.squeeze(jax.lax.dynamic_slice(tensor_train_jnp_2, [i, 0, 0, 0], [1, dL2, dR2, dP2]), axis=0)
    res = jnp.tensordot(T1_i, T2_i, axes=([1, 2], [1, 2]))  ## (axis 0: left-aux. from 1), (axis 1: left-aux. from 2)

    args = (res, tensor_train_jnp_1, tensor_train_jnp_2)
    (res, _, _) = jax.lax.fori_loop(2, no_legs + 1, contract_phys_indices_jit, args)
    return res


    

###
### Defines the (local_basis_dim x local_basis_dim) derivative tensor for
### a local space with basis {1, cos(th), ..., cos(L*th), sin(th), ..., sin(L*th)}
### with L = (local_basis_dim - 1)/2.
###
def derivative_tensor_jax(local_basis_dim):
    """
    JAX (jnp array) version of the local per-parameter derivative operator d/d(theta) acting on
    the trigonometric basis {1, cos(theta), ..., cos(L*theta), sin(theta), ..., sin(L*theta)},
    L = (local_basis_dim-1)/2 -- see tensor_network_functions_np.derivative_tensor for the
    identical (NumPy) construction. Inserted on a given leg of a parameter tensor train to
    differentiate the corresponding parameter basis function iota_nu(theta).

    Parameters
    ----------
    local_basis_dim : int
        Local per-parameter basis dimension d_tilde (odd, 2L+1).

    Returns
    -------
    B_jnp : ndarray, shape (local_basis_dim, local_basis_dim)
    """
    L = int((local_basis_dim - 1) / 2)
    B = np.zeros((local_basis_dim, local_basis_dim))
    BL = np.diag(np.arange(1, L+1))
    B[1:(L+1), (L+1):local_basis_dim] = BL
    B[(L+1):local_basis_dim, 1:(L+1)] = - BL
    B_jnp = jnp.asarray(B)
    return B_jnp



###
### Defines the local basis vector (local_basis_dim, 1)
### whose components are the local basis elements evaluated on 'param_m', i.e.,
### (1, cos(th), ..., cos(L*th), sin(th), ..., sin(L*th)) with th=param_m and
### with L = (local_basis_dim - 1)/2, and proper normalization in [-pi,pi].
###
def local_basis_vector_jax(param_m, local_basis_dim):
    """
    JAX version of local_basis_vector: evaluate the local per-parameter trigonometric basis
    (1, cos(theta), ..., cos(L*theta), sin(theta), ..., sin(L*theta)), L = (local_basis_dim-1)/2,
    normalized on [-pi, pi], at a single parameter value theta = param_m.

    Parameters
    ----------
    param_m : float
        Value of the parameter theta_m.
    local_basis_dim : int
        Local per-parameter basis dimension d_tilde.

    Returns
    -------
    basis_vec : ndarray, shape (local_basis_dim, 1)
    """
    L = int((local_basis_dim - 1) / 2)
    basis_vec = jnp.zeros((local_basis_dim, 1))
    basis_vec = basis_vec.at[0].set(1.0)
    basis_vec = basis_vec.at[1:(L+1), :].set(jnp.sqrt(2.0) * jnp.expand_dims(jnp.cos(param_m * jnp.arange(1,L+1)), axis=1))
    basis_vec = basis_vec.at[(L+1):local_basis_dim, :].set(jnp.sqrt(2.0) * jnp.expand_dims(jnp.sin(param_m * jnp.arange(1,L+1)), axis=1))
    return basis_vec



### Batched (over parameters theta_m) version of local_basis_vector_jax, vmapped over the
### first argument while local_basis_dim (d_tilde) is shared across all parameters.
fun_loc_basis_vec_jax = jax.vmap(local_basis_vector_jax, (0, None))




###
### Defines the local basis vectors as a (no_params, 1, 1, local_basis_dim) jnp array
### for the parameter configuration given in input.
###
def local_basis_vectors_jaxjit(params_jnp, local_dim_param):
    """
    Evaluate the local per-parameter basis vector (see local_basis_vector_jax) at every
    parameter theta_m in 'params_jnp', and package the results as a rank-4 array with a
    tensor-train-compatible leg layout (no_params, 1, 1, local_basis_dim), ready to be
    contracted against a parameter-side tensor train V^T (see
    contract_tensortrain_with_localvectors_jaxjit).

    Parameters
    ----------
    params_jnp : ndarray, shape (no_params,)
        Parameter configuration theta.
    local_dim_param : int
        Local per-parameter basis dimension d_tilde.

    Returns
    -------
    Ith_vec : ndarray, shape (no_params, 1, 1, local_dim_param)
    """
    Ith_vec = jnp.expand_dims(jnp.transpose(fun_loc_basis_vec_jax(params_jnp, local_dim_param), axes=(0, 2, 1)), axis=1)
    return Ith_vec



###
### Contracts physical legs of 'tensor_train_jnp' with
### 'local_vectors_jnp' from the right, while
### also contracting the (dummy) right-most auxiliary leg.
### JAX implementation.
###
def contract_tensortrain_with_localvectors_jaxjit(tensor_train_jnp, local_vectors_jnp, no_legs,
                                                  maxdims_tensortrain, maxdims_locvec):
    """
    Contract a parameter-side tensor train (representing V^T, bond dimension chi) with a set of
    local parameter basis vectors (from local_basis_vectors_jaxjit) evaluated at a parameter
    point theta, over both physical legs and right-auxiliary legs, working from the last site to
    the first. This computes V^T @ iota(theta) (equivalently V_j^T @ iota(theta) if one leg has
    first been replaced by the derivative_tensor_jax, for FIM computations), without ever
    forming the dense K = local_basis_dim**no_legs parameter basis.

    Parameters
    ----------
    tensor_train_jnp : ndarray
        Padded tensor train representing V^T, shape (no_legs, dL1, dR1, dP1).
    local_vectors_jnp : ndarray
        Padded local parameter basis vectors, shape (no_legs, dL2, dR2, dP2), typically from
        local_basis_vectors_jaxjit.
    no_legs : int
        Number of tensor-train sites (parameters M).
    maxdims_tensortrain : array-like, length 3
        Padded (dim_left, dim_right, physical) leg dimensions of the tensor train.
    maxdims_locvec : array-like, length 3
        Padded (dim_left, dim_right, physical) leg dimensions of the local basis vectors.

    Returns
    -------
    res : ndarray, shape (dim_left, 1)
        Contracted vector V^T @ iota(theta) (padded to the tensor train's left auxiliary
        dimension).
    """
    dL1 = maxdims_tensortrain[0]
    dR1 = maxdims_tensortrain[1]
    dP1 = maxdims_tensortrain[2]
    dL2 = maxdims_locvec[0]
    dR2 = maxdims_locvec[1]
    dP2 = maxdims_locvec[2]

    @jax.jit
    def contract_phys_indices_jit(j, args):
        res, tensor_train_jnp, local_vectors_jnp = args
        i = no_legs - j
        T1_i = jnp.squeeze(jax.lax.dynamic_slice(tensor_train_jnp, [i, 0, 0, 0], [1, dL1, dR1, dP1]), axis=0)
        T2_i = jnp.squeeze(jax.lax.dynamic_slice(local_vectors_jnp, [i, 0, 0, 0], [1, dL2, dR2, dP2]), axis=0)
        res_i = jnp.tensordot(T1_i, res, axes=(1, 0))  ## rank-3 with (axis 0: left-aux., axis 1: phys., axis 2: right-aux)
        res_i = jnp.transpose(res_i, axes=(0, 2, 1))  ## rank-3 with (axis 0: left-aux., axis 1: right-aux, axis 2: phys.)
        # Contract physical and right-aux. legs of last tensor
        res = jnp.tensordot(res_i, T2_i, axes=([1, 2], [1, 2]))  ## (axis 0: left-aux. from 1), (axis 1: left-aux. from 2)
        return (res, tensor_train_jnp, local_vectors_jnp)

    # Contract physical and right-aux. legs of last tensor
    i = no_legs - 1
    T1_i = jnp.squeeze(jax.lax.dynamic_slice(tensor_train_jnp, [i, 0, 0, 0], [1, dL1, 1, dP1]), axis=0)
    T2_i = jnp.squeeze(jax.lax.dynamic_slice(local_vectors_jnp, [i, 0, 0, 0], [1, dL2, 1, dP2]), axis=0)
    res = jnp.tensordot(T1_i, T2_i, axes=([1, 2], [1, 2]))  ## (axis 0: left-aux. from 1), (axis 1: left-aux. from 2)

    args = (res, tensor_train_jnp, local_vectors_jnp)
    (res, _, _) = jax.lax.fori_loop(2, no_legs + 1, contract_phys_indices_jit, args)
    return res




###
### Contracts physical legs of tensor train 'tensor_train_jnp' with 
### local derivative tensor at site j.
###
def contract_tensortrain_with_local_derivtensor_jax(j, tensor_train_jnp, B, maxdims_tensortrain):
    """
    Insert the local derivative operator B (from derivative_tensor_jax) on the physical leg of
    site 'j' of a tensor train, leaving all other sites unchanged. This implements
    differentiation of the parameter-side map V^T with respect to theta_j
    (V_j^T ~ V^T with a d/d(theta_j) inserted on leg j), the key building block for computing
    the Fisher Information Matrix entries F_{j,k}(theta) from the tensor-train representation of
    V^T (see FIM_sample_jax).

    Parameters
    ----------
    j : int
        Index of the tensor-train site (parameter) to differentiate.
    tensor_train_jnp : ndarray
        Padded tensor train representing V^T.
    B : ndarray, shape (local_basis_dim, local_basis_dim)
        Local derivative operator (from derivative_tensor_jax).
    maxdims_tensortrain : array-like, length 3
        Padded (dim_left, dim_right, physical) leg dimensions of the tensor train.

    Returns
    -------
    Vj_tensors : ndarray
        Tensor train equal to tensor_train_jnp everywhere except site j, where the physical leg
        has been contracted with B.
    """
    dL = maxdims_tensortrain[0]
    dR = maxdims_tensortrain[1]
    dP = maxdims_tensortrain[2]
    Vj_tensors = jnp.copy(tensor_train_jnp)
    Vj = jnp.squeeze(jax.lax.dynamic_slice(tensor_train_jnp, [j, 0, 0, 0], [1, dL, dR, dP]), axis=0)
    Vj = jnp.tensordot(Vj, B, axes=(2, 0))
    Vj_tensors = Vj_tensors.at[j, 0:dL, 0:dR, 0:dP].set(Vj)
    return Vj_tensors




###
### FIM for given parameters calculated as a TN (JAX).
###
def FIM_sample_jax(local_basis_vectors_jnp, tensor_train_jnp, maxdims_tensortrain, maxdims_locvec,
                   no_params, B, Svals):
    """
    Evaluate the (no_params x no_params) Fisher Information Matrix F(theta) at a single parameter
    point theta, from the tensor-train representation of the parameter-side map V^T and the
    correlation spectrum Svals -- the JAX/JIT-compiled counterpart of
    tensor_network_functions_np.FIM_sample. For each parameter j, the derivative operator B is
    inserted on leg j (contract_tensortrain_with_local_derivtensor_jax) and the resulting tensor
    train is contracted with the local basis vectors at theta
    (contract_tensortrain_with_localvectors_jaxjit); the FIM is then F = W @ diag(Svals) @ W^T
    with W stacking these contracted (S-weighted) vectors over all parameters j.

    Parameters
    ----------
    local_basis_vectors_jnp : ndarray
        Local parameter basis vectors at theta (from local_basis_vectors_jaxjit).
    tensor_train_jnp : ndarray
        Padded tensor train representing V^T.
    maxdims_tensortrain, maxdims_locvec : array-like, length 3
        Padded leg dimensions of the tensor train and of the local basis vectors.
    no_params : int
        Number of trainable parameters M.
    B : ndarray, shape (local_basis_dim, local_basis_dim)
        Local derivative operator (from derivative_tensor_jax).
    Svals : ndarray, shape (D,)
        Correlation spectrum (singular values of Gamma).

    Returns
    -------
    FIM : ndarray, shape (no_params, no_params)
    """

    def contract_Vj_with_localvectors(j):
        Vj_tensors_jnp = contract_tensortrain_with_local_derivtensor_jax(j, tensor_train_jnp, B, maxdims_tensortrain)
        VjI_params = contract_tensortrain_with_localvectors_jaxjit(Vj_tensors_jnp, local_basis_vectors_jnp, no_params, 
                                                                   maxdims_tensortrain, maxdims_locvec)
        return VjI_params

    contract_Vjs = jax.vmap(contract_Vj_with_localvectors)

    Vjs_contracted = jnp.squeeze(contract_Vjs(jnp.arange(0,no_params)))  # (no_params, max_aux_dim -> dim_in)
    W = Vjs_contracted * Svals  # (no_params, max_aux_dim -> dim_in)
    FIM = jnp.matmul(W, jnp.transpose(W))
    return FIM




###
### Expectation value of trace(FIM) calculated as a TN (JAX).
###
def expected_trace_FIM_jax(tensor_train_jnp, maxdims_tensortrain, no_params, B, Svals):
    """
    Compute E_theta[tr(F(theta))] from the tensor-train representation of V^T and the
    correlation spectrum Svals -- the JAX/JIT-compiled counterpart of
    tensor_network_functions_np.expected_trace_FIM. Used to normalize the FIM (see
    normalized_FIM_sample_jax) analogously to Eq. (13) of the paper, while scaling to larger
    numbers of parameters M via the tensor-train contraction.

    Parameters
    ----------
    tensor_train_jnp : ndarray
        Padded tensor train representing V^T.
    maxdims_tensortrain : array-like, length 3
        Padded leg dimensions of the tensor train.
    no_params : int
        Number of trainable parameters M.
    B : ndarray, shape (local_basis_dim, local_basis_dim)
        Local derivative operator (from derivative_tensor_jax).
    Svals : ndarray, shape (D,)
        Correlation spectrum (singular values of Gamma).

    Returns
    -------
    trFIM : float
        Expected trace of the (unnormalized) FIM.
    """
    Svals2 = jnp.diag(Svals ** 2.0)

    def contract_Vj_with_Vj(j):
        Vj_tensors_jnp = contract_tensortrain_with_local_derivtensor_jax(j, tensor_train_jnp, B, maxdims_tensortrain)
        Mjj = contract_tensortrains_from_right_jaxjit(Vj_tensors_jnp, Vj_tensors_jnp, no_params, maxdims_tensortrain, maxdims_tensortrain)
        Fjj = jnp.trace(jnp.matmul(Svals2, Mjj))
        return Fjj

    exp_diag_FIM = jax.vmap(contract_Vj_with_Vj)
    trFIM = jnp.sum(exp_diag_FIM(jnp.arange(0,no_params)))
    return trFIM




###
### Normalized FIM for given parameters calculated as a TN (JAX).
###
def normalized_FIM_sample_jax(local_basis_vectors_jnp, tensor_train_jnp, maxdims_tensortrain, maxdims_locvec,
                              no_params, B, Svals):
    """
    Normalized FIM F_hat(theta) = (M / tr(F(theta))) * F(theta) (Eq. (13) of the paper),
    evaluated from the tensor-train representation of V^T (combines FIM_sample_jax and
    expected_trace_FIM_jax). The spectrum of F_hat(theta), averaged over theta, feeds into the
    effective-dimension formula (Eq. (12)).

    Parameters
    ----------
    local_basis_vectors_jnp : ndarray
        Local parameter basis vectors at theta (from local_basis_vectors_jaxjit).
    tensor_train_jnp : ndarray
        Padded tensor train representing V^T.
    maxdims_tensortrain, maxdims_locvec : array-like, length 3
        Padded leg dimensions of the tensor train and of the local basis vectors.
    no_params : int
        Number of trainable parameters M.
    B : ndarray, shape (local_basis_dim, local_basis_dim)
        Local derivative operator (from derivative_tensor_jax).
    Svals : ndarray, shape (D,)
        Correlation spectrum (singular values of Gamma).

    Returns
    -------
    normalized_FIM : ndarray, shape (no_params, no_params)
    """
    FIM = FIM_sample_jax(local_basis_vectors_jnp, tensor_train_jnp, maxdims_tensortrain, maxdims_locvec,
                         no_params, B, Svals)
    exp_trF = expected_trace_FIM_jax(tensor_train_jnp, maxdims_tensortrain, no_params, B, Svals)
    return no_params * FIM / exp_trF