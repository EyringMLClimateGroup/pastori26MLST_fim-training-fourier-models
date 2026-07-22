"""
NumPy implementation of tensor-train (matrix product state) utilities used to represent the
parameter-side map V^T (from Gamma = U @ diag(S) @ V^T) as a tensor network of bond dimension
chi, together with the routines needed to compute the FIM directly from this tensor-train
representation (i.e. without ever forming the dense K = local_basis_dim**no_params parameter
basis). This is the reference/small-scale (non-JAX, non-JIT) counterpart of
tensor_network_functions_jax.py, useful for testing and for the tensorized-model FIM checks.
"""

import numpy as np
import copy




###
### Construct TN representation of orthogonal matrix. The second dimension is decomposed.
###
def tensortrain_from_ortho_matrix(V, no_legs, local_basis_dim):
    """
    Decompose a dense (dim_in x dim_out) orthogonal matrix V into a tensor train of 'no_legs'
    rank-3 tensors via successive SVDs, with the output ("dim_out") dimension split into
    no_legs legs of local_basis_dim each (dim_out = local_basis_dim**no_legs). This turns a
    parameter-side matrix V (from Gamma = U @ diag(S) @ V^T) into its matrix-product-state form,
    with the tensor-train bond dimension controlled implicitly by successive SVD truncations to
    min(left_dim, right_dim).

    Parameters
    ----------
    V : ndarray, shape (dim_in, dim_out)
        Dense orthogonal matrix to decompose (dim_out = local_basis_dim**no_legs).
    no_legs : int
        Number of tensor-train sites (e.g. number of parameters M).
    local_basis_dim : int
        Local physical leg dimension per site (d_tilde).

    Returns
    -------
    tensors_list : list of ndarray
        List of no_legs rank-3 tensors (dim_left, dim_right, local_basis_dim) forming the
        tensor-train representation of V.
    """
    dim_in = V.shape[0]
    dim_out = V.shape[1]
    tensors_list = []
    dim_left = dim_in
    dim_right = dim_out
    Wh = copy.deepcopy(V)
    for m in range(no_legs):
        dim_right = int(dim_right/local_basis_dim)
        tWh_r = np.reshape(Wh, (dim_left*local_basis_dim, dim_right))
        min_dim = np.min((tWh_r.shape[0], tWh_r.shape[1]))
        U, S, Wh = np.linalg.svd(tWh_r, full_matrices=True)
        U = U[:, 0:min_dim]
        U = np.matmul(U, np.diag(S))
        Um = np.reshape(U, (dim_left, local_basis_dim, min_dim))
        Um = np.transpose(Um, (0, 2, 1))
        tensors_list.append(Um)
        Wh = Wh[0:min_dim, :]
        dim_left = min_dim
    return tensors_list



###
### Contracts physical legs of tensor trains from the right, while
### also contracting the right-most auxiliary leg.
###
### 'tensor_train_X' should be lists of no_legs tensors of nominal rank 3,
### i.e., whose shape should be a 3-tuple. The last tensor dimension
### is interpreted as the physical leg, while the first and second
### tensor dimensions are the left and right auxiliary legs, respectively.
### Indices could be trivial, i.e., one-dimensional.
###
def contract_tensortrains_from_right(tensor_train_1, tensor_train_2, no_legs):
    """
    Contract two tensor trains of 'no_legs' sites against each other over both their physical
    legs and their right auxiliary legs, working from the last site towards the first. Used to
    compute inner products/overlaps between tensor-train-represented vectors, e.g. to build
    Mjk = <V_j|V_k> contractions entering the FIM (FIM_sample, expected_trace_FIM).

    Parameters
    ----------
    tensor_train_1, tensor_train_2 : list of ndarray
        Lists of 'no_legs' rank-3 tensors (dim_left, dim_right, physical), as returned by
        tensortrain_from_ortho_matrix. The two tensor trains may have different bond dimensions.
    no_legs : int
        Number of tensor-train sites.

    Returns
    -------
    res : ndarray, shape (dim_left_1, dim_left_2)
        Result of contracting both tensor trains down to their left-most auxiliary legs.
    """
    # Contract physical and right-aux. legs of last tensor
    res = np.tensordot(tensor_train_1[-1], tensor_train_2[-1], axes=([1, 2], [1, 2]))  ## (axis 0: left-aux. from 1), (axis 1: left-aux. from 2)
    for i in range(2, no_legs + 1):
        res_i = np.tensordot(tensor_train_1[-i], res, axes=(1, 0))  ## rank-3 with (axis 0: left-aux., axis 1: phys., axis 2: right-aux)
        res_i = np.transpose(res_i, axes=(0, 2, 1))  ## rank-3 with (axis 0: left-aux., axis 1: right-aux, axis 2: phys.)
        # Contract physical and right-aux. legs of last tensor
        res = np.tensordot(res_i, tensor_train_2[-i], axes=([1, 2], [1, 2]))  ## (axis 0: left-aux. from 1), (axis 1: left-aux. from 2)
    return res




###
### Defines the (local_basis_dim x local_basis_dim) derivative tensor for
### a local space with basis {1, cos(th), ..., cos(L*th), sin(th), ..., sin(L*th)}
### with L = (local_basis_dim - 1)/2.
###
def derivative_tensor(local_basis_dim):
    """
    Matrix representation of d/d(theta) acting on the local per-parameter trigonometric basis
    {1, cos(theta), ..., cos(L*theta), sin(theta), ..., sin(L*theta)}, L = (local_basis_dim-1)/2.
    Applied to one leg of a parameter tensor train, it differentiates the corresponding
    parameter basis function iota_nu(theta) -- see analytical_FIM_functions_np.derivative_tensor
    for the identical (dense) construction.

    Parameters
    ----------
    local_basis_dim : int
        Local per-parameter basis dimension d_tilde (odd, 2L+1).

    Returns
    -------
    B : ndarray, shape (local_basis_dim, local_basis_dim)
    """
    L = int((local_basis_dim - 1) / 2)
    B = np.zeros((local_basis_dim, local_basis_dim))
    BL = np.diag(np.arange(1, L+1))
    B[1:(L+1), (L+1):local_basis_dim] = BL
    B[(L+1):local_basis_dim, 1:(L+1)] = - BL
    return B



###
### Defines the local basis vector (local_basis_dim, 1)
### whose components are the local basis elements evaluated on 'param_m', i.e.,
### (1, cos(th), ..., cos(L*th), sin(th), ..., sin(L*th)) with th=param_m and
### with L = (local_basis_dim - 1)/2, and proper normalization in [-pi,pi].
###
def local_basis_vector(param_m, local_basis_dim):
    """
    Evaluate the local per-parameter trigonometric basis
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
    basis_vec = np.zeros((local_basis_dim, 1))
    basis_vec[0] = 1.0
    basis_vec[1:(L+1), :] = np.sqrt(2.0) * np.expand_dims(np.cos(param_m * np.arange(1,L+1)), axis=1)
    basis_vec[(L+1):local_basis_dim, :] = np.sqrt(2.0) * np.expand_dims(np.sin(param_m * np.arange(1,L+1)), axis=1)
    return basis_vec




###
### Expectation value over param space of trace of FIM calculated as a TN.
###
def expected_trace_FIM(no_params, local_basis_dim, Svals, V_tensors):
    """
    Compute E_theta[tr(F(theta))] directly from the tensor-train representation V_tensors of the
    parameter-side map V^T and the correlation spectrum Svals, without expanding V into a dense
    (D x K) matrix. Used to normalize the FIM (see normalized_FIM_sample) exactly as in
    analytical_FIM_functions_np.expected_trace_FIM_fullmatrices, but scaling to larger numbers of
    parameters M via the tensor-train contraction.

    Parameters
    ----------
    no_params : int
        Number of trainable parameters M (tensor-train sites).
    local_basis_dim : int
        Local per-parameter basis dimension d_tilde.
    Svals : ndarray, shape (D,)
        Correlation spectrum (singular values of Gamma).
    V_tensors : list of ndarray
        Tensor-train representation of V^T (see tensortrain_from_ortho_matrix).

    Returns
    -------
    exp_trF : float
        Expected trace of the (unnormalized) FIM.
    """
    Svals2 = np.diag(Svals ** 2.0)
    B = derivative_tensor(local_basis_dim)
    exp_trF = 0.0
    for j in range(no_params):
        Vj_tensors = copy.deepcopy(V_tensors)
        Vj = Vj_tensors[j]
        Vj = np.tensordot(Vj, B, axes=(2, 0))
        Vj_tensors[j] = Vj
        Mjj = contract_tensortrains_from_right(Vj_tensors, Vj_tensors, no_params)
        Fjj = np.trace(np.matmul(Svals2, Mjj))
        exp_trF = exp_trF + Fjj
    return exp_trF




###
### FIM for given parameters calculated as a TN.
###
def FIM_sample(params, no_params, local_basis_dim, Svals, V_tensors):
    """
    Evaluate the (no_params x no_params) Fisher Information Matrix F(theta) at a single parameter
    point theta, using the tensor-train representation V_tensors of V^T and the correlation
    spectrum Svals, in place of the dense computation in
    analytical_FIM_functions_np.FIM_sample_fullmatrices. For each pair (j, k) it inserts the
    local derivative_tensor B on leg j (resp. k) of V_tensors and contracts with the local basis
    vector at theta on every other leg, then contracts the two resulting tensor trains
    (weighted by S^2) via contract_tensortrains_from_right.

    Parameters
    ----------
    params : ndarray, shape (no_params,)
        Parameter configuration theta at which the FIM is evaluated.
    no_params : int
        Number of trainable parameters M.
    local_basis_dim : int
        Local per-parameter basis dimension d_tilde.
    Svals : ndarray, shape (D,)
        Correlation spectrum (singular values of Gamma).
    V_tensors : list of ndarray
        Tensor-train representation of V^T.

    Returns
    -------
    FIM : ndarray, shape (no_params, no_params)
    """
    Svals2 = np.diag(Svals ** 2.0)
    B = derivative_tensor(local_basis_dim)
    local_basis_vectors = []
    for j in range(no_params):
        param_j = params[j]
        loc_vec = local_basis_vector(param_j, local_basis_dim)
        local_basis_vectors.append(loc_vec)
    FIM = np.zeros((no_params, no_params))
    for j in range(no_params-1):
        Vj_tensors = copy.deepcopy(V_tensors)
        Vj = Vj_tensors[j]
        Vj = np.tensordot(Vj, B, axes=(2, 0))
        Vj_tensors[j] = Vj
        for jp in range(no_params):
            Ijp = local_basis_vectors[jp]
            Vjp = Vj_tensors[jp]
            Vjp = np.tensordot(Vjp, Ijp, axes=(2, 0))
            Vj_tensors[jp] = Vjp
        for k in range(j, no_params):
            Vk_tensors = copy.deepcopy(V_tensors)
            Vk = Vk_tensors[k]
            Vk = np.tensordot(Vk, B, axes=(2, 0))
            Vk_tensors[k] = Vk
            for kp in range(no_params):
                Ikp = local_basis_vectors[kp]
                Vkp = Vk_tensors[kp]
                Vkp = np.tensordot(Vkp, Ikp, axes=(2, 0))
                Vk_tensors[kp] = Vkp
            Mjk = contract_tensortrains_from_right(Vj_tensors, Vk_tensors, no_params)
            Fjk = np.trace(np.matmul(Svals2, Mjk))
            FIM[j,k] = Fjk
            FIM[k,j] = Fjk
    j = no_params - 1
    Vj_tensors = copy.deepcopy(V_tensors)
    Vj = Vj_tensors[j]
    Vj = np.tensordot(Vj, B, axes=(2, 0))
    Vj_tensors[j] = Vj
    for jp in range(no_params):
        Ijp = local_basis_vectors[jp]
        Vjp = Vj_tensors[jp]
        Vjp = np.tensordot(Vjp, Ijp, axes=(2, 0))
        Vj_tensors[jp] = Vjp
    Mjj = contract_tensortrains_from_right(Vj_tensors, Vj_tensors, no_params)
    Fjj = np.trace(np.matmul(Svals2, Mjj))
    FIM[j,j] = Fjj
    return FIM

###
### Normalized FIM for given parameters calculated as a TN.
###
def normalized_FIM_sample(params, no_params, local_basis_dim, Svals, V_tensors):
    """
    Normalized FIM F_hat(theta) = (M / tr(F(theta))) * F(theta) (Eq. (13) of the paper),
    evaluated via the tensor-train representation V_tensors of V^T (combines FIM_sample and
    expected_trace_FIM).

    Parameters
    ----------
    params : ndarray, shape (no_params,)
        Parameter configuration theta.
    no_params : int
        Number of trainable parameters M.
    local_basis_dim : int
        Local per-parameter basis dimension d_tilde.
    Svals : ndarray, shape (D,)
        Correlation spectrum (singular values of Gamma).
    V_tensors : list of ndarray
        Tensor-train representation of V^T.

    Returns
    -------
    normalized_FIM : ndarray, shape (no_params, no_params)
    """
    FIM = FIM_sample(params, no_params, local_basis_dim, Svals, V_tensors)
    exp_trF = expected_trace_FIM(no_params, local_basis_dim, Svals, V_tensors)
    return no_params * FIM / exp_trF
