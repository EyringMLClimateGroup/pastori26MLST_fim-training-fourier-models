"""
Construction of a *fully* tensorized Fourier model, in which both factors of the structure
constants Gamma = U @ diag(S) @ V^T (Eq. (22) of the paper) are represented as tensor networks:
- the parameter side V^T is a tensor train (matrix product state) over the M parameter legs,
  exactly as in tensorized_model_constructor_functions.py;
- the input side U is represented as an orthogonal Matrix Product Operator (MPO) of local
  dimension d, followed by an isometric Tree Tensor Network (TTN) of bond dimension 'bond_dim'
  that maps the N-feature input space (dimension D = d**N) down to the correlation-space
  dimension d_out.

This lets both the number of input features N and the number of parameters M grow while
avoiding the exponential storage cost of the dense D x K matrix Gamma, at the price of the
approximation controlled by the MPO/TTN bond dimensions. The module provides routines to: build
random orthogonal MPOs (shallow, made of nearest-neighbor two-site orthogonal gates) and random
isometric TTNs; contract an input's local basis vectors through the MPO+TTN pipeline; and
evaluate the resulting fully tensorized Fourier model on a batch of inputs.
"""

import numpy as np
import jax
from jax import numpy as jnp
import scipy
import copy
import importlib

import tensor_network_functions_jax
importlib.reload(tensor_network_functions_jax)
import tensor_network_functions_jax as TN_fns_jax




############################################################################################################
################################## FUNCTIONS FOR ORTHOGONAL MPO CREATION ###################################
############################################################################################################

def dimensions_shallow_orthogonal_MPO(N, d):
    '''
    Compute the (auxiliary-left, auxiliary-right, physical-in, physical-out) leg dimensions of
    each of the N tensors forming a shallow orthogonal Matrix Product Operator (MPO), i.e. the
    tensor-network representation of the input-side orthogonal matrix U (from
    Gamma = U @ diag(S) @ V^T) acting on N local input-feature legs of local dimension d.
    Auxiliary legs have dimension d**2 (except at the open boundaries, dimension 1).

    Parameters
    ----------
    N : int
        Number of local (input-feature) legs / sites of the MPO.
    d : int
        Local physical dimension per site (per-feature input basis dimension).

    Returns
    -------
    MPO_dims : ndarray, shape (N, 4)
        Rows are (dim_left, dim_right, dim_phys_in, dim_phys_out) for each site.
    '''
    MPO_dims = np.zeros((N, 4), dtype=np.int32)
    MPO_dims[:, 0] = d**2
    MPO_dims[:, 1] = d**2
    MPO_dims[:, 2] = d
    MPO_dims[:, 3] = d
    MPO_dims[0, 0] = 1
    MPO_dims[N-1, 1] = 1
    return MPO_dims



def generate_random_shallow_orthogonal_MPO_padded(N, d, max_phys_dim):
    '''
    Create a random orthogonal MPO (the tensor-network representation of the input-side
    orthogonal matrix U in Gamma = U @ diag(S) @ V^T) obtained from one layer of nearest-neighbor
    two-site orthogonal gates, i.e.,
       ___    ___
    --|___|--|___|-----------------------
              _|_    ___
    ---------|___|--|___|----------------
                     _|_    ___
    ----------------|___|--|___|---------
                            _|_    ___
    -----------------------|___|--|___|--
    Each two-site gate is a Haar-random orthogonal matrix, SVD-split into two nearest-neighbor
    MPO tensors and fused with its neighbors. Returns an array of rank-4 tensors with dimensions:
    0: indexing the tensor (site), 1: left-auxiliary leg, 2: right-auxiliary leg,
    3: incoming physical leg, 4: outgoing physical leg.

    'd' is the actual local physical dimension (per-feature input basis dimension), which is
    then padded with zeros to have a maximal physical leg dimension 'max_phys_dim' for
    compatibility with JAX's fixed-shape contraction routines.

    Parameters
    ----------
    N : int
        Number of local (input-feature) legs / sites of the MPO.
    d : int
        Local physical dimension per site.
    max_phys_dim : int
        Padded physical leg dimension used for uniform JAX array shapes.

    Returns
    -------
    ortho_mpo_tensors : ndarray, shape (N, d**2, d**2, max_phys_dim, max_phys_dim)
        Padded array of MPO tensors representing the orthogonal matrix U.
    '''
    ortho_mpo_tensors = np.zeros((N, d**2, d**2, max_phys_dim, max_phys_dim))
    U_im1 = scipy.stats.ortho_group.rvs(dim=d, size=1)
    U_im1 = np.expand_dims(U_im1, axis=0)
    U_im1 = np.expand_dims(U_im1, axis=0)  # (1, 1, d, d)
    U_list_im1 = [U_im1, U_im1]
    for i in range(N-1):
        U_i = scipy.stats.ortho_group.rvs(dim=d**2, size=1)
        U_list_i = []
    
        # Reshape U_i into a 2-site tensor of shape (d, d, d, d)
        # U((i1,i2),(j1,j2)) --> U(i1,i2,j1,j2)
        U_tensor_i = U_i.reshape([d] * (2 * 2))
        # Permute indices so that signature is U(i1,j1,i2,j2)
        perm0 = [n for n in range(4)]
        perm = [n for n in range(4)]
        for n in range(2):
            perm[2*n] = perm0[n]
            perm[2*n + 1] = perm0[2 + n]
        perm = tuple(perm)
        U_tensor_i = np.transpose(U_tensor_i, axes=perm)

        U_tensor_i = np.reshape(U_tensor_i, (d*d, d*d))
        UU, S, VVh = np.linalg.svd(U_tensor_i, full_matrices=True)
        III = np.argsort(S)
        UU = UU[:, III]
        S = S[III]
        VVh = VVh[III, :]
        min_dim = d**2
        UU = np.matmul(UU, np.diag(S))
        dim_left_1 = 1
        dim_right_1 = min_dim
        UU1 = np.reshape(UU, (dim_left_1, d**2, dim_right_1))
        UU1 = np.transpose(UU1, (0, 2, 1))
        UU1 = np.reshape(UU1, (dim_left_1, dim_right_1, d, d))
        U_list_i.append(UU1)
        dim_left_2 = min_dim
        dim_right_2 = 1
        UU2 = np.reshape(VVh, (dim_left_2, d**2, dim_right_2))
        UU2 = np.transpose(UU2, (0, 2, 1))
        UU2 = np.reshape(UU2, (dim_left_2, dim_right_2, d, d))
        U_list_i.append(UU2)

        ### Multiply last tensor of previous bond with first tensor of current bond
        U_i = U_list_i[0]
        res_0 = np.tensordot(U_list_im1[1], U_i, axes=(3, 2))  # (L1, R1=1, Pi1, L2=1, R2, Po2) = (L1, 1, Pi1, 1, R2, Po2)
        res_0 = np.squeeze(res_0, axis=(1, 3))  # (L1, Pi1, R2, Po2)
        res_0 = np.transpose(res_0, axes=(0, 2, 1, 3))   # (L1, R2, Pi1, Po2)
        U_list_i[0] = res_0
        ortho_mpo_tensors[i, :res_0.shape[0], :res_0.shape[1], :res_0.shape[2], :res_0.shape[3]] = res_0
        U_list_im1 = U_list_i

    ### Last leg
    U_i = scipy.stats.ortho_group.rvs(dim=d, size=1)
    U_i = np.expand_dims(U_i, axis=0)
    U_i = np.expand_dims(U_i, axis=0)  # (1, 1, d, d)
    res_0 = np.tensordot(U_list_im1[1], U_i, axes=(3, 2))  # (L1, R1=1, Pi1, L2=1, R2, Po2) = (L1, 1, Pi1, 1, R2, Po2)
    res_0 = np.squeeze(res_0, axis=(1, 3))  # (L1, Pi1, R2, Po2)
    res_0 = np.transpose(res_0, axes=(0, 2, 1, 3))   # (L1, R2, Pi1, Po2)
    ortho_mpo_tensors[N-1, :res_0.shape[0], :res_0.shape[1], :res_0.shape[2], :res_0.shape[3]] = res_0

    return ortho_mpo_tensors





############################################################################################################
################################## FUNCTIONS FOR ISOMETRIC TTN CREATION ####################################
############################################################################################################

def dimensions_isometric_TTN_tensors(N, d_in, bond_dim, d_out):
    '''
    Compute the (dim_in_1, dim_in_2, dim_out) leg dimensions of each isometric tensor in a Tree
    Tensor Network (TTN) with 2 incoming and 1 outgoing leg per node. The TTN implements the
    isometric mapping T from the (post-MPO) input space down to the correlation-space dimension
    d_out (Eq. (22): Gamma ~= U @ T @ diag(S) @ V^T). The number of leaves N is assumed to be a
    power of 2, so the network has log2(N) layers, with N/2**m tensors in layer m; all internal
    legs have dimension 'bond_dim' except the leaves (d_in) and the root (d_out).

    Parameters
    ----------
    N : int
        Number of leaves of the TTN (must be a power of 2).
    d_in : int
        Leg dimension at the leaves (post-MPO local input dimension).
    bond_dim : int
        Internal TTN bond dimension.
    d_out : int
        Output leg dimension at the root (correlation-space dimension).

    Returns
    -------
    TTN_dims : ndarray, shape (no_tensors, 3)
        Rows are (dim_in_1, dim_in_2, dim_out) for each TTN tensor, ordered layer by layer.
    '''
    no_tensors = 0
    no_layers = int(np.log2(N))
    for l in range(no_layers):
        no_tensors_l = int(N / (2**(l+1)))
        no_tensors = no_tensors + no_tensors_l
    TTN_dims = np.zeros((no_tensors, 3), dtype=np.int32)
    
    if N==2:
        TTN_dims[0, 0] = d_in
        TTN_dims[0, 1] = d_in
        TTN_dims[0, 2] = d_out
    else:
        cc = 0
        for l in range(no_layers):
            no_tensors_l = int(N / (2**(l+1)))
            if l==0:
                for j in range(no_tensors_l): 
                    TTN_dims[cc, 0] = d_in
                    TTN_dims[cc, 1] = d_in
                    TTN_dims[cc, 2] = bond_dim
                    cc = cc + 1
            else:
                if l==(no_layers-1):
                    for j in range(no_tensors_l): 
                        TTN_dims[cc, 0] = bond_dim
                        TTN_dims[cc, 1] = bond_dim
                        TTN_dims[cc, 2] = d_out
                        cc = cc + 1
                else:
                    for j in range(no_tensors_l): 
                        TTN_dims[cc, 0] = bond_dim
                        TTN_dims[cc, 1] = bond_dim
                        TTN_dims[cc, 2] = bond_dim
                        cc = cc + 1
    return TTN_dims



def generate_random_isometric_TTN(N, d_in, bond_dim, d_out):
    '''
    Generate a random isometric Tree Tensor Network (TTN), i.e. layers of random isometric
    tensors with 2 incoming and 1 outgoing leg each, obtained by truncating Haar-random
    orthogonal matrices to their first d_out columns at each node. This TTN implements the
    isometric mapping T from the (post-MPO) input space to the d_out-dimensional correlation
    space (Eq. (22) of the paper). The number of leaves N is assumed to be a power of 2, so the
    network has log2(N) layers with N/2**m tensors in layer m.

    Parameters
    ----------
    N : int
        Number of leaves of the TTN (must be a power of 2).
    d_in : int
        Leg dimension at the leaves (post-MPO local input dimension).
    bond_dim : int
        Internal TTN bond dimension.
    d_out : int
        Output leg dimension at the root (correlation-space dimension).

    Returns
    -------
    ttn_tensors : ndarray, shape (no_tensors, max_d_in, max_d_in, max_d_out)
        Padded array of TTN isometric tensors, ordered layer by layer.
    '''
    no_tensors = 0
    no_layers = int(np.log2(N))
    for l in range(no_layers):
        no_tensors_l = int(N / (2**(l+1)))
        no_tensors = no_tensors + no_tensors_l
    max_d_in = np.max((d_in, bond_dim))
    max_d_out = np.max((d_out, bond_dim))
    ttn_tensors = np.zeros((no_tensors, max_d_in, max_d_in, max_d_out))
    
    if N==2:
        bond_dim = d_out
    M = N / 2
    d_in_c = d_in
    d_out_c = bond_dim
    cc = 0
    while M >= 1:
        D = np.max((d_in_c**2, d_out_c))
        M = int(M)
        for m in range(M):
            U = scipy.stats.ortho_group.rvs(dim=D, size=1)
            IU = U[:, 0:d_out_c]
            IT = np.reshape(IU, (d_in_c, d_in_c, d_out_c))
            ttn_tensors[cc, :IT.shape[0], :IT.shape[1], :IT.shape[2]] = IT
            cc = cc + 1
        M = M / 2
        d_in_c = d_out_c
        if int(M)==1:
            d_out_c = d_out
    return ttn_tensors




############################################################################################################
######################################## TN CONTRACTION FUNCTIONS ##########################################
############################################################################################################

def contract_vector_with_MPO_TTN_jaxjit(N, local_vectors, mpo_tensors, ttn_tensors,
                                        maxdims_mpo, max_phys_dim, no_ttn_layers, d_out):
    '''
    Contract a product-state vector (the per-feature local input basis vectors in
    'local_vectors') with the orthogonal MPO ('mpo_tensors', representing U) and then with the
    isometric TTN ('ttn_tensors', representing T), implementing
    T @ U @ e(x) for a single input point x (Eq. (22) of the paper: Gamma ~= U @ T @ diag(S) @ V^T,
    with T @ U here playing the role of the isometry into the d_out-dimensional correlation space).

    Parameters
    ----------
    N : int
        Number of input-feature legs.
    local_vectors : ndarray, shape (N, 1, 1, max_phys_dim)
        Local input basis vectors for one input point (see local_feature_input_vectors_padded_jax).
    mpo_tensors : ndarray
        Orthogonal MPO tensors (see generate_random_shallow_orthogonal_MPO_padded), representing U.
    ttn_tensors : ndarray
        Isometric TTN tensors (see generate_random_isometric_TTN), representing T.
    maxdims_mpo : ndarray
        Padded (left, right) auxiliary leg dimensions of the MPO.
    max_phys_dim : int
        Padded physical leg dimension shared by the MPO and local input vectors.
    no_ttn_layers : int
        Number of TTN layers, log2(N).
    d_out : int
        Output dimension of the TTN (correlation-space dimension).

    Returns
    -------
    res : ndarray, shape (d_out,)
        Isometrically mapped vector T @ U @ e(x).
    '''
    Lu = maxdims_mpo[0]
    Ru = maxdims_mpo[1]

    @jax.jit
    def contract_phys_leg_vector_with_mpo_jit(m, args):
        contracted_tensors, local_vectors, mpo_tensors = args
        v_m = jax.lax.dynamic_slice(local_vectors, [m, 0, 0, 0], [1, 1, 1, max_phys_dim])
        v_m = jnp.squeeze(v_m, axis=0)  # (1, 1, Pv(o))
        U_m = jax.lax.dynamic_slice(mpo_tensors, [m, 0, 0, 0, 0], [1, Lu, Ru, max_phys_dim, max_phys_dim])
        U_m = jnp.squeeze(U_m, axis=0)  # (Lu, Ru, Pu(i), Pu(o))
        res = jnp.tensordot(v_m, U_m, axes=(2, 2))  # (1, 1, Lu, Ru, Pu(o))
        res = jnp.squeeze(res, axis=(0, 1))  # (Lu, Ru, Pu(o))
        contracted_tensors = contracted_tensors.at[m, :, :, :].set(res)
        return (contracted_tensors, local_vectors, mpo_tensors)

    @jax.jit
    def contract_with_ttn_tensor_jit(m, args):
        contracted_tensors, ttn_tensors, counter = args
        C1 = jax.lax.dynamic_slice(contracted_tensors, [2*m, 0, 0, 0], [1, Lu, Ru, max_phys_dim])
        C1 = jnp.squeeze(C1, axis=0)  # (Lc1, Rc1<->Lc2, Pc1(o)<->Pt(i1))
        C2 = jax.lax.dynamic_slice(contracted_tensors, [2*m+1, 0, 0, 0], [1, Lu, Ru, max_phys_dim])
        C2 = jnp.squeeze(C2, axis=0)  # (Lc2<->Rc1, Rc2, Pc2(o)<->Pt(i2))
        IT = jax.lax.dynamic_slice(ttn_tensors, [counter, 0, 0, 0], [1, max_phys_dim, max_phys_dim, max_phys_dim])
        IT = jnp.squeeze(IT, axis=0)  # (Pt(i1), Pt(i2), Pt(o))
        res = jnp.tensordot(C1, IT, axes=(2, 0))  # (Lc1, Rc1<->Lc2, Pt(i2)<->Pc2(o), Pt(o))
        res = jnp.tensordot(res, C2, axes=((1, 2), (0, 2)))  # (Lc1, Pt(o), Rc2)
        res = jnp.transpose(res, axes=(0, 2, 1))  # (Lc1, Rc2, Pt(o))
        contracted_tensors = contracted_tensors.at[m, :, :, :].set(res)
        counter = counter + 1
        return (contracted_tensors, ttn_tensors, counter)

    @jax.jit
    def contract_with_ttn_layer(l, args):
        contracted_tensors, ttn_tensors, counter, M = args
        args = (contracted_tensors, ttn_tensors, counter)
        (contracted_tensors, _, counter) = jax.lax.fori_loop(0, M, contract_with_ttn_tensor_jit, args)
        M = M // 2
        return (contracted_tensors, ttn_tensors, counter, M)

    contracted_tensors = jnp.zeros((N, Lu, Ru, max_phys_dim))
    args = (contracted_tensors, local_vectors, mpo_tensors)
    (contracted_tensors, _, _) = jax.lax.fori_loop(0, N, contract_phys_leg_vector_with_mpo_jit, args)

    M = int(N / 2)
    counter = 0
    args = (contracted_tensors, ttn_tensors, counter, M)
    (contracted_tensors, _, _, _) = jax.lax.fori_loop(0, no_ttn_layers, contract_with_ttn_layer, args)
    res = jnp.squeeze(jax.lax.dynamic_slice(contracted_tensors, [0, 0, 0, 0], [1, 1, 1, d_out]))
    return res





############################################################################################################
################################### FOURIER MODEL DEFINITION FUNCTIONS #####################################
############################################################################################################

def local_feature_input_vectors_padded_jax(x, N, L, max_loc_input_dim):
    '''
    Build the per-feature local input basis vectors (i.e. the local factors of e_mu(x), with
    MaxFreq = L) for one input point x with N components, orthonormalized on [-pi, pi].

    Returns a jnp array of local input vectors for the input vector x with N components.
    The shape of the returned array is (N, 1, 1, max_loc_input_dim), and in the last
    physical dimension only the first 2*L+1 components are non-zero and correspond, for
    component n, to the vector
    (1, cos(1*x[n]), ..., cos(L*x[n]), sin(1*x[n]), ..., sin(L*x[n])).

    Parameters
    ----------
    x : ndarray, shape (N,)
        Single input point.
    N : int
        Number of input features.
    L : int
        Maximum frequency per feature (MaxFreq).
    max_loc_input_dim : int
        Padded physical leg dimension (>= 2*L+1) for uniform JAX array shapes.

    Returns
    -------
    local_input_vectors : ndarray, shape (N, 1, 1, max_loc_input_dim)
    '''
    xs = jnp.expand_dims(x, axis=1)
    
    ### Define local basis (local meaning related to one feature)
    all_freqs_cos = jnp.arange(0, L+1)
    all_freqs_sin = copy.deepcopy(all_freqs_cos[1:])
    all_norms_cos = jnp.sqrt(2.0) * jnp.ones(all_freqs_cos.shape)
    all_norms_cos = all_norms_cos.at[0].divide(jnp.sqrt(2.0))
    all_norms_sin = jnp.sqrt(2.0) * jnp.ones(all_freqs_sin.shape)
    all_offsets_cos = jnp.zeros(all_freqs_cos.shape)
    all_offsets_sin = - jnp.pi * jnp.ones(all_freqs_sin.shape) / 2.0
    all_freqs = jnp.tile(jnp.concatenate((all_freqs_cos, all_freqs_sin)), (N, 1))  ## (N, 2L+1)
    all_offsets = jnp.tile(jnp.concatenate((all_offsets_cos, all_offsets_sin)), (N, 1))  ## (N, 2L+1)
    all_norms = jnp.tile(jnp.concatenate((all_norms_cos, all_norms_sin)), (N, 1))  ## (N, 2L+1)
    local_basis_ins = all_norms * jnp.cos(all_freqs * xs + all_offsets)  ### (N, 2L+1)

    local_input_vectors = jnp.zeros((N, 1, 1, max_loc_input_dim))
    local_input_vectors = local_input_vectors.at[:, 0, 0, :(2*L+1)].set(local_basis_ins)    
    return local_input_vectors



def isometric_mapping_input_vector_jax(x, N, L, max_phys_dim,
                                       ortho_mpo_tensors, ttn_tensors, maxdims_mpo,
                                       no_ttn_layers, d_out):
    '''
    Full input-side isometric map for a single input point x: build its local basis vectors
    (local_feature_input_vectors_padded_jax) and contract them with the orthogonal MPO (U) and
    TTN (T) via contract_vector_with_MPO_TTN_jaxjit, producing T @ U @ e(x) in the
    d_out-dimensional correlation space.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Single input point.
    N : int
        Number of input features.
    L : int
        Maximum frequency per feature (MaxFreq).
    max_phys_dim : int
        Padded physical leg dimension shared by MPO and local input vectors.
    ortho_mpo_tensors : ndarray
        Orthogonal MPO tensors representing U.
    ttn_tensors : ndarray
        Isometric TTN tensors representing T.
    maxdims_mpo : ndarray
        Padded (left, right) auxiliary leg dimensions of the MPO.
    no_ttn_layers : int
        Number of TTN layers, log2(N).
    d_out : int
        Output dimension of the TTN (correlation-space dimension).

    Returns
    -------
    mpo_ttn_output : ndarray, shape (d_out,)
        Isometrically mapped vector T @ U @ e(x).
    '''
    local_vectors_padded_jnp = local_feature_input_vectors_padded_jax(x, N, L, max_phys_dim)
    mpo_ttn_output = contract_vector_with_MPO_TTN_jaxjit(N, local_vectors_padded_jnp, ortho_mpo_tensors, 
                                                         ttn_tensors, maxdims_mpo, max_phys_dim, 
                                                         no_ttn_layers, d_out)
    return mpo_ttn_output




def isometric_mapping_batch_input_vectors_jax(inputs, single_input_mappung_fun):
    '''
    Batched version of isometric_mapping_input_vector_jax: vmaps the per-input isometric MPO+TTN
    mapping T @ U @ e(x) over a batch of inputs.

    Contract the basis vectors from a batch of inputs ('inputs' with shape (batch_size, N))
    with the orthogonal MPO and TTN compiled in the function 'single_input_mappung_fun' given in input.
    The isometrically mapped batch of vectors is returned as output with shape (batch_size, d_out).

    The function 'single_input_mappung_fun' can be previously jit-compiled as follows
    @jax.jit
    def single_input_mappung_fun(x):
        return fully_tensorized_model_constructor_functions.isometric_mapping_input_vector_jax(x, N, L, max_phys_dim,
                                                                                               ortho_mpo_tensors_padded_jnp,
                                                                                               ttn_tensors_jnp, maxdims_mpo,
                                                                                               no_ttn_layers, d_out)

    Parameters
    ----------
    inputs : ndarray, shape (batch_size, N)
        Batch of input points x.
    single_input_mappung_fun : callable
        JIT-compiled per-input isometric mapping function x -> T @ U @ e(x), shape (d_out,).

    Returns
    -------
    mpo_ttn_outputs : ndarray, shape (batch_size, d_out)
    '''
    inputs_mappung_fun = jax.vmap(single_input_mappung_fun)
    mpo_ttn_outputs = inputs_mappung_fun(inputs)
    return mpo_ttn_outputs




###
### Define the (Fourier) model based on fully tensorized structure constants.
### The tensor networks implementing the isometry from the input space to the
### correlation space are compiled in the function 'input_to_correl_iso_map_jaxjit'.
### The tensor train mapping the parameter space to the correlation space is given in
### input as 'V_MPS_jnp', with correlation spectrum 'Svals_jnp'.
###
### This function can be wrapped anf JIT compiled in the main as follows:
###
### local_dim_param = ...
### no_params = ...
### no_features = ...
### dim_corr_space = ...
### Svals_jnp = ...
### V_MPS_jnp = ...
### maxdims_Vtens = ...
### maxdims_Itens = ...
###
### @jax.jit
### def Fourier_model(params, inputs):
###     return fully_tensorized_Fourier_model_constructor_jax(inputs, params, local_dim_param, no_params, dim_corr_space,
###                                                           input_to_correl_iso_map_jaxjit, Svals_jnp, V_tensors_jnp, 
###                                                           maxdims_Vtens, maxdims_Itens)
### 
def fully_tensorized_Fourier_model_constructor_jax(inputs, params_jnp, local_dim_param, no_params, dim_corr_space,
                                                   input_to_correl_iso_map_jaxjit, Svals_jnp, V_tensors_jnp,
                                                   maxdims_Vtens, maxdims_Itens):
    """
    Evaluate the fully tensorized Fourier model
        f_theta(x) = (T @ U @ e(x))^T diag(S) (V^T iota(theta))
    (Eq. (22) of the paper), where both factors of Gamma = U @ diag(S) @ V^T are represented as
    tensor networks: the input-side isometry T @ U (MPO + TTN) is compiled in
    'input_to_correl_iso_map_jaxjit' (see isometric_mapping_input_vector_jax), and the
    parameter-side map V^T is a tensor train ('V_tensors_jnp', see
    tensorized_model_constructor_functions.tensorized_Fourier_model_constructor_jax).

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
    dim_corr_space : int
        Dimension of the correlation space (output dimension of the input-side isometry, i.e. d_out).
    input_to_correl_iso_map_jaxjit : callable
        JIT-compiled batched isometric input mapping (see isometric_mapping_batch_input_vectors_jax).
    Svals_jnp : ndarray, shape (dim_corr_space,)
        Correlation spectrum (singular values of Gamma).
    V_tensors_jnp : ndarray
        Tensor train representing V^T, one tensor per parameter leg.
    maxdims_Vtens, maxdims_Itens : ndarray
        Padded auxiliary/physical leg dimensions of the V tensor train and of the local
        parameter-basis vectors.

    Returns
    -------
    outputs : ndarray, shape (batch_size,)
        Model predictions f_theta(x) for each input in the batch.
    """
    ### inputs:  (batch_size, no_of_features)
    ### params:  (no_params, )

    ### Construct local tensors for local parameter basis @params
    loc_Ivecs_jnp = TN_fns_jax.local_basis_vectors_jaxjit(params_jnp, local_dim_param)

    ### Compute VI = (V^T) * Ivec,  (dim_corr_space, 1)
    VI = TN_fns_jax.contract_tensortrain_with_localvectors_jaxjit(V_tensors_jnp, loc_Ivecs_jnp, no_params, 
                                                                  maxdims_Vtens, maxdims_Itens)
    VI = jax.lax.dynamic_slice(VI, [0, 0], [dim_corr_space, 1])

    ### Compute SVI = S * (V^T) * Ivec,  (dim_corr_space, 1)
    SVI = VI * jnp.expand_dims(Svals_jnp, axis=1)

    ### Compute TUx = T_TTN * U_MPO * X(inputs),  (batch_size, dim_corr_space)
    TUx = isometric_mapping_batch_input_vectors_jax(inputs, input_to_correl_iso_map_jaxjit)

    ### Outputs
    outputs = jnp.squeeze(jnp.matmul(TUx, SVI))  ### (batch_size, )

    return outputs
    