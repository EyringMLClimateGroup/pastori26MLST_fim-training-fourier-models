"""
Analytical (closed-form) computation of the Fisher Information Matrix (FIM) for a
partial Fourier series model built from the structure constants Gamma = U @ diag(Svals) @ Vh.

Following the paper's notation, the model is f_theta(x) = sum_mu c_mu(theta) e_mu(x), with
c_mu(theta) = sum_nu Gamma_{mu,nu} iota_nu(theta), and the parameter-space basis functions
iota_nu(theta) built out of the local per-parameter basis {1, cos(theta), ..., sin(L*theta)}
(local dimension d_tilde = local_basis_dim). These routines evaluate the FIM
F_{j,k}(theta) = E_x[df_theta/dtheta_j * df_theta/dtheta_k] and its trace directly from the
singular values (correlation spectrum) Svals and right-singular-vector matrix Vh of Gamma,
without going through a tensor-network contraction. They are intended for small numbers of
parameters/features, where the (D x K) matrices involved can be built and multiplied explicitly.
"""

import numpy as np




###
### Defines the (local_basis_dim x local_basis_dim) derivative tensor for
### a local space with basis {1, cos(th), ..., cos(L*th), sin(th), ..., sin(L*th)}
### with L = (local_basis_dim - 1)/2.
###
def derivative_tensor(local_basis_dim):
    """
    Build the matrix representation of d/d(theta) acting on the local per-parameter
    trigonometric basis {1, cos(theta), ..., cos(L*theta), sin(theta), ..., sin(L*theta)},
    with L = (local_basis_dim - 1) / 2.

    Since d[cos(k*theta)]/d(theta) = -k*sin(k*theta) and d[sin(k*theta)]/d(theta) = k*cos(k*theta),
    this is a skew-symmetric (local_basis_dim x local_basis_dim) matrix with the off-diagonal
    blocks holding k=1,...,L. It is the local building block used (via Kronecker products) to
    differentiate the parameter-space basis functions iota_nu(theta) in the FIM computation.

    Parameters
    ----------
    local_basis_dim : int
        Dimension d_tilde of the local per-parameter basis (must be odd, 2L+1).

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
### FIM for given parameters.
###
def FIM_sample_fullmatrices(basis_funcs_th, no_params, local_basis_dim, Svals, Vh):
    """
    Evaluate the (no_params x no_params) Fisher Information Matrix F(theta) at a single
    parameter configuration theta, given the parameter-space basis functions iota_nu(theta)
    already evaluated at that point (basis_funcs_th) and the correlation-spectrum decomposition
    Gamma = U @ diag(Svals) @ Vh.

    For each pair of parameters (j, k) this differentiates iota(theta) with respect to
    theta_j and theta_k (using the local derivative_tensor B on the corresponding leg of the
    Kronecker-product parameter basis) and contracts with S^2 = diag(Svals**2) through Vh, i.e.
    F_{j,k}(theta) = (d iota/d theta_j)^T Vh^T S^2 Vh (d iota/d theta_k), matching
    F_{j,k}(theta) = E_x[df_theta/dtheta_j * df_theta/dtheta_k] of the paper.

    Parameters
    ----------
    basis_funcs_th : ndarray, shape (K, 1)
        Parameter-space basis functions iota_nu(theta) evaluated at the parameter point theta.
    no_params : int
        Number of trainable parameters M.
    local_basis_dim : int
        Local per-parameter basis dimension d_tilde.
    Svals : ndarray, shape (D,)
        Correlation spectrum (singular values s_rho of Gamma).
    Vh : ndarray, shape (D, K)
        Right-singular-vector matrix of Gamma (K = local_basis_dim**no_params).

    Returns
    -------
    FIM : ndarray, shape (no_params, no_params)
        Fisher Information Matrix F(theta).
    """
    D = Vh.shape[1]
    Svals2 = np.diag(Svals ** 2.0)
    B = derivative_tensor(local_basis_dim)
    FIM = np.zeros((no_params, no_params))
    for j in range(no_params-1):
        Bj = 1.0
        for jp in range(no_params):
            if (jp==j):
                Bj = np.kron(Bj, B)
            else:
                Bj = np.kron(Bj, np.eye(local_basis_dim))
        Vj = np.matmul(Vh, Bj)  # (Din, D)
        Vj = np.matmul(Vj, basis_funcs_th)  # (Din, 1)
        for k in range(j, no_params):
            Bk = 1.0
            for kp in range(no_params):
                if (kp==k):
                    Bk = np.kron(Bk, B)
                else:
                    Bk = np.kron(Bk, np.eye(local_basis_dim))
            Vk = np.matmul(Vh, Bk)  # (Din, D)
            Vk = np.matmul(Vk, basis_funcs_th)  # (Din, 1)
            SVk = np.matmul(Svals2, Vk)  # (Din, 1)
            Fjk = np.squeeze(np.matmul(np.transpose(Vj), SVk))
            FIM[j,k] = Fjk
            FIM[k,j] = Fjk
    j = no_params - 1
    Bj = 1.0
    for jp in range(no_params):
        if (jp==j):
            Bj = np.kron(Bj, B)
        else:
            Bj = np.kron(Bj, np.eye(local_basis_dim))
    Vj = np.matmul(Vh, Bj)  # (Din, D)
    Vj = np.matmul(Vj, basis_funcs_th)  # (Din, 1)
    SVj = np.matmul(Svals2, Vj)  # (Din, 1)
    Fjj = np.squeeze(np.matmul(np.transpose(Vj), SVj))
    FIM[j,j] = Fjj
    return FIM




###
### Expectation value over param space of trace of FIM.
###
def expected_trace_FIM_fullmatrices(no_params, local_basis_dim, Svals, Vh):
    """
    Compute E_theta[tr(F(theta))], the parameter-space average of the trace of the Fisher
    Information Matrix, in closed form from the correlation spectrum Svals and Vh
    (Gamma = U @ diag(Svals) @ Vh). This is the normalization tr(F(theta)) used to build the
    normalized FIM F_hat(theta) = (no_params / tr(F(theta))) * F(theta) of Eq. (13) in the paper,
    which removes overall scale so that only the shape of the FIM spectrum (and hence the
    effective dimension) is compared across models.

    Parameters
    ----------
    no_params : int
        Number of trainable parameters M.
    local_basis_dim : int
        Local per-parameter basis dimension d_tilde.
    Svals : ndarray, shape (D,)
        Correlation spectrum (singular values of Gamma).
    Vh : ndarray, shape (D, K)
        Right-singular-vector matrix of Gamma.

    Returns
    -------
    trFIM : float
        Expected trace of the (unnormalized) FIM, averaged over the parameter-space measure.
    """
    D = Vh.shape[1]
    Svals2 = np.diag(Svals ** 2.0)
    B = derivative_tensor(local_basis_dim)
    trFIM = 0.0
    for j in range(no_params):
        Bj = 1.0
        for jp in range(no_params):
            if (jp==j):
                Bj = np.kron(Bj, B)
            else:
                Bj = np.kron(Bj, np.eye(local_basis_dim))
        Vj = np.matmul(Vh, Bj)  # (Din, D)
        Vj2 = np.matmul(Vj, np.transpose(Vj))  # (Din, Din)
        SVj2 = np.matmul(Svals2, Vj2)  # (Din, 1)
        Fjj = np.squeeze(np.trace(SVj2))
        trFIM = trFIM + Fjj
    return trFIM




###
### Normalized FIM for given parameters.
###
def normalized_FIM_sample_fullmatrices(basis_funcs_th, no_params, local_basis_dim, Svals, Vh):
    """
    Evaluate the normalized Fisher Information Matrix F_hat(theta) = (M / tr(F(theta))) * F(theta)
    at a single parameter point theta (Eq. (13) of the paper), combining
    FIM_sample_fullmatrices and expected_trace_FIM_fullmatrices. The eigenvalue spectrum of
    F_hat(theta), averaged/integrated over theta, is what enters the effective-dimension
    formula (Eq. (12)).

    Parameters
    ----------
    basis_funcs_th : ndarray, shape (K, 1)
        Parameter-space basis functions iota_nu(theta) evaluated at theta.
    no_params : int
        Number of trainable parameters M.
    local_basis_dim : int
        Local per-parameter basis dimension d_tilde.
    Svals : ndarray, shape (D,)
        Correlation spectrum (singular values of Gamma).
    Vh : ndarray, shape (D, K)
        Right-singular-vector matrix of Gamma.

    Returns
    -------
    normalized_FIM : ndarray, shape (no_params, no_params)
        Normalized FIM F_hat(theta).
    """
    FIM = FIM_sample_fullmatrices(basis_funcs_th, no_params, local_basis_dim, Svals, Vh)
    exp_trF = expected_trace_FIM_fullmatrices(no_params, local_basis_dim, Svals, Vh)
    return no_params * FIM / exp_trF
