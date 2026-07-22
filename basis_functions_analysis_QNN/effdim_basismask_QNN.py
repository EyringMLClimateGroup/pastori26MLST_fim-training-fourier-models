"""
Scan a set of QNN measurement-layer layouts and, for each one, quantify how much of the joint
input x parameter Fourier basis it can express and how that translates into effective dimension.

For a fixed QNN circuit skeleton (qnn_layout, ent_layer_layout, no_qubits), this script varies the
measurement-layer connectivity meas_layer_layout_vec (each entry giving a different amount of
correlation-spreading among the measured qubits, hence a different backward light-cone per qubit).
For every layout it:
  1. computes the boolean basis-function mask over {e_mu(x)} x {iota_nu(theta)} via
     compute_basis_functions_masks_qnns.qnn_qubits_backward_lightcones /
     basis_functions_mask_vector_qnn (i.e. which entries of the structure-constants matrix Gamma
     that layout can realize),
  2. draws no_matrix_realiz random Gamma matrices restricted to that mask, takes their SVD
     Gamma = U @ diag(S) @ V^T, and computes the correlation-spectrum purity tr(S^4),
  3. for each such Gamma, draws no_samples random parameter vectors (uniform in
     [params_min, params_max]) and computes the corresponding normalized FIM samples (via
     tensor_network_functions_np), from which it computes the average FIM purity and the
     normalized effective dimension (Eq. 12 of the paper) using eff_dim_liminf below,
  4. pickles, per measurement layout, the basis-set size (nnz_ratio = |B_X|/D), the S-purity
     tr(S^4) samples, the FIM-purity samples and the normalized effective-dimension samples, to
     results_folder, for later plotting (see plot_effdim_layouts.ipynb).

Edit the following to change the experiment: no_qubits (qubit count), qnn_layout (sequence of
'var'/'inp'/'ent' layers defining the re-uploading circuit), ent_layer_layout (entangling-layer
connectivity), meas_layer_layout_vec / name_meas_layer_layout_vec (the list of measurement-layer
connectivities to scan, and their labels), no_samples (Monte-Carlo parameter draws per Gamma
realization for the effective-dimension estimate), and no_matrix_realiz (number of random Gamma
draws per layout, i.e. per correlation-spectrum/FIM-purity statistic).
"""

# Importing necessary packages
import sys
import os
import importlib
import pickle

import numpy as np

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
import tensor_network_functions_np as TN_fns_np

import FIM_functions_jax
importlib.reload(FIM_functions_jax)

import analytical_FIM_functions_np
importlib.reload(analytical_FIM_functions_np)

import compute_basis_functions_masks_qnns
importlib.reload(compute_basis_functions_masks_qnns)
import compute_basis_functions_masks_qnns as basis_masks





### ---------------------------------------------------------------------------------------- ###
## ----------------------------------- Experiments specs ------------------------------------ ##
### ---------------------------------------------------------------------------------------- ###

# Folder in which to save results
results_folder = '/work/bd1179/b309245/fourier_models_train_and_FIM/effdim_basismask_QNN/'

# No. of qubits
no_qubits = 4
name_no_qubits = str(no_qubits)

# Layouts of QNNs
qnn_layout = ['inp', 'var', 'inp', 'var']
ent_layer_layout = 0
#meas_layer_layout_vec = [0,
#                         [(0, 1)],
#                         [(0, 1), (1, 2)], 
#                         [(0, 1), (2, 3)],
#                         [(0, 1), (1, 2), (2, 3)],
#                         1,
#                         2]
#name_meas_layer_layout_vec = ['NONE',
#                              '1PAIR',
#                              '2PAIR_NNS',
#                              '2PAIR_SEP',
#                              '3PAIR_NNS',
#                              '1PBC',
#                              '2PBC']
#
#meas_layer_layout_vec = [1,
#                         2]
#name_meas_layer_layout_vec = ['1PBC',
#                              '2PBC']
meas_layer_layout_vec = [2]
name_meas_layer_layout_vec = ['2PBC']


# No. of random parameter samples for evaluating normalized eff. dim.
no_samples = 200
no_par_samples_name = str(no_samples)

# No. of random orthogonal matrix realizations per S decay exponent
no_matrix_realiz = 50
no_V_samples_name = str(no_matrix_realiz)

# No. of layers, parameters, features
no_var_layers = 0
no_enc_layers = 0
for ll in qnn_layout:
    if (ll=='var'):
        no_var_layers = no_var_layers + 1
    if (ll=='inp'):
        no_enc_layers = no_enc_layers + 1
no_params = no_qubits * no_var_layers
no_encodings = no_qubits * no_enc_layers
max_freq = no_enc_layers
no_features = no_qubits
name_no_var_layers = str(no_var_layers)
name_no_enc_layers = str(no_enc_layers)
name_no_params = str(no_params)

# Local and global dimensions
d_loc_ins = 2 * max_freq + 1
d_loc_par = 3
name_d_loc_ins = str(d_loc_ins)
name_d_loc_par = str(d_loc_par)
D_ins = d_loc_ins ** no_features
D_pars = d_loc_par ** no_params
D_tot = D_ins * D_pars

# Bounds for the (uniformly distributed) parameters
params_min = - np.pi
params_max = + np.pi





### ---------------------------------------------------------------------------------------- ###
## ---------------------------- Function for effective dimension ---------------------------- ##
### ---------------------------------------------------------------------------------------- ###

def eff_dim_liminf(FIMs):
    """
    Monte-Carlo estimator of the normalized effective dimension d_eff (Eq. 12 of the paper) in the
    large-sample-size limit, evaluated from a batch of normalized FIM samples F_hat(theta_i).

    The exact effective-dimension formula (Abbas et al.) involves a limit n -> infinity in the
    number of data samples n entering as a constant c_n = n / (2 pi log n) multiplying each FIM;
    here this limit is approximated by fixing a single large constant cn = 1e12 and computing
    2 * (logsumexp_i(0.5 * logdet(I + cn * F_hat(theta_i))) - log(no_samples)) / log(cn), which
    converges to d_eff (in [0, M], M = no. of parameters) as cn -> infinity. Averaging is done via
    logsumexp over the no_samples random parameter draws for numerical stability.

    Parameters
    ----------
    FIMs : ndarray, shape (no_samples, no_params, no_params)
        Batch of normalized FIM samples F_hat(theta_i), one per random parameter draw theta_i.

    Returns
    -------
    effdim : float
        Monte-Carlo estimate of the (un-normalized, i.e. not yet divided by no_params) effective
        dimension in the large-cn limit.
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

for nmeas in range(len(meas_layer_layout_vec)):
    meas_layer_layout = meas_layer_layout_vec[nmeas]
    name_meas_layer_layout = name_meas_layer_layout_vec[nmeas]

    BWLC_dict, params_dict, BWLC_ins_dict, inputs_dict = basis_masks.qnn_qubits_backward_lightcones(no_qubits, qnn_layout, 
                                                                                                    ent_layer_layout, meas_layer_layout)

    basis_mask = basis_masks.basis_functions_mask_vector_qnn(no_qubits, BWLC_dict, BWLC_ins_dict, no_params, 
                                                             d_loc_par, no_features, d_loc_ins)
    gamma_mask = np.reshape(basis_mask, (D_ins, D_pars))
    nnz_ratio = np.sum(basis_mask) / D_tot

    name_end = ('_NoQubits' + name_no_qubits + '_NoParams' + name_no_params + '_NoVarLayers' + name_no_var_layers + '_NoEncLayers' + name_no_enc_layers + 
                '_NsamplesPar' + no_par_samples_name + '_NsamplesV' + no_V_samples_name + '__MeasLayout' + name_meas_layer_layout)

    meas_layer_layout_all = []
    nnz_ratio_all = []
    S_purity_all = []
    avg_FIM_purity_all = []
    norm_eff_dim_all = []
    
    ### Loop over random model draws
    for nm in range(no_matrix_realiz):
        Gamma = 2.0 * np.random.rand(D_ins, D_pars) - 1.0
        Gamma = Gamma * gamma_mask

        ### if matrix is long with large max. dim., SVD is inefficient
        ### and it is convenient to diagonalize G * G.T if A long
        ### G = U S V.T  ==>  G.T = V S U.T  ==>  G*G.T = U S^2 U.T
        ### S V.T = U.T G  ==>  V.T = inv(S) U.T G
        if (D_pars>6000 and D_ins<2000):
            G2 = np.matmul(Gamma, np.transpose(Gamma))
            S2, U = np.linalg.eigh(G2)
            III = np.argsort(S2)
            S2 = np.real(S2[III])
            U = U[:, III]
            ### check if there are eigenvalues very close to 0, if so, remove them
            III = (S2 > 1.0e-08)
            S2 = S2[III]
            U = U[:, III]
            S = np.sqrt(S2)
            ### Get Vt matrix
            Vh = np.matmul(np.matmul(np.diag(1.0/S), np.transpose(U)), Gamma)
        else:
            U, S, Vh = np.linalg.svd(Gamma, full_matrices=True)

        ### Calculate S purity
        S_normalized = S / np.sqrt(np.sum(S**2.0))
        purity_S = np.sum(S_normalized ** 4.0)

        ### Decompose in local tensors
        tensors_list = TN_fns_np.tensortrain_from_ortho_matrix(Vh, no_params, d_loc_par)

        ### Loop over random parameter samples
        nFIMs = []
        FIM_purities = []
        for ns in range(no_samples):
            params = (params_max - params_min) * np.random.rand(no_params) + params_min

            ### Compute FIM
            nFIM = TN_fns_np.normalized_FIM_sample(params, no_params, d_loc_par, S, tensors_list)
            nFIMs.append(nFIM)

            ### Compute FIM purity
            evals, _ = np.linalg.eig(nFIM)
            evals = np.real(evals)
            evals = evals / np.sum(evals)
            pur_FIM = np.sum(evals ** 2.0)
            FIM_purities.append(pur_FIM)
            
        nFIMs = np.asarray(nFIMs)
        FIM_purities = np.asarray(FIM_purities)
        avg_FIM_pur = np.mean(FIM_purities)
        nED = eff_dim_liminf(nFIMs) / no_params
        
        meas_layer_layout_all.append(meas_layer_layout)
        nnz_ratio_all.append(nnz_ratio)
        S_purity_all.append(purity_S)
        avg_FIM_purity_all.append(avg_FIM_pur)
        norm_eff_dim_all.append(nED)

    nnz_ratio_all = np.asarray(nnz_ratio_all)
    S_purity_all = np.asarray(S_purity_all)
    avg_FIM_purity_all = np.asarray(avg_FIM_purity_all)
    norm_eff_dim_all = np.asarray(norm_eff_dim_all)

    dict_norm_ed = dict()
    dict_norm_ed['meas_layer_layout_all'] = meas_layer_layout_all
    dict_norm_ed['nnz_ratio_all'] = nnz_ratio_all
    dict_norm_ed['S_purity_all'] = S_purity_all
    dict_norm_ed['avg_FIM_purity_all'] = avg_FIM_purity_all
    dict_norm_ed['norm_eff_dim_all'] = norm_eff_dim_all

    filename = 'norm_eff_dim' + name_end + '.pkl'
    path_file = os.path.join(results_folder, filename)
    with open(path_file, 'wb') as f:
        pickle.dump(dict_norm_ed, f)
    
    print(' ******** Saved results')
    print(' ')