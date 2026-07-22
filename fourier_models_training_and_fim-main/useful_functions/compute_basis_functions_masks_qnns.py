"""
Routines connecting a qubit-based Quantum Neural Network's (QNN) circuit layout to the subset of
Fourier basis functions it can actually express.

As shown in the paper, a data re-uploading QNN's expectation-value output is itself a partial
Fourier series f_theta(x) = sum_{omega,omega_tilde} Gamma_tilde_{omega,omega_tilde} e^{i omega x}
e^{i omega_tilde theta} (Eq. (6)), but not every combination of input frequencies omega and
parameter frequencies omega_tilde is reachable: which ones are depends on which qubits each
input/parameter angle is encoded on, and how entangling layers spread that dependence to the
measured qubit(s) (i.e. each qubit's causal "backward light-cone" of gates). This module first
computes, for a given QNN layout (sequence of variational/input-encoding/entangling layers), the
backward light-cone of each qubit -- which variational parameters and which encoded input
features can influence it -- and then converts these light-cones into "basis function masks":
boolean vectors over the tensor-product basis {e_mu(x)} x {iota_nu(theta)} (or over each factor
separately) marking which basis functions have a nonzero coefficient in the QNN's Fourier
expansion. These masks are the practical tool used to restrict a generic Fourier/tensor-network
model to the subspace actually expressible by a given QNN architecture, and hence to compute
its FIM/effective dimension.
"""

import numpy as np





def qnn_qubits_backward_lightcones(no_qubits, qnn_layout, ent_layer_layout, meas_layer_layout):
    '''
    Trace, for a given QNN circuit layout, which variational parameters and which encoded input
    features can causally influence (are in the "backward light-cone" of) each qubit that is
    ultimately measured. This determines which parameter/input frequencies omega_tilde/omega a
    given qubit's expectation value can contain in its Fourier expansion (Eq. (6) of the paper):
    only parameters/features within a qubit's backward light-cone can contribute a nonzero
    Fourier coefficient for that qubit's output.

    Returns dictionaries containing:
    - Variational parameters IDs together with set of qubits whose backward light-cone
      contains the paramerer with given ID
    - Backward light-cone (for parameters) for each qubit
    - Encoded features' IDs together with set of qubits whose backward light-cone
      contains the encoded feature with given ID
    - Backward light-cone (for input features, together with multiplicity) for each qubit

    'qnn_layout' is a list containing the layers of the QNN identified as follows:
    - 'var': variational layer
    - 'inp': input encoding layer
    - 'ent': entangling layer
    Implicitly, at the end a measurement layer is performed.

    'ent_layer_layout' and 'meas_layer_layout' contain the layout of the entangling and measurement
    layers, respectively, according to the following convention:
    - integer values K such as 0, 1, 2, ... denote K layers of 2-qubit gates acting on NN with PBC.
    - a list of 2-ples (n1, n2) denote the pairs of qubits acted upon by entangling gates.

    Parameters
    ----------
    no_qubits : int
        Number of qubits in the QNN.
    qnn_layout : list of str
        Sequence of layers ('var', 'inp', 'ent') defining the circuit (a re-uploading QNN
        interleaves 'inp' and 'var'/'ent' layers).
    ent_layer_layout : int or list of (int, int) tuples
        Entangling-layer connectivity: an int K means K layers of nearest-neighbor (periodic)
        2-qubit gates; a list of pairs gives explicit qubit pairs acted on.
    meas_layer_layout : int or list of (int, int) tuples
        Same convention as ent_layer_layout, but describing spreading of correlations by a final
        (implicit) measurement/entangling stage.

    Returns
    -------
    BWLC_dict : dict[int, list[int]]
        For each qubit, the list of variational-parameter IDs in its backward light-cone.
    params_dict : dict[int, dict]
        For each parameter ID, dict with key 'in_BWLC_of' listing the qubits it influences.
    BWLC_ins_dict : dict[int, dict]
        For each qubit, dict with keys 'ins' (list of encoded-feature names, e.g. 'in0') and
        'counts' (multiplicity, i.e. number of encoding layers/re-uploads of that feature
        reaching the qubit -- sets the maximum frequency L reachable for that feature).
    inputs_dict : dict[str, dict]
        For each encoded feature+layer name (e.g. 'in0_l1'), dict with key 'in_BWLC_of' listing
        the qubits it influences.
    '''
    no_var_layers = 0
    no_enc_layers = 0
    for ll in qnn_layout:
        if (ll=='var'):
            no_var_layers = no_var_layers + 1
        if (ll=='inp'):
            no_enc_layers = no_enc_layers + 1
    no_params = no_qubits * no_var_layers
    no_encodings = no_qubits * no_enc_layers

    first_var_layer = len(qnn_layout)
    first_enc_layer = len(qnn_layout)
    for l in range(len(qnn_layout)):
        ll = qnn_layout[l]
        if (ll=='var' and l<first_var_layer):
            first_var_layer = l
        if (ll=='inp' and l<first_enc_layer):
            first_enc_layer = l

    
    ### Dictionary of parameters (with field containing which qubits' 
    ### backwards light-cone they belong to)
    params_dict = dict()
    for p in range(no_params):
        par_dict = dict()
        par_dict['in_BWLC_of'] = []
        params_dict[p] = par_dict
    cvl = 0
    for l in range(first_var_layer, len(qnn_layout)):
        ll = qnn_layout[l]
        if (ll=='var'):
            for q in range(no_qubits):
                n_par = cvl*no_qubits + q
                par_dict = params_dict[n_par]
                par_dict['in_BWLC_of'].append(q)
                params_dict[n_par] = par_dict
            cvl = cvl + 1
        if (ll=='ent'):
            if (type(ent_layer_layout) is list):
                for ccl in range(0,cvl):
                    for q in range(no_qubits):
                        n_par = ccl*no_qubits + q
                        par_dict = params_dict[n_par]
                        in_BWLC_of = par_dict['in_BWLC_of']
                        curr_domain_size = len(in_BWLC_of)
                        for j in range(curr_domain_size):
                            qq = in_BWLC_of[j]
                            for pair in ent_layer_layout:
                                if (qq in pair):
                                    q1 = pair[0]
                                    q2 = pair[1]
                                    if (q1 not in in_BWLC_of):
                                        in_BWLC_of.append(q1)
                                    if (q2 not in in_BWLC_of):
                                        in_BWLC_of.append(q2)
                        par_dict['in_BWLC_of'] = in_BWLC_of
                        params_dict[n_par] = par_dict
            else:
                depth_ent = ent_layer_layout
                for ccl in range(0,cvl):
                    for q in range(no_qubits):
                        n_par = ccl*no_qubits + q
                        par_dict = params_dict[n_par]
                        in_BWLC_of = par_dict['in_BWLC_of']
                        curr_domain_size = len(in_BWLC_of)
                        for j in range(curr_domain_size):
                            qq = in_BWLC_of[j]
                            for nn in range(depth_ent + 1):
                                qp1 = np.mod((qq + nn), no_qubits)
                                qm1 = np.mod((qq - nn), no_qubits)
                                if (qp1 not in in_BWLC_of):
                                    in_BWLC_of.append(qp1)
                                if (qm1 not in in_BWLC_of):
                                    in_BWLC_of.append(qm1)
                        par_dict['in_BWLC_of'] = in_BWLC_of
                        params_dict[n_par] = par_dict
    if (type(meas_layer_layout) is list):
        for p in range(no_params):
            par_dict = params_dict[p]
            in_BWLC_of = par_dict['in_BWLC_of']
            curr_domain_size = len(in_BWLC_of)
            for j in range(curr_domain_size):
                qq = in_BWLC_of[j]
                for pair in meas_layer_layout:
                    if (qq in pair):
                        q1 = pair[0]
                        q2 = pair[1]
                        if (q1 not in in_BWLC_of):
                            in_BWLC_of.append(q1)
                        if (q2 not in in_BWLC_of):
                            in_BWLC_of.append(q2)
            par_dict['in_BWLC_of'] = in_BWLC_of
            params_dict[p] = par_dict
    else:
        depth_meas = meas_layer_layout
        for p in range(no_params):
            par_dict = params_dict[p]
            in_BWLC_of = par_dict['in_BWLC_of']
            curr_domain_size = len(in_BWLC_of)
            for j in range(curr_domain_size):
                q = in_BWLC_of[j]
                for nn in range(depth_meas + 1):
                    qp1 = np.mod((q + nn), no_qubits)
                    qm1 = np.mod((q - nn), no_qubits)
                    if (qp1 not in in_BWLC_of):
                        in_BWLC_of.append(qp1)
                    if (qm1 not in in_BWLC_of):
                        in_BWLC_of.append(qm1)
            par_dict['in_BWLC_of'] = in_BWLC_of
            params_dict[p] = par_dict

    
    ### Dictionary of encoded inputs (with field containing which qubits' 
    ### backwards light-cone they belong to)
    inputs_dict = dict()
    for l in range(no_enc_layers):
        for q in range(no_qubits):
            name_in = 'in' + str(q) + '_l' + str(l)
            in_dict = dict()
            in_dict['in_BWLC_of'] = []
            inputs_dict[name_in] = in_dict
    cel = 0
    for l in range(first_enc_layer, len(qnn_layout)):
        ll = qnn_layout[l]
        if (ll=='inp'):
            for q in range(no_qubits):
                name_in = 'in' + str(q) + '_l' + str(cel)
                in_dict = inputs_dict[name_in]
                in_dict['in_BWLC_of'].append(q)
                inputs_dict[name_in] = in_dict
            cel = cel + 1
        if (ll=='ent'):
            if (type(ent_layer_layout) is list):
                for ccl in range(0,cel):
                    for q in range(no_qubits):
                        name_in = 'in' + str(q) + '_l' + str(ccl)
                        in_dict = inputs_dict[name_in]
                        in_BWLC_of = in_dict['in_BWLC_of']
                        curr_domain_size = len(in_BWLC_of)
                        for j in range(curr_domain_size):
                            qq = in_BWLC_of[j]
                            for pair in ent_layer_layout:
                                if (qq in pair):
                                    q1 = pair[0]
                                    q2 = pair[1]
                                    if (q1 not in in_BWLC_of):
                                        in_BWLC_of.append(q1)
                                    if (q2 not in in_BWLC_of):
                                        in_BWLC_of.append(q2)
                        in_dict['in_BWLC_of'] = in_BWLC_of
                        inputs_dict[name_in] = in_dict                            
            else:
                depth_ent = ent_layer_layout
                for ccl in range(0,cel):
                    for q in range(no_qubits):
                        name_in = 'in' + str(q) + '_l' + str(ccl)
                        in_dict = inputs_dict[name_in]
                        in_BWLC_of = in_dict['in_BWLC_of']
                        curr_domain_size = len(in_BWLC_of)
                        for j in range(curr_domain_size):
                            qq = in_BWLC_of[j]
                            for nn in range(depth_ent + 1):
                                qp1 = np.mod((qq + nn), no_qubits)
                                qm1 = np.mod((qq - nn), no_qubits)
                                if (qp1 not in in_BWLC_of):
                                    in_BWLC_of.append(qp1)
                                if (qm1 not in in_BWLC_of):
                                    in_BWLC_of.append(qm1)
                        in_dict['in_BWLC_of'] = in_BWLC_of
                        inputs_dict[name_in] = in_dict
    if (type(meas_layer_layout) is list):
        for l in range(no_enc_layers):
            for q in range(no_qubits):
                name_in = 'in' + str(q) + '_l' + str(l)
                in_dict = inputs_dict[name_in]
                in_BWLC_of = in_dict['in_BWLC_of']
                curr_domain_size = len(in_BWLC_of)
                for j in range(curr_domain_size):
                    qq = in_BWLC_of[j]
                    for pair in meas_layer_layout:
                        if (qq in pair):
                            q1 = pair[0]
                            q2 = pair[1]
                            if (q1 not in in_BWLC_of):
                                in_BWLC_of.append(q1)
                            if (q2 not in in_BWLC_of):
                                in_BWLC_of.append(q2)
                in_dict['in_BWLC_of'] = in_BWLC_of
                inputs_dict[name_in] = in_dict
    else:
        depth_meas = meas_layer_layout
        for l in range(no_enc_layers):
            for q in range(no_qubits):
                name_in = 'in' + str(q) + '_l' + str(l)
                in_dict = inputs_dict[name_in]
                in_BWLC_of = in_dict['in_BWLC_of']
                curr_domain_size = len(in_BWLC_of)
                for j in range(curr_domain_size):
                    qj = in_BWLC_of[j]
                    for nn in range(depth_meas + 1):
                        qp1 = np.mod((qj + nn), no_qubits)
                        qm1 = np.mod((qj - nn), no_qubits)
                        if (qp1 not in in_BWLC_of):
                            in_BWLC_of.append(qp1)
                        if (qm1 not in in_BWLC_of):
                            in_BWLC_of.append(qm1)
                in_dict['in_BWLC_of'] = in_BWLC_of
                inputs_dict[name_in] = in_dict

    
    ### Dictionary backwards light-cone for parameters (contains
    ### the BWLC of each qubit)
    BWLC_dict = dict()
    for q in range(no_qubits):
        BWLC_dict[q] = []
    for p in range(no_params):
        par_dict = params_dict[p]
        in_BWLC_of = par_dict['in_BWLC_of']
        for q in in_BWLC_of:
            if (p not in BWLC_dict[q]):
                BWLC_dict[q].append(p)

    ### Dictionary backwards light-cone for input encodings (contains
    ### the BWLC of each qubit)
    BWLC_ins_dict_0 = dict()
    for q in range(no_qubits):
        BWLC_ins_dict_0[q] = []
    for l in range(no_enc_layers):
        for q in range(no_qubits):
            name_in = 'in' + str(q) + '_l' + str(l)
            in_dict = inputs_dict[name_in]
            in_BWLC_of = in_dict['in_BWLC_of']
            for qb in in_BWLC_of:
                if (name_in not in BWLC_ins_dict_0[qb]):
                    BWLC_ins_dict_0[qb].append(name_in)
    BWLC_ins_dict = dict()
    for q in range(no_qubits):
        BWLC_q_0 = BWLC_ins_dict_0[q]
        BWLC_q = []
        for ib0 in BWLC_q_0:
            ib = ib0[:-3]
            if (ib not in BWLC_q):
                BWLC_q.append(ib)
        counts_q = [0 for _ in range(len(BWLC_q))]
        for i in range(len(BWLC_q)):
            ib = BWLC_q[i]
            for ib0 in BWLC_q_0:
                if ib in ib0:
                    counts_q[i] = counts_q[i] + 1
        BWLC_dict_q = dict()
        BWLC_dict_q['ins'] = BWLC_q
        BWLC_dict_q['counts'] = counts_q
        BWLC_ins_dict[q] = BWLC_dict_q

    return BWLC_dict, params_dict, BWLC_ins_dict, inputs_dict



def basis_functions_mask_vector_qubit_qnn(j, BWLC_params, BWLC_inputs, no_params, d_loc_pars, no_features, d_loc_ins):
    """
    Build the boolean mask, over the full tensor-product basis {e_mu(x)} x {iota_nu(theta)} for
    qubit j, marking which basis functions have a nonzero coefficient in that qubit's Fourier
    expansion. A parameter's local basis leg is masked "all frequencies allowed" if the
    parameter is in qubit j's backward light-cone (BWLC_params[j]) and "constant only" (mask =
    [1,0,...,0]) otherwise; an input feature's local basis leg is masked up to the frequency L_i
    given by how many times that feature reaches qubit j (its re-upload multiplicity in
    BWLC_inputs[j]) if the feature is in the light-cone, and "constant only" otherwise. The full
    mask is the Kronecker product of these per-parameter/per-feature local masks.

    Parameters
    ----------
    j : int
        Qubit index.
    BWLC_params : dict[int, list[int]]
        Backward light-cone of variational parameters for each qubit (from
        qnn_qubits_backward_lightcones).
    BWLC_inputs : dict[int, dict]
        Backward light-cone of encoded features (with multiplicities) for each qubit.
    no_params : int
        Number of trainable parameters M.
    d_loc_pars : int
        Local per-parameter basis dimension d_tilde.
    no_features : int
        Number of input features N.
    d_loc_ins : int
        Local per-feature input basis dimension d.

    Returns
    -------
    basis_mask : ndarray of int (0/1), shape (d_loc_ins**no_features * d_loc_pars**no_params,)
        Mask over the full input x parameter basis for qubit j's contribution.
    """
    Lmax = int((d_loc_ins - 1) / 2)
    
    BWLC_j = BWLC_params[j]
    basis_params_mask = 1
    for p in range(no_params):
        if (p in BWLC_j):
            loc_p_mask = np.ones(d_loc_pars, dtype=np.int32)
        else:
            loc_p_mask = np.zeros(d_loc_pars, dtype=np.int32)
            loc_p_mask[0] = 1
        basis_params_mask = np.kron(basis_params_mask, loc_p_mask)
    
    BWLC_ins_j = BWLC_inputs[j]
    basis_inputs_mask = 1
    for f in range(no_features):
        name_f = 'in' + str(f)
        if (name_f in BWLC_ins_j['ins']):
            ind_i = BWLC_ins_j['ins'].index(name_f)
            L_i = BWLC_ins_j['counts'][ind_i]
            loc_in_mask = np.zeros(d_loc_ins, dtype=np.int32)
            loc_in_mask[0] = 1
            loc_in_mask[1:(1 + L_i)] = np.ones(L_i, dtype=np.int32)
            loc_in_mask[(1 + Lmax):(1 + Lmax + L_i)] = np.ones(L_i, dtype=np.int32)
        else:
            loc_in_mask = np.zeros(d_loc_ins, dtype=np.int32)
            loc_in_mask[0] = 1
        basis_inputs_mask = np.kron(basis_inputs_mask, loc_in_mask)

    basis_mask = np.kron(basis_inputs_mask, basis_params_mask)
    return basis_mask



def basis_functions_mask_vector_qnn(no_qubits, BWLC_params, BWLC_inputs, no_params, d_loc_pars, no_features, d_loc_ins):
    """
    Full-QNN basis function mask: the logical OR, over all measured qubits, of each qubit's mask
    from basis_functions_mask_vector_qubit_qnn. Marks which basis functions of the tensor-product
    input x parameter basis are expressible by the QNN as a whole (e.g. when summing/averaging
    single-qubit expectation values into the QNN output), i.e. which entries of the structure
    constants Gamma_tilde can be nonzero.

    Parameters
    ----------
    no_qubits : int
        Number of (measured) qubits in the QNN.
    BWLC_params, BWLC_inputs : dict
        Backward light-cones for parameters and inputs (from qnn_qubits_backward_lightcones).
    no_params : int
        Number of trainable parameters M.
    d_loc_pars : int
        Local per-parameter basis dimension d_tilde.
    no_features : int
        Number of input features N.
    d_loc_ins : int
        Local per-feature input basis dimension d.

    Returns
    -------
    basis_mask : ndarray of bool, shape (d_loc_ins**no_features * d_loc_pars**no_params,)
        Mask over the full input x parameter basis for the whole QNN.
    """
    D_ins = d_loc_ins ** no_features
    D_pars = d_loc_pars ** no_params
    D_tot = D_ins * D_pars
    basis_mask = np.zeros(D_tot, dtype=np.int32)
    for j in range(no_qubits):
        basis_mask_j = basis_functions_mask_vector_qubit_qnn(j, BWLC_params, BWLC_inputs, no_params, 
                                                             d_loc_pars, no_features, d_loc_ins)
        basis_mask = np.logical_or(basis_mask, basis_mask_j)
    return basis_mask




def inputs_params_basis_mask_vectors_qubits_qnn(j, BWLC_params, BWLC_inputs, no_params, d_loc_pars, no_features, d_loc_ins):
    """
    Same computation as basis_functions_mask_vector_qubit_qnn, but returns the input-space mask
    (over e_mu(x)) and parameter-space mask (over iota_nu(theta)) for qubit j separately, instead
    of their combined Kronecker product. Useful when the input and parameter basis masks need to
    be applied independently (e.g. to mask a tensor-network model's U/V factors separately
    rather than the dense joint basis).

    Parameters
    ----------
    j : int
        Qubit index.
    BWLC_params, BWLC_inputs : dict
        Backward light-cones for parameters and inputs (from qnn_qubits_backward_lightcones).
    no_params : int
        Number of trainable parameters M.
    d_loc_pars : int
        Local per-parameter basis dimension d_tilde.
    no_features : int
        Number of input features N.
    d_loc_ins : int
        Local per-feature input basis dimension d.

    Returns
    -------
    basis_inputs_mask : ndarray of int (0/1), shape (d_loc_ins**no_features,)
        Mask over the input basis e_mu(x) for qubit j.
    basis_params_mask : ndarray of int (0/1), shape (d_loc_pars**no_params,)
        Mask over the parameter basis iota_nu(theta) for qubit j.
    """
    Lmax = int((d_loc_ins - 1) / 2)
    
    BWLC_j = BWLC_params[j]
    basis_params_mask = 1
    for p in range(no_params):
        if (p in BWLC_j):
            loc_p_mask = np.ones(d_loc_pars, dtype=np.int32)
        else:
            loc_p_mask = np.zeros(d_loc_pars, dtype=np.int32)
            loc_p_mask[0] = 1
        basis_params_mask = np.kron(basis_params_mask, loc_p_mask)
    
    BWLC_ins_j = BWLC_inputs[j]
    basis_inputs_mask = 1
    for f in range(no_features):
        name_f = 'in' + str(f)
        if (name_f in BWLC_ins_j['ins']):
            ind_i = BWLC_ins_j['ins'].index(name_f)
            L_i = BWLC_ins_j['counts'][ind_i]
            loc_in_mask = np.zeros(d_loc_ins, dtype=np.int32)
            loc_in_mask[0] = 1
            loc_in_mask[1:(1 + L_i)] = np.ones(L_i, dtype=np.int32)
            loc_in_mask[(1 + Lmax):(1 + Lmax + L_i)] = np.ones(L_i, dtype=np.int32)
        else:
            loc_in_mask = np.zeros(d_loc_ins, dtype=np.int32)
            loc_in_mask[0] = 1
        basis_inputs_mask = np.kron(basis_inputs_mask, loc_in_mask)

    return basis_inputs_mask, basis_params_mask



def inputs_params_basis_mask_vectors_qnn(no_qubits, BWLC_params, BWLC_inputs, no_params, d_loc_pars, no_features, d_loc_ins):
    """
    Full-QNN version of inputs_params_basis_mask_vectors_qubits_qnn: the logical OR, over all
    measured qubits, of each qubit's separate input-space and parameter-space masks. Gives the
    subset of the input basis {e_mu(x)} and of the parameter basis {iota_nu(theta)} that the QNN
    architecture as a whole can access, independently of one another.

    Parameters
    ----------
    no_qubits : int
        Number of (measured) qubits in the QNN.
    BWLC_params, BWLC_inputs : dict
        Backward light-cones for parameters and inputs (from qnn_qubits_backward_lightcones).
    no_params : int
        Number of trainable parameters M.
    d_loc_pars : int
        Local per-parameter basis dimension d_tilde.
    no_features : int
        Number of input features N.
    d_loc_ins : int
        Local per-feature input basis dimension d.

    Returns
    -------
    basis_inputs_mask : ndarray of bool, shape (d_loc_ins**no_features,)
        Mask over the input basis e_mu(x) for the whole QNN.
    basis_params_mask : ndarray of bool, shape (d_loc_pars**no_params,)
        Mask over the parameter basis iota_nu(theta) for the whole QNN.
    """
    D_ins = d_loc_ins ** no_features
    D_pars = d_loc_pars ** no_params
    basis_inputs_mask = np.zeros(D_ins, dtype=np.int32)
    basis_params_mask = np.zeros(D_pars, dtype=np.int32)
    for j in range(no_qubits):
        basis_inputs_mask_j, basis_params_mask_j = inputs_params_basis_mask_vectors_qubits_qnn(j, BWLC_params, BWLC_inputs, 
                                                                                               no_params, d_loc_pars, 
                                                                                               no_features, d_loc_ins)
        basis_inputs_mask = np.logical_or(basis_inputs_mask, basis_inputs_mask_j)
        basis_params_mask = np.logical_or(basis_params_mask, basis_params_mask_j)
    return basis_inputs_mask, basis_params_mask
