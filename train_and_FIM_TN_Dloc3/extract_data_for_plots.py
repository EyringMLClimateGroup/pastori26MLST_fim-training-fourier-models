"""Aggregates the per-(regime, decay_exp/eps, model draw) pickled results written by
train_and_FIM_biased.py, train_and_FIM_UNbiased.py and train_and_FIM_SLIGHTbiased.py in
this same TN folder into three consolidated per-regime result dicts for the companion
plotting notebook (plot_training_and_FIM_withbias.ipynb). Does not train or compute
anything new: it globs/loads dict_results*.pkl (and, for the slightly-biased case,
model_biases*.pkl) files and, for each scanned decay_exp (and, for the slightly-biased
case, each eps in eps_vec), collects across model draws: delta_minMSE_mean/std/min/max,
MSE_full/cut statistics, normalized FIM spectra, norm_eff_dim_full/cut, and (slightly-biased
only) the model_bias_full/cut diagnostic distances.

- Identifies the right files to load using: no_of_features, max_frequency, no_params,
  dim_basis_single_param, bond_dim, cutoff, batch_size, learning_rate.
- Swept variables: decay_exp_vec (both loops) and eps_vec (slightly-biased loop only).
- Pickles the aggregated dicts as extracted_results_SLIGHTbiased*.pkl,
  extracted_results_FULLbiased*.pkl and extracted_results_UNbiased*.pkl.
"""

# Importing necessary packages
import sys
import os
import importlib
import re
import pickle

import numpy as np






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

import tensorized_model_constructor_functions
importlib.reload(tensorized_model_constructor_functions)

import tensor_network_functions_jax
importlib.reload(tensor_network_functions_jax)





### ---------------------------------------------------------------------------------------- ###
## ------------------------------------ Basic model specs ----------------------------------- ##
### ---------------------------------------------------------------------------------------- ###

# Folder in which to load data from
results_folder = '/work/bd1179/b309245/fourier_models_train_and_FIM/train_and_FIM_TN_Dloc3/'

### 'biased_data_gen': the data generating funct. is fully encompassed by both full and cutoff models
### 'UNbiased_data_gen': the data generating funct. is random, not encompassed by full and cutoff models
### 'SLIGHTbiased_data_gen': the data generating funct. is only approximately encompassed by full and cutoff models

name_data_gen_fullbiased = 'biased_data_gen'
name_data_gen_unbiased = 'UNbiased_data_gen'
name_data_gen = 'SLIGHTbiased_data_gen'

eps_vec = [0.0001, 0.0002, 0.0004, 0.0007, 0.001, 0.002, 0.004, 0.0053, 
           0.007, 0.0083, 0.01, 0.0125, 0.0165, 0.02, 0.025, 0.03, 
           0.035, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.2, 
           0.3, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0]
name_eps_vec = ['0p0001', '0p0002', '0p0004', '0p0007', '0p001', '0p002', '0p004', '0p0053', 
                '0p007', '0p0083', '0p01', '0p0125', '0p0165', '0p02', '0p025', '0p03', 
                '0p035', '0p04', '0p05', '0p06', '0p07', '0p08', '0p1', '0p2', 
                '0p3', '0p4', '0p6', '0p8', '1p0', '2p0', '3p0', '4p0']

# Learning rate
learning_rate = 0.02
learning_rate_name = '0p02'

# Batch size for training
batch_size = 5
batch_size_name = str(batch_size)

### Decay exponents for cutoff model
decay_exp_vec = [0.05, 0.07, 0.1, 0.2, 0.333, 0.5, 0.7, 1.0, 2.0, 3.33]
decay_exp_name_vec = ['0p05', '0p07', '0p1', '0p2', '0p333', '0p5', '0p7', '1p0', '2p0', '3p33']

### No. of features (dimension of input vectors)
no_of_features = 1
name_no_features = str(no_of_features)

#max_frequency = 9; no_params = 20; bond_dim = 50; cutoff = 2
max_frequency = 17; no_params = 32; bond_dim = 60; cutoff = 7

name_max_freq = str(max_frequency)
name_no_params = str(no_params)
name_bond_dim = str(bond_dim)
cutoff_name = str(cutoff)

### No. basis states per parameter
dim_basis_single_param = 3  ### (1, cos(th), sin(th))
name_dim_basis_param = str(dim_basis_single_param)





### ---------------------------------------------------------------------------------------- ###
## ---------------------------------------- Load data --------------------------------------- ##
### ---------------------------------------------------------------------------------------- ###

dict_slightbiased_all = dict()
for nep in range(len(eps_vec)):
    eps = eps_vec[nep]
    name_eps = name_eps_vec[nep]
    name_end = ('_BondDim' + name_bond_dim + '_Nfeatures' + name_no_features + '_MaxFreq' + name_max_freq + 
                '_Nparams' + name_no_params + '_DimBasisParam' + name_dim_basis_param + '_' + name_data_gen + 
                '_Eps' + name_eps + '_batch' + batch_size_name + '_lr' + learning_rate_name + 
                '_cutoff' + cutoff_name + '_decayexp')
    filename0 = 'dict_results' + name_end
    filename00 = 'model_biases' + name_end
    listallfiles = [f for f in os.listdir(results_folder) if (f.startswith(filename0))]
    dict_eps = dict()
    for filename in listallfiles:
        res = re.findall((filename0 + '(\S+)_modeldraw(\S+).pkl'), filename)
        decexp_name = res[0][0]
        nm_name = res[0][1]
        nm = int(nm_name)
        dict_d = dict()
        dict_d['delta_minMSE_mean'] = []
        dict_d['delta_minMSE_std'] = []
        dict_d['delta_minMSE_min'] = []
        dict_d['delta_minMSE_max'] = []
        dict_d['MSE_full_mean'] = []
        dict_d['MSE_full_std'] = []
        dict_d['MSE_full_min'] = []
        dict_d['MSE_full_max'] = []
        dict_d['MSE_cut_mean'] = []
        dict_d['MSE_cut_std'] = []
        dict_d['MSE_cut_min'] = []
        dict_d['MSE_cut_max'] = []
        dict_d['mean_normFIM_spectra_full'] = []
        dict_d['std_normFIM_spectra_full'] = []
        dict_d['min_normFIM_spectra_full'] = []
        dict_d['max_normFIM_spectra_full'] = []
        dict_d['norm_eff_dim_full'] = []
        dict_d['mean_normFIM_spectra_cut'] = []
        dict_d['std_normFIM_spectra_cut'] = []
        dict_d['min_normFIM_spectra_cut'] = []
        dict_d['max_normFIM_spectra_cut'] = []
        dict_d['norm_eff_dim_cut'] = []
        dict_d['model_bias_full'] = []
        dict_d['model_bias_cut'] = []
        dict_eps[decexp_name] = dict_d
    for filename in listallfiles:
        res = re.findall((filename0 + '(\S+)_modeldraw(\S+).pkl'), filename)
        decexp_name = res[0][0]
        nm_name = res[0][1]
        nm = int(nm_name)
        fn1 = filename
        path_file = os.path.join(results_folder, fn1)
        with open(path_file, 'rb') as f:
            dict_model = pickle.load(f)
        fn2 = filename00 + decexp_name + '_modeldraw' + nm_name + '.pkl'
        path_file = os.path.join(results_folder, fn2)
        with open(path_file, 'rb') as f:
            dict_biases = pickle.load(f)
        dict_d = dict_eps[decexp_name]
        dict_d['delta_minMSE_mean'].append(np.mean(dict_model['delta_minMSE_full_m_cut']))
        dict_d['delta_minMSE_std'].append(np.std(dict_model['delta_minMSE_full_m_cut']))
        dict_d['delta_minMSE_min'].append(np.min(dict_model['delta_minMSE_full_m_cut']))
        dict_d['delta_minMSE_max'].append(np.max(dict_model['delta_minMSE_full_m_cut']))
        dict_d['MSE_full_mean'].append(dict_model['MSE_full_mean'])
        dict_d['MSE_full_std'].append(dict_model['MSE_full_std'])
        dict_d['MSE_full_min'].append(dict_model['MSE_full_min'])
        dict_d['MSE_full_max'].append(dict_model['MSE_full_max'])
        dict_d['MSE_cut_mean'].append(dict_model['MSE_cut_mean'])
        dict_d['MSE_cut_std'].append(dict_model['MSE_cut_std'])
        dict_d['MSE_cut_min'].append(dict_model['MSE_cut_min'])
        dict_d['MSE_cut_max'].append(dict_model['MSE_cut_max'])
        dict_d['mean_normFIM_spectra_full'].append(dict_model['mean_normFIM_spectra_full'])
        dict_d['std_normFIM_spectra_full'].append(dict_model['std_normFIM_spectra_full'])
        dict_d['min_normFIM_spectra_full'].append(dict_model['min_normFIM_spectra_full'])
        dict_d['max_normFIM_spectra_full'].append(dict_model['max_normFIM_spectra_full'])
        dict_d['norm_eff_dim_full'].append(dict_model['norm_eff_dim_full'])
        dict_d['mean_normFIM_spectra_cut'].append(dict_model['mean_normFIM_spectra_cut'])
        dict_d['std_normFIM_spectra_cut'].append(dict_model['std_normFIM_spectra_cut'])
        dict_d['min_normFIM_spectra_cut'].append(dict_model['min_normFIM_spectra_cut'])
        dict_d['max_normFIM_spectra_cut'].append(dict_model['max_normFIM_spectra_cut'])
        dict_d['norm_eff_dim_cut'].append(dict_model['norm_eff_dim_cut'])
        dict_d['model_bias_full'].append(dict_biases['model_bias_full'])
        dict_d['model_bias_cut'].append(dict_biases['model_bias_cut'])
        dict_eps[decexp_name] = dict_d
    dict_slightbiased_all[eps] = dict_eps
    

name_end = ('_BondDim' + name_bond_dim + '_Nfeatures' + name_no_features + '_MaxFreq' + name_max_freq + 
            '_Nparams' + name_no_params + '_DimBasisParam' + name_dim_basis_param + '_' + name_data_gen_fullbiased + 
            '_batch' + batch_size_name + '_lr' + learning_rate_name + '_cutoff' + cutoff_name + 
            '_decayexp')
filename0 = 'dict_results' + name_end
listallfiles = [f for f in os.listdir(results_folder) if (f.startswith(filename0))]
dict_fullbiased_all = dict()
for filename in listallfiles:
    res = re.findall((filename0 + '(\S+)_modeldraw(\S+).pkl'), filename)
    decexp_name = res[0][0]
    nm_name = res[0][1]
    nm = int(nm_name)
    dict_d = dict()
    dict_d['delta_minMSE_mean'] = []
    dict_d['delta_minMSE_std'] = []
    dict_d['delta_minMSE_min'] = []
    dict_d['delta_minMSE_max'] = []
    dict_d['MSE_full_mean'] = []
    dict_d['MSE_full_std'] = []
    dict_d['MSE_full_min'] = []
    dict_d['MSE_full_max'] = []
    dict_d['MSE_cut_mean'] = []
    dict_d['MSE_cut_std'] = []
    dict_d['MSE_cut_min'] = []
    dict_d['MSE_cut_max'] = []
    dict_d['mean_normFIM_spectra_full'] = []
    dict_d['std_normFIM_spectra_full'] = []
    dict_d['min_normFIM_spectra_full'] = []
    dict_d['max_normFIM_spectra_full'] = []
    dict_d['norm_eff_dim_full'] = []
    dict_d['mean_normFIM_spectra_cut'] = []
    dict_d['std_normFIM_spectra_cut'] = []
    dict_d['min_normFIM_spectra_cut'] = []
    dict_d['max_normFIM_spectra_cut'] = []
    dict_d['norm_eff_dim_cut'] = []
    dict_fullbiased_all[decexp_name] = dict_d
for filename in listallfiles:
    res = re.findall((filename0 + '(\S+)_modeldraw(\S+).pkl'), filename)
    decexp_name = res[0][0]
    nm_name = res[0][1]
    nm = int(nm_name)
    path_file = os.path.join(results_folder, filename)
    with open(path_file, 'rb') as f:
        dict_model = pickle.load(f)
    dict_d = dict_fullbiased_all[decexp_name]
    dict_d['delta_minMSE_mean'].append(np.mean(dict_model['delta_minMSE_full_m_cut']))
    dict_d['delta_minMSE_std'].append(np.std(dict_model['delta_minMSE_full_m_cut']))
    dict_d['delta_minMSE_min'].append(np.min(dict_model['delta_minMSE_full_m_cut']))
    dict_d['delta_minMSE_max'].append(np.max(dict_model['delta_minMSE_full_m_cut']))
    dict_d['MSE_full_mean'].append(dict_model['MSE_full_mean'])
    dict_d['MSE_full_std'].append(dict_model['MSE_full_std'])
    dict_d['MSE_full_min'].append(dict_model['MSE_full_min'])
    dict_d['MSE_full_max'].append(dict_model['MSE_full_max'])
    dict_d['MSE_cut_mean'].append(dict_model['MSE_cut_mean'])
    dict_d['MSE_cut_std'].append(dict_model['MSE_cut_std'])
    dict_d['MSE_cut_min'].append(dict_model['MSE_cut_min'])
    dict_d['MSE_cut_max'].append(dict_model['MSE_cut_max'])
    dict_d['mean_normFIM_spectra_full'].append(dict_model['mean_normFIM_spectra_full'])
    dict_d['std_normFIM_spectra_full'].append(dict_model['std_normFIM_spectra_full'])
    dict_d['min_normFIM_spectra_full'].append(dict_model['min_normFIM_spectra_full'])
    dict_d['max_normFIM_spectra_full'].append(dict_model['max_normFIM_spectra_full'])
    dict_d['norm_eff_dim_full'].append(dict_model['norm_eff_dim_full'])
    dict_d['mean_normFIM_spectra_cut'].append(dict_model['mean_normFIM_spectra_cut'])
    dict_d['std_normFIM_spectra_cut'].append(dict_model['std_normFIM_spectra_cut'])
    dict_d['min_normFIM_spectra_cut'].append(dict_model['min_normFIM_spectra_cut'])
    dict_d['max_normFIM_spectra_cut'].append(dict_model['max_normFIM_spectra_cut'])
    dict_d['norm_eff_dim_cut'].append(dict_model['norm_eff_dim_cut'])
    dict_fullbiased_all[decexp_name] = dict_d


name_end = ('_BondDim' + name_bond_dim + '_Nfeatures' + name_no_features + '_MaxFreq' + name_max_freq + 
            '_Nparams' + name_no_params + '_DimBasisParam' + name_dim_basis_param + '_' + name_data_gen_unbiased + 
            '_batch' + batch_size_name + '_lr' + learning_rate_name + '_cutoff' + cutoff_name + 
            '_decayexp')
filename0 = 'dict_results' + name_end
listallfiles = [f for f in os.listdir(results_folder) if (f.startswith(filename0))]
dict_unbiased_all = dict()
for filename in listallfiles:
    res = re.findall((filename0 + '(\S+)_modeldraw(\S+).pkl'), filename)
    decexp_name = res[0][0]
    nm_name = res[0][1]
    nm = int(nm_name)
    dict_d = dict()
    dict_d['delta_minMSE_mean'] = []
    dict_d['delta_minMSE_std'] = []
    dict_d['delta_minMSE_min'] = []
    dict_d['delta_minMSE_max'] = []
    dict_d['MSE_full_mean'] = []
    dict_d['MSE_full_std'] = []
    dict_d['MSE_full_min'] = []
    dict_d['MSE_full_max'] = []
    dict_d['MSE_cut_mean'] = []
    dict_d['MSE_cut_std'] = []
    dict_d['MSE_cut_min'] = []
    dict_d['MSE_cut_max'] = []
    dict_d['mean_normFIM_spectra_full'] = []
    dict_d['std_normFIM_spectra_full'] = []
    dict_d['min_normFIM_spectra_full'] = []
    dict_d['max_normFIM_spectra_full'] = []
    dict_d['norm_eff_dim_full'] = []
    dict_d['mean_normFIM_spectra_cut'] = []
    dict_d['std_normFIM_spectra_cut'] = []
    dict_d['min_normFIM_spectra_cut'] = []
    dict_d['max_normFIM_spectra_cut'] = []
    dict_d['norm_eff_dim_cut'] = []
    dict_unbiased_all[decexp_name] = dict_d
for filename in listallfiles:
    res = re.findall((filename0 + '(\S+)_modeldraw(\S+).pkl'), filename)
    decexp_name = res[0][0]
    nm_name = res[0][1]
    nm = int(nm_name)
    path_file = os.path.join(results_folder, filename)
    with open(path_file, 'rb') as f:
        dict_model = pickle.load(f)
    dict_d = dict_unbiased_all[decexp_name]
    dict_d['delta_minMSE_mean'].append(np.mean(dict_model['delta_minMSE_full_m_cut']))
    dict_d['delta_minMSE_std'].append(np.std(dict_model['delta_minMSE_full_m_cut']))
    dict_d['delta_minMSE_min'].append(np.min(dict_model['delta_minMSE_full_m_cut']))
    dict_d['delta_minMSE_max'].append(np.max(dict_model['delta_minMSE_full_m_cut']))
    dict_d['MSE_full_mean'].append(dict_model['MSE_full_mean'])
    dict_d['MSE_full_std'].append(dict_model['MSE_full_std'])
    dict_d['MSE_full_min'].append(dict_model['MSE_full_min'])
    dict_d['MSE_full_max'].append(dict_model['MSE_full_max'])
    dict_d['MSE_cut_mean'].append(dict_model['MSE_cut_mean'])
    dict_d['MSE_cut_std'].append(dict_model['MSE_cut_std'])
    dict_d['MSE_cut_min'].append(dict_model['MSE_cut_min'])
    dict_d['MSE_cut_max'].append(dict_model['MSE_cut_max'])
    dict_d['mean_normFIM_spectra_full'].append(dict_model['mean_normFIM_spectra_full'])
    dict_d['std_normFIM_spectra_full'].append(dict_model['std_normFIM_spectra_full'])
    dict_d['min_normFIM_spectra_full'].append(dict_model['min_normFIM_spectra_full'])
    dict_d['max_normFIM_spectra_full'].append(dict_model['max_normFIM_spectra_full'])
    dict_d['norm_eff_dim_full'].append(dict_model['norm_eff_dim_full'])
    dict_d['mean_normFIM_spectra_cut'].append(dict_model['mean_normFIM_spectra_cut'])
    dict_d['std_normFIM_spectra_cut'].append(dict_model['std_normFIM_spectra_cut'])
    dict_d['min_normFIM_spectra_cut'].append(dict_model['min_normFIM_spectra_cut'])
    dict_d['max_normFIM_spectra_cut'].append(dict_model['max_normFIM_spectra_cut'])
    dict_d['norm_eff_dim_cut'].append(dict_model['norm_eff_dim_cut'])
    dict_unbiased_all[decexp_name] = dict_d


name_end_extr = ('_BondDim' + name_bond_dim + '_Nfeatures' + name_no_features + '_MaxFreq' + name_max_freq + 
                 '_Nparams' + name_no_params + '_DimBasisParam' + name_dim_basis_param + '_cutoff' + cutoff_name)

filename = 'extracted_results_SLIGHTbiased' + name_end_extr + '.pkl'
path_file = os.path.join(results_folder, filename)
with open(path_file, 'wb') as f:
    pickle.dump(dict_slightbiased_all, f)
    
filename = 'extracted_results_FULLbiased' + name_end_extr + '.pkl'
path_file = os.path.join(results_folder, filename)
with open(path_file, 'wb') as f:
    pickle.dump(dict_fullbiased_all, f)

filename = 'extracted_results_UNbiased' + name_end_extr + '.pkl'
path_file = os.path.join(results_folder, filename)
with open(path_file, 'wb') as f:
    pickle.dump(dict_unbiased_all, f)