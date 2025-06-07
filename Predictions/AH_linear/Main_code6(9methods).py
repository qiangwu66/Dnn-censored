# python 3.12.8
#%% ---------------------
import numpy as np
import torch
import matplotlib.pyplot as plt
from data_generator import generate_case_6

from B_spline import *
import pandas as pd
import os
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d

import torchtuples as tt
from pycox.models import CoxCC

from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime

from pycox.models import CoxPH

from pycox.models import DeepHitSingle


from Survival_methods.AH_iteration import Surv_AH
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv 

from lifelines import LogNormalAFTFitter

from Survival_methods.SAFT import SAFT_C_est

import time

from joblib import Parallel, delayed 

results_folder = "results_folder"
os.makedirs(results_folder, exist_ok=True)

#%% ---------------------
def set_seed(seed):
    np.random.seed(seed) 
    torch.manual_seed(seed)

set_seed(1)
#%% -----------------------
tau = 2 
p = 3  
Set_n = np.array([400, 800])  
corr = 0.5  

Set_Dnn_layer = [2, 2] 
Set_Dnn_node = [30, 30] 
Set_Dnn_epoch = [1000, 1000] 
Dnn_Set_lr = np.array([3e-4, 3e-4])
patiences = 10


def Indicator_matrix(a, b):
    a = np.array(a)
    b = np.array(b)
    I_M = (a[:, np.newaxis] >= b).astype(int)
    return I_M

def check_matrix_or_vector(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 2 or (value.ndim == 1 and value.size > 1):
            return True
    return False

s = 500
Data_test = generate_case_6(s, corr, tau)
X_test = Data_test['X'] 
f_X_test = Data_test['f_X']
T_O_test = Data_test['T_O']
T_O_test = np.array([float(f"{x:.4g}") for x in T_O_test])
T_O_test = np.maximum(T_O_test, 1e-4)
De_test = Data_test['De']
durations_test, events_test = Data_test['T_O'], Data_test['De'] 

St_true_X_DMS = np.exp(-(T_O_test + 0.001) ** 2 / 20 - T_O_test * f_X_test) # s


n1 = 101
t_nodes = np.array(np.linspace(0, tau, n1), dtype="float32")
t_nodes = np.maximum(t_nodes, 1e-4)


kmf = KaplanMeierFitter()
kmf.fit(T_O_test, event_observed = 1 - De_test)
G_T_i = np.maximum(kmf.predict(T_O_test).values, np.min(kmf.predict(T_O_test).values[kmf.predict(T_O_test).values > 0])) 


original_times = kmf.survival_function_.index.values[1:] 
original_survival_probs = kmf.survival_function_["KM_estimate"].values[1:]  

s_k = np.linspace(T_O_test.min(), T_O_test.max(), 100, endpoint=False) 
s_k = np.array([float(f"{x:.4g}") for x in s_k])
s_k = np.maximum(s_k, 1e-4)

interpolator = interp1d(original_times, original_survival_probs, kind="previous", fill_value="extrapolate")
G_s_k = np.maximum(interpolator(s_k), np.min(interpolator(s_k)[interpolator(s_k) > 0])) 


I_T_i_s_k = Indicator_matrix(s_k, T_O_test)
I_T_i_s_k_D_1 = Indicator_matrix(s_k, T_O_test) * np.tile(De_test, (len(s_k), 1)) 



t_fig = np.array(np.linspace(0, tau, 20), dtype="float32")
t_fig = np.maximum(t_fig, 1e-4)
Lambda_t_true = (t_fig + 0.001) ** 2 / 20
St_true_X_fig = np.exp(-np.repeat(Lambda_t_true[:,np.newaxis],X_test.shape[0],axis=1) - np.repeat(t_fig[:,np.newaxis],X_test.shape[0],axis=1) * np.repeat(f_X_test[np.newaxis,:],len(t_fig),axis=0)) 


np.savetxt(os.path.join(results_folder, 'X_test.csv'), X_test, delimiter=',')
np.savetxt(os.path.join(results_folder, 'De_test.csv'), De_test, delimiter=',')
np.savetxt(os.path.join(results_folder, 'T_O_test.csv'), T_O_test, delimiter=',')
np.savetxt(os.path.join(results_folder, 't_fig.csv'), t_fig, delimiter=',')
np.savetxt(os.path.join(results_folder, 't_nodes.csv'), t_nodes, delimiter=',')
np.savetxt(os.path.join(results_folder, 's_k.csv'), s_k, delimiter=',')
np.savetxt(os.path.join(results_folder, 'G_s_k.csv'), G_s_k, delimiter=',')
np.savetxt(os.path.join(results_folder, 'G_T_i.csv'), G_T_i, delimiter=',')
np.savetxt(os.path.join(results_folder, 'I_T_i_s_k.csv'), I_T_i_s_k, delimiter=',')
np.savetxt(os.path.join(results_folder, 'I_T_i_s_k_D_1.csv'), I_T_i_s_k_D_1, delimiter=',')
np.savetxt(os.path.join(results_folder, 'St_true_X_DMS.csv'), St_true_X_DMS, delimiter=',')

np.save(os.path.join(results_folder, 't_fig.npy'), t_fig)
np.save(os.path.join(results_folder, 'St_true_X_fig.npy'), St_true_X_fig)


#%% ========================
def single_simulation(b, n, t_nodes, t_fig, tau, De_test, St_true_X_DMS, X_test, T_O_test, s_k, m_AH, nodevec_AH):
    print(f'-------------n={n}, b={b}--------------')
    set_seed(20 + b)
    # ---------------------------
    Data_all = generate_case_6(n, corr, tau) 
    
    train_data = {
        key: (value[:int(0.8 * n)] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data_all.items()
        }
    val_data = {
        key: (value[-int(0.2 * n):] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data_all.items()
        }

    # ----------------Cox Linear-------------------------

    start_time_CoxLinear = time.time()


    columns = [f'feature_{i+1}' for i in range(train_data['X'].shape[1])]
    data = pd.DataFrame(train_data['X'], columns=columns)
    data['duration'] = np.maximum(train_data['T_O'], 1e-4)
    data['event'] = train_data['De']

    cph = CoxPHFitter()
    cph.fit(data, duration_col='duration', event_col='event')

    new_X = X_test[:10, :]
    new_data = pd.DataFrame(new_X, columns=[f'feature_{i+1}' for i in range(X_test.shape[1])])

    S_t_X_CoxPH = cph.predict_survival_function(new_data, times=t_fig)

    CoxPH_IBS = cph.predict_survival_function(X_test, times=s_k) 
    IBS_CoxLinear = np.nanmean(CoxPH_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - CoxPH_IBS) ** 2 * (1 - I_T_i_s_k) / np.tile(G_s_k, (len(T_O_test), 1)).T)

    new_data1 = pd.DataFrame(X_test, columns=[f'feature_{i+1}' for i in range(X_test.shape[1])])
    S_t_X_CoxPH_DMS = cph.predict_survival_function(new_data1, times=T_O_test) 
    DMS_CoxLinear = np.sum(De_test * np.abs(St_true_X_DMS - np.diagonal(S_t_X_CoxPH_DMS)) / np.sum(De_test))

    end_time_CoxLinear = time.time()
    run_time_CoxLinear = end_time_CoxLinear - start_time_CoxLinear

    # ---------------Cox-CC---------------------------
    start_time_CoxCC = time.time()

    y_train = (train_data['T_O'], train_data['De'])
    y_val = (val_data['T_O'], val_data['De'])
    val = tt.tuplefy(val_data['X'], y_val)
    in_features = train_data['X'].shape[1]
    num_nodes = [32, 32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)
    model = CoxCC(net, tt.optim.Adam)
    batch_size = 256
    model.optimizer.set_lr(0.01)
    epochs = 512
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = True
    log = model.fit(train_data['X'], y_train, batch_size, epochs, callbacks, verbose,
                    val_data=val.repeat(10).cat())
    
    model.partial_log_likelihood(*val).mean()
    _ = model.compute_baseline_hazards()
    surv = model.predict_surv_df(X_test) 

    S_t_X_CoxCC = surv.apply(lambda col: np.interp(t_fig, surv.index, col), axis=0).to_numpy() 

    Survial_CoxCC_IBS = surv.apply(lambda col: np.interp(s_k, surv.index, col), axis=0).to_numpy() 
    IBS_CoxCC = np.nanmean(Survial_CoxCC_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - Survial_CoxCC_IBS) ** 2 * (1 - I_T_i_s_k) / np.tile(G_s_k, (len(T_O_test), 1)).T)

    S_t_X_CoxCC_DMS = surv.apply(lambda col: np.interp(T_O_test, surv.index, col), axis=0).to_numpy() 
    DMS_CoxCC = np.sum(De_test * np.abs(St_true_X_DMS - np.diagonal(S_t_X_CoxCC_DMS)) / np.sum(De_test))

    end_time_CoxCC = time.time()
    run_time_CoxCC = end_time_CoxCC - start_time_CoxCC

    # ---------------Cox-Time---------------------------
    start_time_CoxTime = time.time()

    labtrans = CoxTime.label_transform()
    y_train_CoxTime = labtrans.fit_transform(*y_train)
    y_val_CoxTime = labtrans.transform(*y_val)
    val_CoxTime = tt.tuplefy(val_data['X'], y_val_CoxTime)
    in_features = train_data['X'].shape[1]
    num_nodes = [32, 32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
    model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)
    batch_size = 256
    model.optimizer.set_lr(0.01)
    epochs = 512
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = True
    log = model.fit(train_data['X'], y_train_CoxTime, batch_size, epochs, callbacks, verbose,
                    val_data=val_CoxTime.repeat(10).cat())


    model.partial_log_likelihood(*val_CoxTime).mean()
    _ = model.compute_baseline_hazards()

    surv_CoxTime = model.predict_surv_df(X_test) 
    S_t_X_CoxTime = surv_CoxTime.apply(lambda col: np.interp(t_fig, surv_CoxTime.index, col), axis=0).to_numpy() 
    Survial_CoxTime_IBS = surv_CoxTime.apply(lambda col: np.interp(s_k, surv_CoxTime.index, col), axis=0).to_numpy() 
    IBS_CoxTime = np.nanmean(Survial_CoxTime_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - Survial_CoxTime_IBS) ** 2 * (1 - I_T_i_s_k) / np.tile(G_s_k, (len(T_O_test), 1)).T)
    S_t_X_CoxTime_DMS = surv_CoxTime.apply(lambda col: np.interp(T_O_test, surv_CoxTime.index, col), axis=0).to_numpy() 
    DMS_CoxTime = np.sum(De_test * np.abs(St_true_X_DMS - np.diagonal(S_t_X_CoxTime_DMS)) / np.sum(De_test))  
    end_time_CoxTime = time.time()
    run_time_CoxTime = end_time_CoxTime - start_time_CoxTime

    # ---------------DeepSurv---------------------------
    start_time_DeepSurv = time.time()

    in_features = train_data['X'].shape[1]
    num_nodes = [32, 32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features,
                                  batch_norm,dropout, output_bias=output_bias)

    model = CoxPH(net, tt.optim.Adam)
    batch_size = 256
    model.optimizer.set_lr(0.01)
    epochs = 512
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = True
    log = model.fit(train_data['X'], y_train, batch_size, epochs, callbacks, verbose, 
                    val_data=val, val_batch_size=batch_size)


    model.partial_log_likelihood(*val).mean()
    _ = model.compute_baseline_hazards()
    surv_DeepSurv = model.predict_surv_df(X_test) 
    S_t_X_DeepSurv = surv_DeepSurv.apply(lambda col: np.interp(t_fig, surv_DeepSurv.index, col), axis=0).to_numpy()
    Survial_DeepSurv_IBS = surv_DeepSurv.apply(lambda col: np.interp(s_k, surv_DeepSurv.index, col), axis=0).to_numpy() 
    IBS_DeepSurv = np.nanmean(Survial_DeepSurv_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - Survial_DeepSurv_IBS) ** 2 * (1 - I_T_i_s_k) / np.tile(G_s_k, (len(T_O_test), 1)).T)
    S_t_X_DeepSurv_DMS = surv_DeepSurv.apply(lambda col: np.interp(T_O_test, surv_DeepSurv.index, col), axis=0).to_numpy() 
    DMS_DeepSurv = np.sum(De_test * np.abs(St_true_X_DMS - np.diagonal(S_t_X_DeepSurv_DMS)) / np.sum(De_test))
    end_time_DeepSurv = time.time()
    run_time_DeepSurv = end_time_DeepSurv - start_time_DeepSurv

    # ---------------DeepHit---------------------------
    start_time_DeepHit = time.time()

    num_durations = 10
    labtrans = DeepHitSingle.label_transform(num_durations)
    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_train_DeepHit = labtrans.fit_transform(*y_train)
    y_val_DeepHit = labtrans.transform(*y_val)
    val_DeepHit = tt.tuplefy(val_data['X'], y_val_DeepHit)

    in_features = train_data['X'].shape[1]
    num_nodes = [32, 32]
    out_features = labtrans.out_features
    batch_norm = True
    dropout = 0.1

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
    model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
    batch_size = 256
    model.optimizer.set_lr(0.01)
    epochs = 100
    callbacks = [tt.callbacks.EarlyStopping()]
    log = model.fit(train_data['X'],  y_train_DeepHit, batch_size, epochs, callbacks, val_data=val_DeepHit)
    surv_DeepHit = model.predict_surv_df(X_test)
    S_t_X_DeepHit = surv_DeepHit.apply(lambda col: np.interp(t_fig, surv_DeepHit.index, col), axis=0).to_numpy()
    Survial_DeepHit_IBS = surv_DeepHit.apply(lambda col: np.interp(s_k, surv_DeepHit.index, col), axis=0).to_numpy() 
    IBS_DeepHit = np.nanmean(Survial_DeepHit_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - Survial_DeepHit_IBS) ** 2 * (1 - I_T_i_s_k) / np.tile(G_s_k, (len(T_O_test), 1)).T)
    S_t_X_DeepHit_DMS = surv_DeepHit.apply(lambda col: np.interp(T_O_test, surv_DeepHit.index, col), axis=0).to_numpy() 
    DMS_DeepHit = np.sum(De_test * np.abs(St_true_X_DMS - np.diagonal(S_t_X_DeepHit_DMS)) / np.sum(De_test))
    end_time_DeepHit = time.time()
    run_time_DeepHit = end_time_DeepHit - start_time_DeepHit

    # ---------------AH---------------------------
    start_time_AH = time.time()

    S_AH = Surv_AH(train_data, Data_test, t_nodes, m_AH, nodevec_AH, tau, s_k, t_fig)
    S_t_X_AH = S_AH["S_t_X_AH_fig"] 
    S_t_X_AH_IBS = S_AH["S_t_X_AH_IBS"] 
    IBS_AH = np.nanmean(S_t_X_AH_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - S_t_X_AH_IBS) ** 2 * (1 - I_T_i_s_k) / np.tile(G_s_k, (len(T_O_test), 1)).T)
    S_t_X_AH_DMS = S_AH["S_t_X_AH_DMS"] 
    DMS_AH = np.sum(De_test * np.abs(St_true_X_DMS - np.diagonal(S_t_X_AH_DMS)) / np.sum(De_test))
    end_time_AH = time.time()
    run_time_AH = end_time_AH - start_time_AH

    # ---------------RSF---------------------------
    start_time_RSF = time.time()

    columns = [f'feature_{i+1}' for i in range(train_data['X'].shape[1])] 
    X_train_DF = pd.DataFrame(train_data['X'], columns=columns)
    X_test_DF = pd.DataFrame(X_test, columns=columns)
    y_train_DF = pd.DataFrame({
    "event": train_data['De'].astype(bool),
    "time": train_data['T_O']
    })
    y_test_DF = pd.DataFrame({
    "event": De_test.astype(bool),
    "time": T_O_test
    })
    y_train_DF = Surv.from_dataframe("event", "time", y_train_DF)
    y_test_DF = Surv.from_dataframe("event", "time", y_test_DF)
    rsf = RandomSurvivalForest(
        n_estimators=1000, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=20)
    rsf.fit(X_train_DF, y_train_DF)
    surv_original = rsf.predict_survival_function(X_test_DF, return_array=True) 
    S_t_X_RSF = np.zeros((len(t_fig), len(X_test))) 
    for i, surv_func in enumerate(surv_original):
        interp_func = interp1d(rsf.unique_times_, surv_func, kind="previous", bounds_error=False, fill_value=(1.0, 0.0))
        S_t_X_RSF[:, i] = interp_func(t_fig)

    Survial_RSF_IBS = np.zeros((len(s_k), len(X_test)))  
    for i, surv_func in enumerate(surv_original):
        interp_func = interp1d(rsf.unique_times_, surv_func, kind="previous", bounds_error=False, fill_value=(1.0, 0.0))
        Survial_RSF_IBS[:, i] = interp_func(s_k)
    IBS_RSF = np.nanmean(Survial_RSF_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - Survial_RSF_IBS) ** 2 * (1 - I_T_i_s_k) / np.tile(G_s_k, (len(T_O_test), 1)).T)

    Survial_RSF_DMS = np.zeros((len(T_O_test), len(X_test)))  
    for i, surv_func in enumerate(surv_original):
        interp_func = interp1d(rsf.unique_times_, surv_func, kind="previous", bounds_error=False, fill_value=(1.0, 0.0))
        Survial_RSF_DMS[:, i] = interp_func(T_O_test)
    DMS_RSF = np.sum(De_test * np.abs(St_true_X_DMS - np.diagonal(Survial_RSF_DMS)) / np.sum(De_test))
    end_time_RSF = time.time()
    run_time_RSF = end_time_RSF - start_time_RSF

    # ----------------PAFT-------------------------
    start_time_PAFT = time.time()

    columns = [f'feature_{i+1}' for i in range(train_data['X'].shape[1])] 
    data = pd.DataFrame(train_data['X'], columns=columns)
    data['duration'] = np.maximum(train_data['T_O'], 1e-4)
    data['event'] = train_data['De']
    new_data = pd.DataFrame(X_test, columns=[f'feature_{i+1}' for i in range(X_test.shape[1])])
    aft_model = LogNormalAFTFitter()
    aft_model.fit(data, duration_col='duration', event_col='event')
 
    S_t_X_PAFT = aft_model.predict_survival_function(new_data, times=t_fig) 
    S_t_X_PAFT_IBS = aft_model.predict_survival_function(new_data, times=s_k) 
    IBS_PAFT = np.nanmean(S_t_X_PAFT_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - S_t_X_PAFT_IBS) ** 2 * (1 - I_T_i_s_k) / np.tile(G_s_k, (len(T_O_test), 1)).T)

    S_t_X_PAFT_DMS = aft_model.predict_survival_function(new_data, times=T_O_test) 
    DMS_PAFT = np.sum(De_test * np.abs(St_true_X_DMS - np.diagonal(S_t_X_PAFT_DMS)) / np.sum(De_test))

    end_time_PAFT = time.time()
    run_time_PAFT = end_time_PAFT - start_time_PAFT

    # ----------------SAFT-------------------------
    start_time_SAFT = time.time()

    beta_coefs = cph.summary[['coef']].values[:,0]
    Omega_b = np.max(np.concatenate([train_data['T_O'], T_O_test])) * np.exp(np.max(np.concatenate([train_data['X'] @ beta_coefs, X_test @ beta_coefs])))
    nodes_num = int((0.8 * n) ** (1 / 4)) 
    node_vec = np.array(np.linspace(0, Omega_b, nodes_num + 2), dtype="float32")      
    t_x_nodes = np.array(np.linspace(0, Omega_b, 501), dtype="float32")
    beta_X_train = np.array(train_data['X'] @ beta_coefs, dtype='float32') 
    Y_beta_X_train = np.array(train_data['T_O'] * np.exp(beta_X_train), dtype="float32")
    I_t_x_nodes_Y_X_train = Indicator_matrix(Y_beta_X_train, t_x_nodes) 
    c_saft = SAFT_C_est(train_data['De'], t_x_nodes, beta_X_train, Y_beta_X_train, I_t_x_nodes_Y_X_train, nodes_num, node_vec, Omega_b)

    beta_X_test = np.array(X_test @ beta_coefs, dtype='float32') 

    S_t_X_SAFT = []
    for t in t_fig:
        t_beta_X_test = np.array(t * np.exp(beta_X_test), dtype="float32") 
        S_t_X_SAFT.append(np.exp(- (Omega_b/len(t_x_nodes)) * Indicator_matrix(t_beta_X_test, t_x_nodes) @ np.exp(B_S(nodes_num, t_x_nodes, node_vec) @ c_saft))) 
    S_t_X_SAFT = np.array(S_t_X_SAFT)

    S_t_X_SAFT_IBS = []
    for t in s_k:
        t_beta_X_test = np.array(t * np.exp(beta_X_test), dtype="float32") 
        S_t_X_SAFT_IBS.append(np.exp(- (Omega_b/len(t_x_nodes)) * Indicator_matrix(t_beta_X_test, t_x_nodes) @ np.exp(B_S(nodes_num, t_x_nodes, node_vec) @ c_saft)))
    S_t_X_SAFT_IBS = np.array(S_t_X_SAFT_IBS)
    IBS_SAFT = np.nanmean(S_t_X_SAFT_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - S_t_X_SAFT_IBS) ** 2 * (1 - I_T_i_s_k) / np.tile(G_s_k, (len(T_O_test), 1)).T)

    S_t_X_SAFT_DMS = []
    for t in T_O_test:
        t_beta_X_test = np.array(t * np.exp(beta_X_test), dtype="float32")
        S_t_X_SAFT_DMS.append(np.exp(-(Omega_b/len(t_x_nodes)) * Indicator_matrix(t_beta_X_test, t_x_nodes) @ np.exp(B_S(nodes_num, t_x_nodes, node_vec) @ c_saft))) 
    S_t_X_SAFT_DMS = np.array(S_t_X_SAFT_DMS) 
    DMS_SAFT = np.sum(De_test * np.abs(St_true_X_DMS - np.diagonal(S_t_X_SAFT_DMS)) / np.sum(De_test))
    
    end_time_SAFT = time.time()
    run_time_SAFT = end_time_SAFT - start_time_SAFT


    return {
        "S_t_X_Cox_Linear": S_t_X_CoxPH,
        "DMS_Cox_Linear": DMS_CoxLinear,
        "IBS_Cox_Linear": IBS_CoxLinear,
        "run_time_Cox_Linear": run_time_CoxLinear,
        "S_t_X_Cox_CC": S_t_X_CoxCC,
        "DMS_Cox_CC": DMS_CoxCC,
        "IBS_Cox_CC": IBS_CoxCC,
        "run_time_Cox_CC": run_time_CoxCC,
        "S_t_X_Cox_Time": S_t_X_CoxTime,
        "DMS_Cox_Time": DMS_CoxTime,
        "IBS_Cox_Time": IBS_CoxTime,
        "run_time_Cox_Time": run_time_CoxTime,
        "S_t_X_DeepSurv": S_t_X_DeepSurv,      
        "DMS_DeepSurv": DMS_DeepSurv,
        "IBS_DeepSurv": IBS_DeepSurv,
        "run_time_DeepSurv": run_time_DeepSurv,
        "S_t_X_DeepHit": S_t_X_DeepHit,      
        "DMS_DeepHit": DMS_DeepHit,
        "IBS_DeepHit": IBS_DeepHit,
        "run_time_DeepHit": run_time_DeepHit,
        "S_t_X_AH": S_t_X_AH,      
        "DMS_AH": DMS_AH,
        "IBS_AH": IBS_AH,
        "run_time_AH": run_time_AH,
        "S_t_X_RSF": S_t_X_RSF,      
        "DMS_RSF": DMS_RSF,
        "IBS_RSF": IBS_RSF,
        "run_time_RSF": run_time_RSF,
        "S_t_X_PAFT": S_t_X_PAFT,      
        "DMS_PAFT": DMS_PAFT,
        "IBS_PAFT": IBS_PAFT,
        "run_time_PAFT": run_time_PAFT,
        "S_t_X_SAFT": S_t_X_SAFT,      
        "DMS_SAFT": DMS_SAFT,
        "IBS_SAFT": IBS_SAFT,
        "run_time_SAFT": run_time_SAFT,
    }

#%% ======================
results_400 = []  
results_800 = []  
n_jobs = 10
B = 200

for ii in range(len(Set_n)):
    n = Set_n[ii]
    Dnn_layer = Set_Dnn_layer[ii]
    Dnn_node = Set_Dnn_node[ii]
    Dnn_epoch = Set_Dnn_epoch[ii]
    Dnn_lr = Dnn_Set_lr[ii]

    m = 3 
    nodevec = np.array(np.linspace(0, tau, m + 2), dtype="float32") 
    m_AH = int(int(0.8 * n) ** (1 / 4)) 
    nodevec_AH = np.array(np.linspace(0, tau, m_AH + 2), dtype="float32") 


    if ii == 0:  # n=400
        results_400 = Parallel(n_jobs=n_jobs)(
            delayed(single_simulation)(b, n, t_nodes, t_fig, tau, De_test, St_true_X_DMS, X_test, T_O_test, s_k, m_AH, nodevec_AH) 
            for b in range(B)
        )
    else:  # n=800
        results_800 = Parallel(n_jobs=n_jobs)(
            delayed(single_simulation)(b, n, t_nodes, t_fig, tau, De_test, St_true_X_DMS, X_test, T_O_test, s_k, m_AH, nodevec_AH) 
            for b in range(B)
        )

#%% ======================
def process_results(results, file_prefix):
    S_t_X_Cox_Linear, S_t_X_Cox_CC, S_t_X_Cox_Time, S_t_X_DeepSurv, S_t_X_DeepHit, S_t_X_AH, S_t_X_RSF, S_t_X_PAFT, S_t_X_SAFT = [], [], [], [], [], [], [], [], []
    DMS_Cox_Linear, DMS_Cox_CC, DMS_Cox_Time, DMS_DeepSurv, DMS_DeepHit, DMS_AH, DMS_RSF, DMS_PAFT, DMS_SAFT = [], [], [], [], [], [], [], [], []
    IBS_Cox_Linear, IBS_Cox_CC, IBS_Cox_Time, IBS_DeepSurv, IBS_DeepHit, IBS_AH, IBS_RSF, IBS_PAFT, IBS_SAFT = [], [], [], [], [], [], [], [], []
    run_time_Cox_Linear, run_time_Cox_CC, run_time_Cox_Time, run_time_DeepSurv, run_time_DeepHit, run_time_AH, run_time_RSF, run_time_PAFT, run_time_SAFT = [], [], [], [], [], [], [], [], []

    for res in results:
        S_t_X_Cox_Linear.append(res["S_t_X_Cox_Linear"])
        DMS_Cox_Linear.append(res["DMS_Cox_Linear"])
        IBS_Cox_Linear.append(res["IBS_Cox_Linear"])
        run_time_Cox_Linear.append(res["run_time_Cox_Linear"])
        S_t_X_Cox_CC.append(res["S_t_X_Cox_CC"])
        DMS_Cox_CC.append(res["DMS_Cox_CC"])
        IBS_Cox_CC.append(res["IBS_Cox_CC"])
        run_time_Cox_CC.append(res["run_time_Cox_CC"])
        S_t_X_Cox_Time.append(res["S_t_X_Cox_Time"])
        DMS_Cox_Time.append(res["DMS_Cox_Time"])
        IBS_Cox_Time.append(res["IBS_Cox_Time"])
        run_time_Cox_Time.append(res["run_time_Cox_Time"])
        S_t_X_DeepSurv.append(res["S_t_X_DeepSurv"])
        DMS_DeepSurv.append(res["DMS_DeepSurv"])
        IBS_DeepSurv.append(res["IBS_DeepSurv"])
        run_time_DeepSurv.append(res["run_time_DeepSurv"])
        S_t_X_DeepHit.append(res["S_t_X_DeepHit"])
        DMS_DeepHit.append(res["DMS_DeepHit"])
        IBS_DeepHit.append(res["IBS_DeepHit"])
        run_time_DeepHit.append(res["run_time_DeepHit"])
        S_t_X_AH.append(res["S_t_X_AH"])
        DMS_AH.append(res["DMS_AH"])
        IBS_AH.append(res["IBS_AH"])
        run_time_AH.append(res["run_time_AH"])
        S_t_X_RSF.append(res["S_t_X_RSF"])
        DMS_RSF.append(res["DMS_RSF"])
        IBS_RSF.append(res["IBS_RSF"])
        run_time_RSF.append(res["run_time_RSF"])
        S_t_X_PAFT.append(res["S_t_X_PAFT"])
        DMS_PAFT.append(res["DMS_PAFT"])
        IBS_PAFT.append(res["IBS_PAFT"])
        run_time_PAFT.append(res["run_time_PAFT"])
        S_t_X_SAFT.append(res["S_t_X_SAFT"])
        DMS_SAFT.append(res["DMS_SAFT"])
        IBS_SAFT.append(res["IBS_SAFT"])
        run_time_SAFT.append(res["run_time_SAFT"])


    S_t_X_Cox_Linear = np.array(S_t_X_Cox_Linear)
    DMS_Cox_Linear = np.array(DMS_Cox_Linear)
    IBS_Cox_Linear = np.array(IBS_Cox_Linear)
    run_time_Cox_Linear = np.array(run_time_Cox_Linear)
    S_t_X_Cox_CC = np.array(S_t_X_Cox_CC)
    DMS_Cox_CC = np.array(DMS_Cox_CC)
    IBS_Cox_CC = np.array(IBS_Cox_CC)
    run_time_Cox_CC = np.array(run_time_Cox_CC)
    S_t_X_Cox_Time = np.array(S_t_X_Cox_Time)
    DMS_Cox_Time = np.array(DMS_Cox_Time)
    IBS_Cox_Time = np.array(IBS_Cox_Time)
    run_time_Cox_Time = np.array(run_time_Cox_Time)
    S_t_X_DeepSurv = np.array(S_t_X_DeepSurv)
    DMS_DeepSurv = np.array(DMS_DeepSurv)
    IBS_DeepSurv = np.array(IBS_DeepSurv)
    run_time_DeepSurv = np.array(run_time_DeepSurv)
    S_t_X_DeepHit = np.array(S_t_X_DeepHit)
    DMS_DeepHit = np.array(DMS_DeepHit)
    IBS_DeepHit = np.array(IBS_DeepHit)
    run_time_DeepHit = np.array(run_time_DeepHit)
    S_t_X_AH = np.array(S_t_X_AH)
    DMS_AH = np.array(DMS_AH)
    IBS_AH = np.array(IBS_AH)
    run_time_AH = np.array(run_time_AH)
    S_t_X_RSF = np.array(S_t_X_RSF)
    DMS_RSF = np.array(DMS_RSF)
    IBS_RSF = np.array(IBS_RSF)
    run_time_RSF = np.array(run_time_RSF)
    S_t_X_PAFT = np.array(S_t_X_PAFT)
    DMS_PAFT = np.array(DMS_PAFT)
    IBS_PAFT = np.array(IBS_PAFT)
    run_time_PAFT = np.array(run_time_PAFT)
    S_t_X_SAFT = np.array(S_t_X_SAFT)
    DMS_SAFT = np.array(DMS_SAFT)
    IBS_SAFT = np.array(IBS_SAFT)
    run_time_SAFT = np.array(run_time_SAFT)


    np.save(os.path.join(results_folder, f'{file_prefix}_S_t_X_Cox_Linear.npy'), S_t_X_Cox_Linear)
    np.save(os.path.join(results_folder, f'{file_prefix}_DMS_Cox_Linear.npy'), DMS_Cox_Linear)
    np.save(os.path.join(results_folder, f'{file_prefix}_IBS_Cox_Linear.npy'), IBS_Cox_Linear)
    np.save(os.path.join(results_folder, f'{file_prefix}_run_time_Cox_Linear.npy'), run_time_Cox_Linear)
    np.save(os.path.join(results_folder, f'{file_prefix}_S_t_X_Cox_CC.npy'), S_t_X_Cox_CC)
    np.save(os.path.join(results_folder, f'{file_prefix}_DMS_Cox_CC.npy'), DMS_Cox_CC)
    np.save(os.path.join(results_folder, f'{file_prefix}_IBS_Cox_CC.npy'), IBS_Cox_CC)
    np.save(os.path.join(results_folder, f'{file_prefix}_run_time_Cox_CC.npy'), run_time_Cox_CC)
    np.save(os.path.join(results_folder, f'{file_prefix}_S_t_X_Cox_Time.npy'), S_t_X_Cox_Time)
    np.save(os.path.join(results_folder, f'{file_prefix}_DMS_Cox_Time.npy'), DMS_Cox_Time)
    np.save(os.path.join(results_folder, f'{file_prefix}_IBS_Cox_Time.npy'), IBS_Cox_Time)
    np.save(os.path.join(results_folder, f'{file_prefix}_run_time_Cox_Time.npy'), run_time_Cox_Time)
    np.save(os.path.join(results_folder, f'{file_prefix}_S_t_X_DeepSurv.npy'), S_t_X_DeepSurv)
    np.save(os.path.join(results_folder, f'{file_prefix}_DMS_DeepSurv.npy'), DMS_DeepSurv)
    np.save(os.path.join(results_folder, f'{file_prefix}_IBS_DeepSurv.npy'), IBS_DeepSurv)
    np.save(os.path.join(results_folder, f'{file_prefix}_run_time_DeepSurv.npy'), run_time_DeepSurv)
    np.save(os.path.join(results_folder, f'{file_prefix}_S_t_X_DeepHit.npy'), S_t_X_DeepHit)
    np.save(os.path.join(results_folder, f'{file_prefix}_DMS_DeepHit.npy'), DMS_DeepHit)
    np.save(os.path.join(results_folder, f'{file_prefix}_IBS_DeepHit.npy'), IBS_DeepHit)
    np.save(os.path.join(results_folder, f'{file_prefix}_run_time_DeepHit.npy'), run_time_DeepHit)
    np.save(os.path.join(results_folder, f'{file_prefix}_S_t_X_AH.npy'), S_t_X_AH)
    np.save(os.path.join(results_folder, f'{file_prefix}_DMS_AH.npy'), DMS_AH)
    np.save(os.path.join(results_folder, f'{file_prefix}_IBS_AH.npy'), IBS_AH)
    np.save(os.path.join(results_folder, f'{file_prefix}_run_time_AH.npy'), run_time_AH)
    np.save(os.path.join(results_folder, f'{file_prefix}_S_t_X_RSF.npy'), S_t_X_RSF)
    np.save(os.path.join(results_folder, f'{file_prefix}_DMS_RSF.npy'), DMS_RSF)
    np.save(os.path.join(results_folder, f'{file_prefix}_IBS_RSF.npy'), IBS_RSF)
    np.save(os.path.join(results_folder, f'{file_prefix}_run_time_RSF.npy'), run_time_RSF)
    np.save(os.path.join(results_folder, f'{file_prefix}_S_t_X_PAFT.npy'), S_t_X_PAFT)
    np.save(os.path.join(results_folder, f'{file_prefix}_DMS_PAFT.npy'), DMS_PAFT)
    np.save(os.path.join(results_folder, f'{file_prefix}_IBS_PAFT.npy'), IBS_PAFT)
    np.save(os.path.join(results_folder, f'{file_prefix}_run_time_PAFT.npy'), run_time_PAFT)
    np.save(os.path.join(results_folder, f'{file_prefix}_S_t_X_SAFT.npy'), S_t_X_SAFT)
    np.save(os.path.join(results_folder, f'{file_prefix}_DMS_SAFT.npy'), DMS_SAFT)
    np.save(os.path.join(results_folder, f'{file_prefix}_IBS_SAFT.npy'), IBS_SAFT)
    np.save(os.path.join(results_folder, f'{file_prefix}_run_time_SAFT.npy'), run_time_SAFT)
    


process_results(results_400, "n400")
process_results(results_800, "n800")





import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

Set_n = np.array([400, 800]) 


results_folder = "results_folder"

file_names = [

    "n400_DMS_Cox_Linear.npy",
    "n800_DMS_Cox_Linear.npy",
    "n400_DMS_Cox_CC.npy",
    "n800_DMS_Cox_CC.npy",
    "n400_DMS_Cox_Time.npy",
    "n800_DMS_Cox_Time.npy",
    "n400_DMS_DeepSurv.npy",
    "n800_DMS_DeepSurv.npy",
    "n400_DMS_DeepHit.npy",
    "n800_DMS_DeepHit.npy",
    "n400_DMS_AH.npy",
    "n800_DMS_AH.npy",
    "n400_DMS_RSF.npy",
    "n800_DMS_RSF.npy",
    "n400_DMS_PAFT.npy",
    "n800_DMS_PAFT.npy",
    "n400_DMS_SAFT.npy",
    "n800_DMS_SAFT.npy",

    "n400_IBS_Cox_Linear.npy",
    "n800_IBS_Cox_Linear.npy",
    "n400_IBS_Cox_CC.npy",
    "n800_IBS_Cox_CC.npy",
    "n400_IBS_Cox_Time.npy",
    "n800_IBS_Cox_Time.npy",
    "n400_IBS_DeepSurv.npy",
    "n800_IBS_DeepSurv.npy",
    "n400_IBS_DeepHit.npy",
    "n800_IBS_DeepHit.npy",
    "n400_IBS_AH.npy",
    "n800_IBS_AH.npy",
    "n400_IBS_RSF.npy",
    "n800_IBS_RSF.npy",
    "n400_IBS_PAFT.npy",
    "n800_IBS_PAFT.npy",
    "n400_IBS_SAFT.npy",
    "n800_IBS_SAFT.npy",
   
    "n400_S_t_X_Cox_Linear.npy",
    "n800_S_t_X_Cox_Linear.npy",
    "n400_S_t_X_Cox_CC.npy",
    "n800_S_t_X_Cox_CC.npy",
    "n400_S_t_X_Cox_Time.npy",
    "n800_S_t_X_Cox_Time.npy",
    "n400_S_t_X_DeepSurv.npy",
    "n800_S_t_X_DeepSurv.npy",
    "n400_S_t_X_DeepHit.npy",
    "n800_S_t_X_DeepHit.npy",
    "n400_S_t_X_AH.npy",
    "n800_S_t_X_AH.npy",
    "n400_S_t_X_RSF.npy",
    "n800_S_t_X_RSF.npy",
    "n400_S_t_X_PAFT.npy",
    "n800_S_t_X_PAFT.npy",
    "n400_S_t_X_SAFT.npy",
    "n800_S_t_X_SAFT.npy",

    "n400_run_time_Cox_Linear.npy",
    "n800_run_time_Cox_Linear.npy",
    "n400_run_time_Cox_CC.npy",
    "n800_run_time_Cox_CC.npy",
    "n400_run_time_Cox_Time.npy",
    "n800_run_time_Cox_Time.npy",
    "n400_run_time_DeepSurv.npy",
    "n800_run_time_DeepSurv.npy",
    "n400_run_time_DeepHit.npy",
    "n800_run_time_DeepHit.npy",
    "n400_run_time_AH.npy",
    "n800_run_time_AH.npy",
    "n400_run_time_RSF.npy",
    "n800_run_time_RSF.npy",
    "n400_run_time_PAFT.npy",
    "n800_run_time_PAFT.npy",
    "n400_run_time_SAFT.npy",
    "n800_run_time_SAFT.npy",

    "St_true_X_fig.npy",
    "t_fig.npy",
]

data = {}
for file_name in file_names:
    file_path = os.path.join(results_folder, file_name)  
    data[file_name] = np.load(file_path) 


DMS_Cox_Linear_400_B = data["n400_DMS_Cox_Linear.npy"]
DMS_Cox_Linear_800_B = data["n800_DMS_Cox_Linear.npy"]
DMS_Cox_CC_400_B = data["n400_DMS_Cox_CC.npy"]
DMS_Cox_CC_800_B = data["n800_DMS_Cox_CC.npy"]
DMS_Cox_Time_400_B = data["n400_DMS_Cox_Time.npy"]
DMS_Cox_Time_800_B = data["n800_DMS_Cox_Time.npy"]
DMS_DeepSurv_400_B = data["n400_DMS_DeepSurv.npy"]
DMS_DeepSurv_800_B = data["n800_DMS_DeepSurv.npy"]
DMS_DeepHit_400_B = data["n400_DMS_DeepHit.npy"]
DMS_DeepHit_800_B = data["n800_DMS_DeepHit.npy"]
DMS_AH_400_B = data["n400_DMS_AH.npy"]
DMS_AH_800_B = data["n800_DMS_AH.npy"]
DMS_RSF_400_B = data["n400_DMS_RSF.npy"]
DMS_RSF_800_B = data["n800_DMS_RSF.npy"]
DMS_PAFT_400_B = data["n400_DMS_PAFT.npy"]
DMS_PAFT_800_B = data["n800_DMS_PAFT.npy"]
DMS_SAFT_400_B = data["n400_DMS_SAFT.npy"]
DMS_SAFT_800_B = data["n800_DMS_SAFT.npy"]


IBS_Cox_Linear_400_B = data["n400_IBS_Cox_Linear.npy"]
IBS_Cox_Linear_800_B = data["n800_IBS_Cox_Linear.npy"]
IBS_Cox_CC_400_B = data["n400_IBS_Cox_CC.npy"]
IBS_Cox_CC_800_B = data["n800_IBS_Cox_CC.npy"]
IBS_Cox_Time_400_B = data["n400_IBS_Cox_Time.npy"]
IBS_Cox_Time_800_B = data["n800_IBS_Cox_Time.npy"]
IBS_DeepSurv_400_B = data["n400_IBS_DeepSurv.npy"]
IBS_DeepSurv_800_B = data["n800_IBS_DeepSurv.npy"]
IBS_DeepHit_400_B = data["n400_IBS_DeepHit.npy"]
IBS_DeepHit_800_B = data["n800_IBS_DeepHit.npy"]
IBS_AH_400_B = data["n400_IBS_AH.npy"]
IBS_AH_800_B = data["n800_IBS_AH.npy"]
IBS_RSF_400_B = data["n400_IBS_RSF.npy"]
IBS_RSF_800_B = data["n800_IBS_RSF.npy"]
IBS_PAFT_400_B = data["n400_IBS_PAFT.npy"]
IBS_PAFT_800_B = data["n800_IBS_PAFT.npy"]
IBS_SAFT_400_B = data["n400_IBS_SAFT.npy"]
IBS_SAFT_800_B = data["n800_IBS_SAFT.npy"]


St_Cox_Linear_X_400_B = data["n400_S_t_X_Cox_Linear.npy"]
St_Cox_Linear_X_800_B = data["n800_S_t_X_Cox_Linear.npy"]
St_Cox_CC_X_400_B = data["n400_S_t_X_Cox_CC.npy"]
St_Cox_CC_X_800_B = data["n800_S_t_X_Cox_CC.npy"]
St_Cox_Time_X_400_B = data["n400_S_t_X_Cox_Time.npy"]
St_Cox_Time_X_800_B = data["n800_S_t_X_Cox_Time.npy"]
St_DeepSurv_X_400_B = data["n400_S_t_X_DeepSurv.npy"]
St_DeepSurv_X_800_B = data["n800_S_t_X_DeepSurv.npy"]
St_DeepHit_X_400_B = data["n400_S_t_X_DeepHit.npy"]
St_DeepHit_X_800_B = data["n800_S_t_X_DeepHit.npy"]
St_AH_X_400_B = data["n400_S_t_X_AH.npy"]
St_AH_X_800_B = data["n800_S_t_X_AH.npy"]
St_RSF_X_400_B = data["n400_S_t_X_RSF.npy"]
St_RSF_X_800_B = data["n800_S_t_X_RSF.npy"]
St_PAFT_X_400_B = data["n400_S_t_X_PAFT.npy"]
St_PAFT_X_800_B = data["n800_S_t_X_PAFT.npy"]
St_SAFT_X_400_B = data["n400_S_t_X_SAFT.npy"]
St_SAFT_X_800_B = data["n800_S_t_X_SAFT.npy"]


run_time_Cox_Linear_400_B = data["n400_run_time_Cox_Linear.npy"]
run_time_Cox_Linear_800_B = data["n800_run_time_Cox_Linear.npy"]
run_time_Cox_CC_400_B = data["n400_run_time_Cox_CC.npy"]
run_time_Cox_CC_800_B = data["n800_run_time_Cox_CC.npy"]
run_time_Cox_Time_400_B = data["n400_run_time_Cox_Time.npy"]
run_time_Cox_Time_800_B = data["n800_run_time_Cox_Time.npy"]
run_time_DeepSurv_400_B = data["n400_run_time_DeepSurv.npy"]
run_time_DeepSurv_800_B = data["n800_run_time_DeepSurv.npy"]
run_time_DeepHit_400_B = data["n400_run_time_DeepHit.npy"]
run_time_DeepHit_800_B = data["n800_run_time_DeepHit.npy"]
run_time_AH_400_B = data["n400_run_time_AH.npy"]
run_time_AH_800_B = data["n800_run_time_AH.npy"]
run_time_RSF_400_B = data["n400_run_time_RSF.npy"]
run_time_RSF_800_B = data["n800_run_time_RSF.npy"]
run_time_PAFT_400_B = data["n400_run_time_PAFT.npy"]
run_time_PAFT_800_B = data["n800_run_time_PAFT.npy"]
run_time_SAFT_400_B = data["n400_run_time_SAFT.npy"]
run_time_SAFT_800_B = data["n800_run_time_SAFT.npy"]


St_true_X_fig = data["St_true_X_fig.npy"]
t_fig = data["t_fig.npy"]



DMS_Cox_Linears = np.array([np.nanmean(DMS_Cox_Linear_400_B), np.nanmean(DMS_Cox_Linear_800_B)])
DMS_Cox_Linears_sd= np.array([np.sqrt(np.nanmean((DMS_Cox_Linear_400_B-np.nanmean(DMS_Cox_Linear_400_B))**2)), np.sqrt(np.nanmean((DMS_Cox_Linear_800_B-np.nanmean(DMS_Cox_Linear_800_B))**2))])
DMS_Cox_CCs = np.array([np.nanmean(DMS_Cox_CC_400_B), np.nanmean(DMS_Cox_CC_800_B)])
DMS_Cox_CCs_sd= np.array([np.sqrt(np.nanmean((DMS_Cox_CC_400_B-np.nanmean(DMS_Cox_CC_400_B))**2)), np.sqrt(np.nanmean((DMS_Cox_CC_800_B-np.nanmean(DMS_Cox_CC_800_B))**2))])
DMS_Cox_Times = np.array([np.nanmean(DMS_Cox_Time_400_B), np.nanmean(DMS_Cox_Time_800_B)])
DMS_Cox_Times_sd= np.array([np.sqrt(np.nanmean((DMS_Cox_Time_400_B-np.nanmean(DMS_Cox_Time_400_B))**2)), np.sqrt(np.nanmean((DMS_Cox_Time_800_B-np.nanmean(DMS_Cox_Time_800_B))**2))])
DMS_DeepSurvs = np.array([np.nanmean(DMS_DeepSurv_400_B), np.nanmean(DMS_DeepSurv_800_B)])
DMS_DeepSurvs_sd= np.array([np.sqrt(np.nanmean((DMS_DeepSurv_400_B-np.nanmean(DMS_DeepSurv_400_B))**2)), np.sqrt(np.nanmean((DMS_DeepSurv_800_B-np.nanmean(DMS_DeepSurv_800_B))**2))])
DMS_DeepHits = np.array([np.nanmean(DMS_DeepHit_400_B), np.nanmean(DMS_DeepHit_800_B)])
DMS_DeepHits_sd= np.array([np.sqrt(np.nanmean((DMS_DeepHit_400_B-np.nanmean(DMS_DeepHit_400_B))**2)), np.sqrt(np.nanmean((DMS_DeepHit_800_B-np.nanmean(DMS_DeepHit_800_B))**2))])
DMS_AHs = np.array([np.nanmean(DMS_AH_400_B), np.nanmean(DMS_AH_800_B)])
DMS_AHs_sd= np.array([np.sqrt(np.nanmean((DMS_AH_400_B-np.nanmean(DMS_AH_400_B))**2)), np.sqrt(np.nanmean((DMS_AH_800_B-np.nanmean(DMS_AH_800_B))**2))])
DMS_RSFs = np.array([np.nanmean(DMS_RSF_400_B), np.nanmean(DMS_RSF_800_B)])
DMS_RSFs_sd= np.array([np.sqrt(np.nanmean((DMS_RSF_400_B-np.nanmean(DMS_RSF_400_B))**2)), np.sqrt(np.nanmean((DMS_RSF_800_B-np.nanmean(DMS_RSF_800_B))**2))])
DMS_PAFTs = np.array([np.nanmean(DMS_PAFT_400_B), np.nanmean(DMS_PAFT_800_B)])
DMS_PAFTs_sd= np.array([np.sqrt(np.nanmean((DMS_PAFT_400_B-np.nanmean(DMS_PAFT_400_B))**2)), np.sqrt(np.nanmean((DMS_PAFT_800_B-np.nanmean(DMS_PAFT_800_B))**2))])
DMS_SAFTs = np.array([np.nanmean(DMS_SAFT_400_B), np.nanmean(DMS_SAFT_800_B)])
DMS_SAFTs_sd= np.array([np.sqrt(np.nanmean((DMS_SAFT_400_B-np.nanmean(DMS_SAFT_400_B))**2)), np.sqrt(np.nanmean((DMS_SAFT_800_B-np.nanmean(DMS_SAFT_800_B))**2))])

#计算IBS
IBS_Cox_Linears = np.array([np.nanmean(IBS_Cox_Linear_400_B), np.nanmean(IBS_Cox_Linear_800_B)])
IBS_Cox_Linears_sd= np.array([np.sqrt(np.nanmean((IBS_Cox_Linear_400_B-np.nanmean(IBS_Cox_Linear_400_B))**2)), np.sqrt(np.nanmean((IBS_Cox_Linear_800_B-np.nanmean(IBS_Cox_Linear_800_B))**2))])
IBS_Cox_CCs = np.array([np.nanmean(IBS_Cox_CC_400_B), np.nanmean(IBS_Cox_CC_800_B)])
IBS_Cox_CCs_sd= np.array([np.sqrt(np.nanmean((IBS_Cox_CC_400_B-np.nanmean(IBS_Cox_CC_400_B))**2)), np.sqrt(np.nanmean((IBS_Cox_CC_800_B-np.nanmean(IBS_Cox_CC_800_B))**2))])
IBS_Cox_Times = np.array([np.nanmean(IBS_Cox_Time_400_B), np.nanmean(IBS_Cox_Time_800_B)])
IBS_Cox_Times_sd= np.array([np.sqrt(np.nanmean((IBS_Cox_Time_400_B-np.nanmean(IBS_Cox_Time_400_B))**2)), np.sqrt(np.nanmean((IBS_Cox_Time_800_B-np.nanmean(IBS_Cox_Time_800_B))**2))])
IBS_DeepSurvs = np.array([np.nanmean(IBS_DeepSurv_400_B), np.nanmean(IBS_DeepSurv_800_B)])
IBS_DeepSurvs_sd= np.array([np.sqrt(np.nanmean((IBS_DeepSurv_400_B-np.nanmean(IBS_DeepSurv_400_B))**2)), np.sqrt(np.nanmean((IBS_DeepSurv_800_B-np.nanmean(IBS_DeepSurv_800_B))**2))])
IBS_DeepHits = np.array([np.nanmean(IBS_DeepHit_400_B), np.nanmean(IBS_DeepHit_800_B)])
IBS_DeepHits_sd= np.array([np.sqrt(np.nanmean((IBS_DeepHit_400_B-np.nanmean(IBS_DeepHit_400_B))**2)), np.sqrt(np.nanmean((IBS_DeepHit_800_B-np.nanmean(IBS_DeepHit_800_B))**2))])
IBS_AHs = np.array([np.nanmean(IBS_AH_400_B), np.nanmean(IBS_AH_800_B)])
IBS_AHs_sd= np.array([np.sqrt(np.nanmean((IBS_AH_400_B-np.nanmean(IBS_AH_400_B))**2)), np.sqrt(np.nanmean((IBS_AH_800_B-np.nanmean(IBS_AH_800_B))**2))])
IBS_RSFs = np.array([np.nanmean(IBS_RSF_400_B), np.nanmean(IBS_RSF_800_B)])
IBS_RSFs_sd= np.array([np.sqrt(np.nanmean((IBS_RSF_400_B-np.nanmean(IBS_RSF_400_B))**2)), np.sqrt(np.nanmean((IBS_RSF_800_B-np.nanmean(IBS_RSF_800_B))**2))])
IBS_PAFTs = np.array([np.nanmean(IBS_PAFT_400_B), np.nanmean(IBS_PAFT_800_B)])
IBS_PAFTs_sd= np.array([np.sqrt(np.nanmean((IBS_PAFT_400_B-np.nanmean(IBS_PAFT_400_B))**2)), np.sqrt(np.nanmean((IBS_PAFT_800_B-np.nanmean(IBS_PAFT_800_B))**2))])
IBS_SAFTs = np.array([np.nanmean(IBS_SAFT_400_B), np.nanmean(IBS_SAFT_800_B)])
IBS_SAFTs_sd= np.array([np.sqrt(np.nanmean((IBS_SAFT_400_B-np.nanmean(IBS_SAFT_400_B))**2)), np.sqrt(np.nanmean((IBS_SAFT_800_B-np.nanmean(IBS_SAFT_800_B))**2))])


run_time_Cox_Linears = np.array([np.nanmean(run_time_Cox_Linear_400_B), np.nanmean(run_time_Cox_Linear_800_B)])
run_time_Cox_CCs = np.array([np.nanmean(run_time_Cox_CC_400_B), np.nanmean(run_time_Cox_CC_800_B)])
run_time_Cox_Times = np.array([np.nanmean(run_time_Cox_Time_400_B), np.nanmean(run_time_Cox_Time_800_B)])
run_time_DeepSurvs = np.array([np.nanmean(run_time_DeepSurv_400_B), np.nanmean(run_time_DeepSurv_800_B)])
run_time_DeepHits = np.array([np.nanmean(run_time_DeepHit_400_B), np.nanmean(run_time_DeepHit_800_B)])
run_time_AHs = np.array([np.nanmean(run_time_AH_400_B), np.nanmean(run_time_AH_800_B)])
run_time_RSFs = np.array([np.nanmean(run_time_RSF_400_B), np.nanmean(run_time_RSF_800_B)])
run_time_PAFTs = np.array([np.nanmean(run_time_PAFT_400_B), np.nanmean(run_time_PAFT_800_B)])
run_time_SAFTs = np.array([np.nanmean(run_time_SAFT_400_B), np.nanmean(run_time_SAFT_800_B)])

# =================tables=======================
output_folder = "Individual_9results"
os.makedirs(output_folder, exist_ok=True)

dic_DMS = {
    "n": Set_n,
    "DMS_Cox_Linear": np.array(DMS_Cox_Linears),
    "DMS_Cox_CC": np.array(DMS_Cox_CCs),
    "DMS_Cox_Time": np.array(DMS_Cox_Times),
    "DMS_DeepSurv": np.array(DMS_DeepSurvs),
    "DMS_DeepHit": np.array(DMS_DeepHits),
    "DMS_RSF": np.array(DMS_RSFs),
    "DMS_PAFT": np.array(DMS_PAFTs),
    "DMS_SAFT": np.array(DMS_SAFTs),
    "DMS_AH": np.array(DMS_AHs),
}
result_DMS = pd.DataFrame(dic_DMS)
DMS_path = os.path.join(output_folder, "Nine_DMS.csv")
result_DMS.to_csv(DMS_path, index=False)


dic_IBS = {
    "n": Set_n,
    "IBS_Cox_Linear": np.array(IBS_Cox_Linears),
    "IBS_Cox_CC": np.array(IBS_Cox_CCs),
    "IBS_Cox_Time": np.array(IBS_Cox_Times),
    "IBS_DeepSurv": np.array(IBS_DeepSurvs),
    "IBS_DeepHit": np.array(IBS_DeepHits),
    "IBS_RSF": np.array(IBS_RSFs),
    "IBS_PAFT": np.array(IBS_PAFTs),
    "IBS_SAFT": np.array(IBS_SAFTs),
    "IBS_AH": np.array(IBS_AHs),
}
result_IBS = pd.DataFrame(dic_IBS)
ibs_path = os.path.join(output_folder, "Nine_IBS.csv")
result_IBS.to_csv(ibs_path, index=False)


dic_run_time = {
    "n": Set_n,
    "run_time_Cox_Linear": np.array(run_time_Cox_Linears),
    "run_time_Cox_CC": np.array(run_time_Cox_CCs),
    "run_time_Cox_Time": np.array(run_time_Cox_Times),
    "run_time_DeepSurv": np.array(run_time_DeepSurvs),
    "run_time_DeepHit": np.array(run_time_DeepHits),
    "run_time_RSF": np.array(run_time_RSFs),
    "run_time_PAFT": np.array(run_time_PAFTs),
    "run_time_SAFT": np.array(run_time_SAFTs),
    "run_time_AH": np.array(run_time_AHs),
}
result_run_time = pd.DataFrame(dic_run_time)
run_time_path = os.path.join(output_folder, "Nine_run_time.csv")
result_run_time.to_csv(run_time_path, index=False)


# =================pictures=======================
subfolder = "Individual_9results"

St_Cox_Linear_X_400 = np.nanmean(St_Cox_Linear_X_400_B, axis=0)
St_Cox_Linear_X_800 = np.nanmean(St_Cox_Linear_X_800_B, axis=0)
St_Cox_CC_X_400 = np.nanmean(St_Cox_CC_X_400_B, axis=0)
St_Cox_CC_X_800 = np.nanmean(St_Cox_CC_X_800_B, axis=0)
St_Cox_Time_X_400 = np.nanmean(St_Cox_Time_X_400_B, axis=0)
St_Cox_Time_X_800 = np.nanmean(St_Cox_Time_X_800_B, axis=0)
St_DeepSurv_X_400 = np.nanmean(St_DeepSurv_X_400_B, axis=0)
St_DeepSurv_X_800 = np.nanmean(St_DeepSurv_X_800_B, axis=0)
St_DeepHit_X_400 = np.nanmean(St_DeepHit_X_400_B, axis=0)
St_DeepHit_X_800 = np.nanmean(St_DeepHit_X_800_B, axis=0)
St_AH_X_400 = np.nanmean(St_AH_X_400_B, axis=0)
St_AH_X_800 = np.nanmean(St_AH_X_800_B, axis=0)
St_RSF_X_400 = np.nanmean(St_RSF_X_400_B, axis=0)
St_RSF_X_800 = np.nanmean(St_RSF_X_800_B, axis=0)
St_PAFT_X_400 = np.nanmean(St_PAFT_X_400_B, axis=0)
St_PAFT_X_800 = np.nanmean(St_PAFT_X_800_B, axis=0)
St_SAFT_X_400 = np.nanmean(St_SAFT_X_400_B, axis=0)
St_SAFT_X_800 = np.nanmean(St_SAFT_X_800_B, axis=0)




for k in range(10):
    fig1 = plt.figure() 
    fig1.suptitle("(f) AH I", fontsize=10) 
    # -----n=400, X-----
    ax1_1 = fig1.add_subplot(1, 2, 1)
    ax1_1.set_title('n=400', fontsize=8, loc='center')  
    ax1_1.set_xlabel("t", fontsize=8)  
    ax1_1.set_ylabel('Conditional Survival Function', fontsize=8) 
    ax1_1.tick_params(axis='both', labelsize=6) 
    # -----n=800, X-----
    ax1_2 = fig1.add_subplot(1, 2, 2)
    ax1_2.set_title('n=800', fontsize=8, loc='center')  
    ax1_2.set_xlabel("t", fontsize=8)  
    ax1_2.tick_params(axis='both', labelsize=6)
    # n=400
    ax1_1.plot(t_fig, St_true_X_fig[:, k], color='black', label='True', linestyle='-')
    ax1_1.plot(t_fig, St_Cox_Linear_X_400[:, k], color='red', label='Cox Linear', linestyle=':')
    ax1_1.plot(t_fig, St_Cox_CC_X_400[:, k], color='chocolate', label='Cox-CC', linestyle='--')
    ax1_1.plot(t_fig, St_Cox_Time_X_400[:, k], color='teal', label='Cox-Time', linestyle='--')
    ax1_1.plot(t_fig, St_DeepSurv_X_400[:, k], color='purple', label='DeepSurv', linestyle='--')
    ax1_1.plot(t_fig, St_DeepHit_X_400[:, k], color='pink', label='DeepHit', linestyle='--')
    ax1_1.plot(t_fig, St_RSF_X_400[:, k], color= 'orange', label='RSF', linestyle='--')
    ax1_1.plot(t_fig, St_PAFT_X_400[:, k], color='gray', label='PAFT', linestyle='--')
    ax1_1.plot(t_fig, St_SAFT_X_400[:, k], color='blue', label='SAFT', linestyle='--')
    ax1_1.plot(t_fig, St_AH_X_400[:, k], color='red', label='AH', linestyle='--')
    ax1_1.legend(loc='best', fontsize=6)
    # n=800
    ax1_2.plot(t_fig, St_true_X_fig[:, k], color='black', label='True', linestyle='-')
    ax1_2.plot(t_fig, St_Cox_Linear_X_800[:, k], color='red', label='Cox Linear', linestyle=':')
    ax1_2.plot(t_fig, St_Cox_CC_X_800[:, k], color='chocolate', label='Cox-CC', linestyle='--')
    ax1_2.plot(t_fig, St_Cox_Time_X_800[:, k], color='teal', label='Cox-Time', linestyle='--')
    ax1_2.plot(t_fig, St_DeepSurv_X_800[:, k], color='purple', label='DeepSurv', linestyle='--')
    ax1_2.plot(t_fig, St_DeepHit_X_800[:, k], color='pink', label='DeepHit', linestyle='--')
    ax1_2.plot(t_fig, St_RSF_X_800[:, k], color= 'orange', label='RSF', linestyle='--')
    ax1_2.plot(t_fig, St_PAFT_X_400[:, k], color='gray', label='PAFT', linestyle='--')
    ax1_2.plot(t_fig, St_SAFT_X_400[:, k], color='blue', label='SAFT', linestyle='--')
    ax1_2.plot(t_fig, St_AH_X_400[:, k], color='red', label='AH', linestyle='--')
    ax1_2.legend(loc='best', fontsize=6)
    file_name = os.path.join(subfolder, f'Nine_Case6_fig_{k}.jpeg') 
    fig1.savefig(file_name, dpi=300, bbox_inches='tight') 
