# python 3.12.8
#%% ----------------------
import numpy as np
import random
import torch
from B_spline import *
import pandas as pd
import os
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d
from Survival_methods.DNN_iteration_AGQ import g_dnn_agq
import time

results_folder = "results_folder"
os.makedirs(results_folder, exist_ok=True)

#%% ----------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)   

set_seed(1)

#%% Data Processing
df = pd.read_csv('support_clean_death_as_event_all_x.csv') # read csv file

X = np.array(df[['male', 'age', 'sps', 'scoma', 'disease1', 'disease2', 'disease3']], dtype='float32')
De = np.array(df['delta'], dtype='float32')
Time = np.array(df['d.time'], dtype='float32')
Time = Time / np.max(Time)

X[:,1] = X[:,1] / np.max(X[:,1])
X[:,2] = X[:,2] / np.max(X[:,2])
X[:,3] = X[:,3] / np.max(X[:,3])

tau = 1
# random order
A = np.arange(len(Time))
np.random.shuffle(A)
X_R = X[A]
De_R = De[A]
Time_R = Time[A]

# -------training data: 64%  validation data: 16%  test data: 20%---------------------------------
# ---training data 5828
X_R_train = X_R[np.arange(5828)]
De_R_train = De_R[np.arange(5828)]
Time_R_train = Time_R[np.arange(5828)]
# ---validation data 1456
X_R_valid = X_R[np.arange(5828, 7284)]
De_R_valid = De_R[np.arange(5828, 7284)]
Time_R_valid = Time_R[np.arange(5828, 7284)]
# ---test data 1820
X_R_test = X_R[np.arange(7284, 9104)]
De_R_test = De_R[np.arange(7284, 9104)]
Time_R_test = Time_R[np.arange(7284, 9104)]


#%%
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


#%% Kaplan-Meier
kmf = KaplanMeierFitter()
kmf.fit(Time_R_test, event_observed = 1 - De_R_test)
G_T_i = np.maximum(kmf.predict(Time_R_test).values, np.min(kmf.predict(Time_R_test).values[kmf.predict(Time_R_test).values > 0])) 

# G_s_k 
original_times = kmf.survival_function_.index.values[1:]  
original_survival_probs = kmf.survival_function_["KM_estimate"].values[1:]  

s_k = np.linspace(Time_R_test.min(), Time_R_test.max(), 100, endpoint=False)
s_k = np.array([float(f"{x:.4g}") for x in s_k]) 
s_k = np.maximum(s_k, 1e-4)

interpolator = interp1d(original_times, original_survival_probs, kind="previous", fill_value="extrapolate")
G_s_k = np.maximum(interpolator(s_k), np.min(interpolator(s_k)[interpolator(s_k) > 0]))


I_T_i_s_k = Indicator_matrix(s_k, Time_R_test)
I_T_i_s_k_D_1 = Indicator_matrix(s_k, Time_R_test) * np.tile(De_R_test, (len(s_k), 1)) 

t_nodes = np.array(np.linspace(0, 1, 101), dtype="float32")
t_nodes = np.maximum(t_nodes, 1e-4)



# ---------------DNN----------------------------
start_time_DNN = time.time()

Dnn_layer = 3
Dnn_node = 50
Dnn_epoch = 1000
Dnn_lr = 3e-4
patiences = 10

Est_dnn = g_dnn_agq(X_R_train, De_R_train, Time_R_train, X_R_valid, De_R_valid, Time_R_valid, X_R_test, Time_R_test, s_k, Dnn_layer, Dnn_node, Dnn_lr, Dnn_epoch, patiences, agq_n_low=50, agq_n_high=100, agq_tol=1e-4, agq_max_subdiv=20, clamp_exp_max=None)

IBS_DNN = np.nanmean(Est_dnn['S_T_X_ibs'].T ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - Est_dnn['S_T_X_ibs'].T) ** 2 * (1 - I_T_i_s_k) / np.tile(G_s_k, (len(Time_R_test), 1)).T)

end_time_DNN = time.time()
run_time_DNN = end_time_DNN - start_time_DNN

# =================IBS-Table=======================
output_folder = "IBS_12methods"
os.makedirs(output_folder, exist_ok=True)


dic_IBS = {
    "IBS_DNN": np.array([IBS_DNN, run_time_DNN]),
}
result_IBS = pd.DataFrame(dic_IBS)
ibs_path = os.path.join(output_folder, f"IBS_realdata_DNN_AGQ-{Dnn_layer}-{Dnn_node}-{Dnn_lr}.csv")
result_IBS.to_csv(ibs_path, index=False)

