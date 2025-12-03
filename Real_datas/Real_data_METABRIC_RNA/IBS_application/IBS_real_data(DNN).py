#%% ----------------------
import numpy as np
import random
import torch
from B_spline import *
import pandas as pd
import os
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d
from Survival_methods.DNN_iteration import g_dnn
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
df = pd.read_csv('METABRIC_RNA.csv') # read csv file

De = np.array(df['overall_survival'], dtype='float32')
Time = np.array(df['overall_survival_months'], dtype='float32')
Time = Time / np.max(Time)


cols = df.columns[11:]
sub = df[cols]

col_min = sub.min(axis=0)
col_max = sub.max(axis=0)
denom = col_max - col_min
denom = denom.replace(0, 1e-12)   # 防止分母为0（整列相等）

df[cols] = (sub - col_min) / denom


X_variables = df.iloc[:, 2:].to_numpy()

X = X_variables.astype('float32')


tau = 1
# random order
A = np.arange(len(Time))
np.random.shuffle(A)
X_R = X[A]
De_R = De[A]
Time_R = Time[A]

# -------training data: 70%  validation data: 10%  test data: 20%---------------------------------
# ---training data 1334
X_R_train = X_R[np.arange(1334)]
De_R_train = De_R[np.arange(1334)]
Time_R_train = Time_R[np.arange(1334)]
# ---validation data 190
X_R_valid = X_R[np.arange(1334, 1524)]
De_R_valid = De_R[np.arange(1334, 1524)]
Time_R_valid = Time_R[np.arange(1334, 1524)]
# ---test data 380
X_R_test = X_R[np.arange(1524, 1904)]
De_R_test = De_R[np.arange(1524, 1904)]
Time_R_test = Time_R[np.arange(1524, 1904)]







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

Dnn_layer = 2
Dnn_node = 35
Dnn_epoch = 2000
Dnn_lr = 2e-4
patiences = 10

Est_dnn = g_dnn(X_R_train, De_R_train, Time_R_train, X_R_valid, De_R_valid, Time_R_valid, X_R_test, Time_R_test, s_k, Dnn_layer, Dnn_node, Dnn_lr, Dnn_epoch, patiences)
IBS_DNN = np.nanmean(Est_dnn['S_T_X_ibs'].T ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - Est_dnn['S_T_X_ibs'].T) ** 2 * (1 - I_T_i_s_k) / np.tile(G_s_k, (len(Time_R_test), 1)).T)

end_time_DNN = time.time()
run_time_DNN = end_time_DNN - start_time_DNN

# =================IBS-Table=======================
output_folder = "IBS_13methods"
os.makedirs(output_folder, exist_ok=True)


dic_IBS = {
    "IBS_DNN": np.array([IBS_DNN, run_time_DNN]),
}
result_IBS = pd.DataFrame(dic_IBS)
ibs_path = os.path.join(output_folder, f"IBS_realdata_DNN.csv")
result_IBS.to_csv(ibs_path, index=False)

