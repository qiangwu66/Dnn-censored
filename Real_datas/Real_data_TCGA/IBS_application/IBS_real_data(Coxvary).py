# python 3.12.8
#%% ----------------------
import numpy as np
import random
import torch
import time

from B_spline import *
import pandas as pd
import os
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d

from Survival_methods.Cox_varytime import Surv_Coxvarying


results_folder = "results_folder"
os.makedirs(results_folder, exist_ok=True)

#%% ----------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)   

set_seed(5)

#%% Data Processing
df = pd.read_csv('Survival_time.csv') # read csv file

# X = np.array(df[['male', 'age', 'sps', 'scoma', 'disease1', 'disease2', 'disease3']], dtype='float32')
De = 1 - np.array(df['censored'], dtype='float32')
Time = np.array(df['Survival_time'], dtype='float32')
Time = Time / np.max(Time)

X_variables = np.loadtxt("X_scaled.csv", delimiter=",")

# X = X_variables[:, :50]
X = X_variables.astype('float32')
# X = X / np.max(X)

tau = 1
# random order
A = np.arange(len(Time))
np.random.shuffle(A)
X_R = X[A]
De_R = De[A]
Time_R = Time[A]

# -------training data: 64%  validation data: 16%  test data: 20%---------------------------------
# ---training data 442
X_R_train = X_R[np.arange(442)]
De_R_train = De_R[np.arange(442)]
Time_R_train = Time_R[np.arange(442)]
# ---validation data 1456
X_R_valid = X_R[np.arange(442, 552)]
De_R_valid = De_R[np.arange(442, 552)]
Time_R_valid = Time_R[np.arange(442, 552)]
# ---test data 1820
X_R_test = X_R[np.arange(552, 736)]
De_R_test = De_R[np.arange(552, 736)]
Time_R_test = Time_R[np.arange(552, 736)]


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

# ----------------Coxvarying-------------------------
start_time_Coxvary = time.time()

m_vary = 6
nodevec_vary = np.array(np.linspace(0, tau, m_vary + 2), dtype="float32")  
S_Coxvary = Surv_Coxvarying(X_R_train, Time_R_train, De_R_train, X_R_test, t_nodes, m_vary, nodevec_vary, tau, s_k)

# IBS_Coxvary
S_t_X_Coxvary_IBS = S_Coxvary["S_t_X_Coxvary_IBS"]
IBS_Coxvary = np.nanmean(S_t_X_Coxvary_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - S_t_X_Coxvary_IBS) ** 2 * (1 - I_T_i_s_k) / np.tile(G_s_k, (len(Time_R_test), 1)).T)

end_time_Coxvary = time.time()
run_time_Coxvary = end_time_Coxvary - start_time_Coxvary

# =================IBS-Table=======================
output_folder = "IBS_13methods"
os.makedirs(output_folder, exist_ok=True)


dic_IBS = {
    "IBS_Coxvarying": np.array([IBS_Coxvary, run_time_Coxvary]),
}
result_IBS = pd.DataFrame(dic_IBS)
ibs_path = os.path.join(output_folder, "IBS_realdata_coxvary.csv")
result_IBS.to_csv(ibs_path, index=False)

