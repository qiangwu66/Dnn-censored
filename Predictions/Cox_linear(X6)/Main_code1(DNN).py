# python 3.12.8
#%% ----------------------
import numpy as np
import torch
import matplotlib.pyplot as plt
from data_generator import generate_case_1
from Survival_methods.DNN_iteration import g_dnn
import pandas as pd
import os
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d

import time

from joblib import Parallel, delayed

results_folder = "results_folder"
os.makedirs(results_folder, exist_ok=True)

#%% ----------------------
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
Data_test = generate_case_1(s, corr, tau)
X_test = Data_test['X'] 
f_X_test = Data_test['f_X']
T_O_test = Data_test['T_O']
T_O_test = np.array([float(f"{x:.4g}") for x in T_O_test])
T_O_test = np.maximum(T_O_test, 1e-4)
De_test = Data_test['De']
durations_test, events_test = Data_test['T_O'], Data_test['De'] 


St_true_X_DMS = np.exp(-((T_O_test + 0.001) ** (3 / 2) / 9) * np.exp(f_X_test)) 

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


#%% Draw horizontal coordinate of the survival curve
t_fig = np.array(np.linspace(0, tau, 20), dtype="float32")
t_fig = np.maximum(t_fig, 1e-4)
Lambda_t_true = (t_fig + 0.001) ** (3/2) / 9
St_true_X_fig = np.exp(-np.repeat(Lambda_t_true[:,np.newaxis],X_test.shape[0],axis=1) * np.exp(np.repeat(f_X_test[np.newaxis,:],len(t_fig),axis=0))) 

np.save(os.path.join(results_folder, 't_fig.npy'), t_fig)
np.save(os.path.join(results_folder, 'St_true_X_fig.npy'), St_true_X_fig)

#%% ========================
def single_simulation(b, n, Dnn_layer, Dnn_node, Dnn_epoch, Dnn_lr, t_nodes, t_fig, tau, De_test, St_true_X_DMS, T_O_test, s_k):
    print(f'-------------n={n}, b={b}--------------')
    set_seed(20 + b)
    # -----------------------
    Data_all = generate_case_1(n, corr, tau) 
    
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

    # ---------------DNN----------------------------

    start_time_DNN = time.time()

    Est_dnn = g_dnn(train_data, val_data, Data_test, t_nodes, t_fig, s_k, Dnn_layer, Dnn_node, Dnn_lr, Dnn_epoch, patiences)
    IBS_DNN = np.nanmean(Est_dnn['S_T_X_ibs'].T ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - Est_dnn['S_T_X_ibs'].T) ** 2 * (1 - I_T_i_s_k) / np.tile(G_s_k, (len(T_O_test), 1)).T)

    DMS_DNN = np.sum(De_test * np.abs(St_true_X_DMS - Est_dnn["S_T_X_DMS"]) / np.sum(De_test))

    end_time_DNN = time.time()
    run_time_DNN = end_time_DNN - start_time_DNN

    return {
        "S_t_X_DNN": Est_dnn['S_T_X_fig'],
        "DMS_DNN":  DMS_DNN,
        "IBS_DNN": IBS_DNN,
        "run_time_DNN": run_time_DNN,
    }

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
    

    if ii == 0:  # n=400
        results_400 = Parallel(n_jobs=n_jobs)(
            delayed(single_simulation)(b, n, Dnn_layer, Dnn_node, Dnn_epoch, Dnn_lr, t_nodes, t_fig, tau, De_test, St_true_X_DMS, T_O_test, s_k) 
            for b in range(B)
        )
    else:  # n=800
        results_800 = Parallel(n_jobs=n_jobs)(
            delayed(single_simulation)(b, n, Dnn_layer, Dnn_node, Dnn_epoch, Dnn_lr, t_nodes, t_fig, tau, De_test, St_true_X_DMS, T_O_test, s_k) 
            for b in range(B)
        )

#%% =====================
def process_results(results, file_prefix):
    S_t_X_DNN = []
    DMS_DNN = []
    IBS_DNN = []
    run_time_DNN = []

    for res in results:
        S_t_X_DNN.append(res["S_t_X_DNN"])
        DMS_DNN.append(res["DMS_DNN"])
        IBS_DNN.append(res["IBS_DNN"])
        run_time_DNN.append(res["run_time_DNN"])

    S_t_X_DNN = np.array(S_t_X_DNN)
    DMS_DNN = np.array(DMS_DNN)
    IBS_DNN = np.array(IBS_DNN)
    run_time_DNN = np.array(run_time_DNN)

    np.save(os.path.join(results_folder, f'{file_prefix}_S_t_X_DNN.npy'), S_t_X_DNN)
    np.save(os.path.join(results_folder, f'{file_prefix}_DMS_DNN.npy'), DMS_DNN)
    np.save(os.path.join(results_folder, f'{file_prefix}_IBS_DNN.npy'), IBS_DNN)
    np.save(os.path.join(results_folder, f'{file_prefix}_run_time_DNN.npy'), run_time_DNN)


process_results(results_400, "n400")
process_results(results_800, "n800")






import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

Set_n = np.array([400, 800]) 

results_folder = "results_folder"

file_names = [

    "n400_DMS_DNN.npy",
    "n800_DMS_DNN.npy",

    "n400_IBS_DNN.npy",
    "n800_IBS_DNN.npy",

    "n400_S_t_X_DNN.npy",
    "n800_S_t_X_DNN.npy",

    "n400_run_time_DNN.npy",
    "n800_run_time_DNN.npy",

    "St_true_X_fig.npy",
    "t_fig.npy",
]


data = {}
for file_name in file_names:
    file_path = os.path.join(results_folder, file_name) 
    data[file_name] = np.load(file_path) 


DMS_DNN_400_B = data["n400_DMS_DNN.npy"]
DMS_DNN_800_B = data["n800_DMS_DNN.npy"]

IBS_DNN_400_B = data["n400_IBS_DNN.npy"]
IBS_DNN_800_B = data["n800_IBS_DNN.npy"]

St_DNN_X_400_B = data["n400_S_t_X_DNN.npy"]
St_DNN_X_800_B = data["n800_S_t_X_DNN.npy"]

run_time_DNN_400_B = data["n400_run_time_DNN.npy"]
run_time_DNN_800_B = data["n800_run_time_DNN.npy"]

St_true_X_fig = data["St_true_X_fig.npy"]
t_fig = data["t_fig.npy"]



DMS_DNNs = np.array([np.nanmean(DMS_DNN_400_B), np.nanmean(DMS_DNN_800_B)])
DMS_DNNs_sd= np.array([np.sqrt(np.nanmean((DMS_DNN_400_B-np.nanmean(DMS_DNN_400_B))**2)), np.sqrt(np.nanmean((DMS_DNN_800_B-np.nanmean(DMS_DNN_800_B))**2))])

IBS_DNNs = np.array([np.nanmean(IBS_DNN_400_B), np.nanmean(IBS_DNN_800_B)])
IBS_DNNs_sd= np.array([np.sqrt(np.nanmean((IBS_DNN_400_B-np.nanmean(IBS_DNN_400_B))**2)), np.sqrt(np.nanmean((IBS_DNN_800_B-np.nanmean(IBS_DNN_800_B))**2))])

run_time_DNNs = np.array([np.nanmean(run_time_DNN_400_B), np.nanmean(run_time_DNN_800_B)])

# =================tables=======================

output_folder = "Individual_DNN"
os.makedirs(output_folder, exist_ok=True)

dic_DMS = {
    "n": Set_n,
    "DMS_DNN": np.array(DMS_DNNs),
}
result_DMS = pd.DataFrame(dic_DMS)
DMS_path = os.path.join(output_folder, "DMS_DNN.csv")
result_DMS.to_csv(DMS_path, index=False)

dic_IBS = {
    "n": Set_n,
    "IBS_DNN": np.array(IBS_DNNs),
}
result_IBS = pd.DataFrame(dic_IBS)
ibs_path = os.path.join(output_folder, "IBS_DNN.csv")
result_IBS.to_csv(ibs_path, index=False)


dic_run_time = {
    "n": Set_n,
    "run_time_DNN": np.array(run_time_DNNs),
}
result_run_time = pd.DataFrame(dic_run_time)
run_time_path = os.path.join(output_folder, "run_time_DNN.csv")
result_run_time.to_csv(run_time_path, index=False)

# =================pictures=======================
subfolder = "Individual_DNN"

St_DNN_X_400 = np.nanmean(St_DNN_X_400_B, axis=0)
St_DNN_X_800 = np.nanmean(St_DNN_X_800_B, axis=0)


for k in range(10):
    fig1 = plt.figure() 
    fig1.suptitle("(a) Cox I", fontsize=10) 
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
    ax1_1.plot(t_fig, St_DNN_X_400[k], color='green', label='Proposed', linestyle='-.')
    ax1_1.legend(loc='best', fontsize=6)
    # n=800
    ax1_2.plot(t_fig, St_true_X_fig[:, k], color='black', label='True', linestyle='-')
    ax1_2.plot(t_fig, St_DNN_X_800[k], color='green', label='Proposed', linestyle='-.')
    ax1_2.legend(loc='best', fontsize=6)
    file_name = os.path.join(subfolder, f'DNN_Case1_fig_{k}.jpeg') 
    fig1.savefig(file_name, dpi=300, bbox_inches='tight')