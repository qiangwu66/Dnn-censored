# python 3.12.8
#%% ----------- Import Packages-----------
import numpy as np
import torch
import matplotlib.pyplot as plt
from data_generator import generate_case_5
from Survival_methods.DNN_iteration import g_dnn
import pandas as pd
import os
from itertools import product
from scipy.stats import norm

from joblib import Parallel, delayed 

results_folder = "results_folder"
os.makedirs(results_folder, exist_ok=True)

#%% -----------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(1)
#%% ----------------------
tau = 2
p = 3  
corr = 0.5  


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


s = 20
Data_test = generate_case_5(s, corr, tau)
X_test = Data_test['X']  
f_X_test = Data_test['f_X']
T_O_test = Data_test['T_O']
De_test = Data_test['De']

np.savetxt(os.path.join(results_folder, 'X_test.csv'), X_test, delimiter=',')

t_range = np.linspace(0, tau, 11)[1:-1]


S_t_X_true = 1 - norm.cdf((np.repeat(np.log(t_range)[:,np.newaxis],X_test.shape[0],axis=1) - np.repeat(f_X_test[np.newaxis,:],len(t_range),axis=0)) / 1)


np.save(os.path.join(results_folder, 't_range_AFT.npy'), t_range)
np.save(os.path.join(results_folder, 'S_t_X_true_AFT.npy'), S_t_X_true)

#%% ========================
def single_simulation(b, n, Dnn_layer, Dnn_node, Dnn_lr, Dnn_epoch, t_range, Data_test, tau, patiences, num_bs):
    print(f'-------------n={n}, b={b}--------------')
    set_seed(20 + b)
    # ---------------------------
    Data_all = generate_case_5(n, corr, tau) 
    
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
    

    Est_dnn = g_dnn(train_data, val_data, Data_test, t_range, Dnn_layer, Dnn_node, Dnn_lr, Dnn_epoch, patiences)

    
    S_t_X_bootstrape = np.full((len(t_range), s, num_bs), np.nan)
    for k in range(num_bs):
        bootstrap_index_k = np.random.choice(range(train_data['X'].shape[0]), size=train_data['X'].shape[0], replace=True)

        train_data_bs_k = {
            'X': np.array(train_data['X'][bootstrap_index_k, :], dtype='float32'),
            'T_O': np.array(train_data['T_O'][bootstrap_index_k], dtype='float32'),
            'De': np.array(train_data['De'][bootstrap_index_k], dtype='float32'),
        }
        Est_dnn_bs_k = g_dnn(train_data_bs_k, val_data, Data_test, t_range, Dnn_layer, Dnn_node, Dnn_lr, Dnn_epoch, patiences)
        S_t_X_bootstrape[:,:,k] = Est_dnn_bs_k['S_T_X_values']

    CI_values = np.full((len(t_range), s), np.nan)
    for i, j in product(range(len(t_range)), range(s)):
        S_sort_k = np.sort(S_t_X_bootstrape[i,j,:]) 
        S_p025 = np.quantile(S_sort_k, 0.025)
        S_p975 = np.quantile(S_sort_k, 0.975)
        if S_p025 <= S_t_X_true[i,j] <= S_p975:
            CI_values[i,j] = 1
        else:
            CI_values[i,j] = 0
    
    CI_values_se = np.full((len(t_range), s), np.nan)
    S_t_X_bar_b = np.mean(S_t_X_bootstrape, axis=-1)
    Se_b = np.std(S_t_X_bootstrape, axis=-1)
    S_lower_b = S_t_X_bar_b - 1.96 * Se_b / np.sqrt(num_bs)
    S_upper_b = S_t_X_bar_b + 1.96 * Se_b / np.sqrt(num_bs)
    for i, j in product(range(len(t_range)), range(s)):
        if S_lower_b[i,j] <= S_t_X_true[i,j] <= S_upper_b[i,j]:
            CI_values_se[i,j] = 1
        else:
            CI_values_se[i,j] = 0

    return {
        "S_t_X_hat": Est_dnn['S_T_X_values'], 
        "CI_values": CI_values, 
        "CI_values_se": CI_values_se, 
        "Se_b": Se_b
    }

#%% ======================
results = [] 
n_jobs = 12
B = 500 

num_bs = 100


n = 400
Dnn_layer = 2
Dnn_node = 35
Dnn_epoch = 1000
Dnn_lr = 2e-4
patiences = 10


results = Parallel(n_jobs=n_jobs)(
            delayed(single_simulation)(b, n, Dnn_layer, Dnn_node, Dnn_lr, Dnn_epoch, t_range, Data_test, tau, patiences, num_bs) 
            for b in range(B)
        )

#%% ====================
def process_results(results):
    S_t_X_hat = []
    CI_values = []
    CI_values_se = []
    Se_b = []

    for res in results:
        S_t_X_hat.append(res["S_t_X_hat"])
        CI_values.append(res["CI_values"])
        CI_values_se.append(res["CI_values_se"])
        Se_b.append(res["Se_b"])

    S_t_X_hat = np.array(S_t_X_hat)
    CI_values = np.array(CI_values)
    CI_values_se = np.array(CI_values_se)
    Se_b = np.array(Se_b)

    np.save(os.path.join(results_folder, 'S_t_X_hat.npy'), S_t_X_hat)
    np.save(os.path.join(results_folder, 'CI_values.npy'), CI_values)
    np.save(os.path.join(results_folder, 'CI_values_se.npy'), CI_values_se)
    np.save(os.path.join(results_folder, 'Se_b.npy'), Se_b)


process_results(results)



#%% 运行结果
import numpy as np
import os
import pandas as pd

n = 400
Dnn_layer = 2
Dnn_node = 35
Dnn_epoch = 1000
Dnn_lr = 2e-4
patiences = 10


results_folder = "results_folder"


file_names = [
     'S_t_X_hat.npy',
     'CI_values.npy',
     'CI_values_se.npy',
     'S_t_X_true_AFT.npy',
     't_range_AFT.npy',
     'Se_b.npy'
]


data = {}
for file_name in file_names:
    file_path = os.path.join(results_folder, file_name) 
    data[file_name] = np.load(file_path) 


S_t_X_hat_B = data['S_t_X_hat.npy'] 
CI_values_B = data['CI_values.npy'] 
CI_values_se_B = data['CI_values_se.npy'] 
S_t_X_true = data['S_t_X_true_AFT.npy'] 
Se_b = data['Se_b.npy'] 
t_range = data['t_range_AFT.npy'] 

Bias = np.mean(S_t_X_hat_B, axis = 0) - S_t_X_true 
SD = np.std(S_t_X_hat_B, axis = 0) 
CP = np.mean(CI_values_B, axis = 0)

CP_se = np.mean(CI_values_se_B, axis = 0)
Se_B = np.mean(Se_b, axis = 0)

# =================tables=======================
output_folder = "Individual_AFT"
os.makedirs(output_folder, exist_ok=True)


result_t_range = pd.DataFrame(t_range)

t_range_path = os.path.join(output_folder, "t_range_AFT.csv")
result_t_range.to_csv(t_range_path, index=False, header=False) 



result_S_t_X_true = pd.DataFrame(S_t_X_true)

S_t_X_true_path = os.path.join(output_folder, "S_t_X_true_AFT.csv")
result_S_t_X_true.to_csv(S_t_X_true_path, index=False, header=False)  



result_Bias = pd.DataFrame(Bias)

Bias_path = os.path.join(output_folder, f"Bias_AFT-{Dnn_layer}-{Dnn_node}-{Dnn_lr}.csv")
result_Bias.to_csv(Bias_path, index=False, header=False)  




result_SD = pd.DataFrame(SD)  

SD_path = os.path.join(output_folder, f"SD_AFT-{Dnn_layer}-{Dnn_node}-{Dnn_lr}.csv")
result_SD.to_csv(SD_path, index=False, header=False) 



result_CP = pd.DataFrame(CP) 

CP_path = os.path.join(output_folder, f"CP_AFT-{Dnn_layer}-{Dnn_node}-{Dnn_lr}.csv")
result_CP.to_csv(CP_path, index=False, header=False)



result_CP_se = pd.DataFrame(CP_se) 

CP_se_path = os.path.join(output_folder, f"CP_se_AFT-{Dnn_layer}-{Dnn_node}-{Dnn_lr}.csv")
result_CP_se.to_csv(CP_se_path, index=False, header=False)


result_se = pd.DataFrame(Se_B)  

Se_B_path = os.path.join(output_folder, f"Se_B_AFT-{Dnn_layer}-{Dnn_node}-{Dnn_lr}.csv")
result_se.to_csv(Se_B_path, index=False, header=False) 
