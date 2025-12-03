# python 3.12.8
#%% ----------------------
import numpy as np
import random
import torch
from data_generator import generate_AFT_1
from Survival_methods.DNN_iteration_g1 import g1_dnn
from Survival_methods.CoxPH_iteration import Estimates_Coxph
import pandas as pd
import os
from joblib import Parallel, delayed


results_folder = "results_folder"
os.makedirs(results_folder, exist_ok=True)

#%% -----------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(1)



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


T_sigma_n100 = []

#%% ========================
def single_simulation(b, n, Dnn_layer1, Dnn_node1, Dnn_lr1, Dnn_epoch, patiences, tau, corr, t_nodes, m, nodevec):
    print(f'-------------b={b}--------------')
    set_seed(500 + b)
    # --------------g1(t,x)-------------
    Data_all = generate_AFT_1(n, corr, tau)

    Data1_all = {
        key: (value[:int(0.5 * n)] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data_all.items()
        }
    Data2_all = {
        key: (value[-int(0.5 * n):] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data_all.items()
        }
    

    train_data1 = {
        key: (value[:int(0.8 * n/2)] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data1_all.items()
        }
    val_data1 = {
        key: (value[-int(0.2 * n/2):] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data1_all.items()
        }
    
    train_data2 = Data2_all
    # train_data2 = {
    #     key: (value[:int(0.8 * n)] if check_matrix_or_vector(value) 
    #     else value)
    #     for key, value in Data2_all.items()
    #     }
    # val_data2 = {
    #     key: (value[-int(0.2 * n):] if check_matrix_or_vector(value) 
    #     else value)
    #     for key, value in Data2_all.items()
    #     }
    


    polled_data = {
        key: np.concatenate((train_data1[key], train_data2[key]), axis=0) if train_data1[key].ndim > 1 else np.concatenate((train_data1[key], train_data2[key]))
        for key in train_data1
    }

    # ----------------dnn----------------
    Est_dnn_g1 = g1_dnn(polled_data, train_data1, val_data1, tau, Dnn_layer1, Dnn_node1, Dnn_lr1, Dnn_epoch, patiences)
   
    g1_T_X_n = Est_dnn_g1['g1_T_X_n']
    sigma_1_n1 = Est_dnn_g1['sigma_1_n1']


    # ----------------CoxPH--------------
    Est_CoxPH_g0 = Estimates_Coxph(polled_data, train_data2, t_nodes, m, nodevec, tau)

    sigma_1_n2_coxph = Est_CoxPH_g0['sigma_1_n2_coxph']
    g0_T_X_n = Est_CoxPH_g0['g0_T_X_n']


    #%%-----------Test Statistics-------------
    sigma_n1 = np.sqrt((1440 / 640) * sigma_1_n1 ** 2 + (1440 / 800) * sigma_1_n2_coxph ** 2)

    I_T_T_n = Indicator_matrix(polled_data['T_O'], polled_data['T_O']) # n * n
    I_T_T_mean = np.mean(I_T_T_n, axis=0) # n
    T_w_n = np.sqrt(0.8 * n) * np.mean(polled_data['De'] * I_T_T_mean * (g1_T_X_n - g0_T_X_n))

    T_sigma_n1 = T_w_n / sigma_n1
    print('T_sigma', np.array([T_sigma_n1]))

    T_sigma_n100.append((abs(T_sigma_n1) > 1.96))
    print(np.mean(np.array([T_sigma_n100])))


    return {
        'T_sigma_n1': T_sigma_n1,
    }

#%% ======================
results = [] 
n_jobs = 1 
B = 500


#%% -----------------------
tau = 2 
corr = 0.5 
n = 1600
Dnn_layer1 = 2
Dnn_node1 = 60
Dnn_lr1 = 4e-4


Dnn_epoch = 1000
patiences = 10

m = 3
nodevec= np.array(np.linspace(0, tau, m + 2), dtype="float32")


t_nodes = np.array(np.linspace(0, tau, 101), dtype="float32")




results = Parallel(n_jobs=n_jobs)(
            delayed(single_simulation)(b, n, Dnn_layer1, Dnn_node1, Dnn_lr1, Dnn_epoch, patiences, tau, corr, t_nodes, m, nodevec) 
            for b in range(B)
        )

#%% ======================
def process_results(results):
    T_sigma_n1 = []

    for res in results:
        T_sigma_n1.append(res["T_sigma_n1"])
       

    T_sigma_n1 = np.array(T_sigma_n1)

    np.save(os.path.join(results_folder, 'T_sigma_n1.npy'), T_sigma_n1)


process_results(results)


#%% -------------------------------
import numpy as np
import os
import pandas as pd


results_folder = "results_folder"

file_names = [
     'T_sigma_n1.npy',
]

data = {}
for file_name in file_names:
    file_path = os.path.join(results_folder, file_name) 
    data[file_name] = np.load(file_path) 


T_sigma_n1_B = data['T_sigma_n1.npy'] # B


size = np.mean(abs(T_sigma_n1_B) > 1.96)



# =================tables=======================
output_folder = "Individual_DNN"
os.makedirs(output_folder, exist_ok=True)

result_size_DNN = pd.DataFrame(
    np.array([size])
    ) 
size_DNN_path = os.path.join(output_folder, f"AFTI-size_DNN.csv")
result_size_DNN.to_csv(size_DNN_path, index=False, header=False) 
