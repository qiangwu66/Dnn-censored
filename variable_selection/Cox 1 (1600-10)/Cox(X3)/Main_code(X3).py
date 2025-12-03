# python 3.12.8
#%% ----------------------
import numpy as np
import random
import torch
from data_generator import generate_Cox_1
from Survival_methods.DNN_iteration_g1 import g1_dnn
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
def single_simulation(b, n, Dnn_layer1, Dnn_node1, Dnn_lr1, Dnn_layer2, Dnn_node2, Dnn_lr2, Dnn_epoch, patiences, tau, corr):
    print(f'-------------b={b}--------------')
    set_seed(600 + b)
    # --------------g1(t,x)-------------
    Data_all_c = generate_Cox_1(n, corr, tau)

    Data1_all_c = {
        key: (value[:int(0.5 * n)] if check_matrix_or_vector(value) 
              else value)
        for key, value in Data_all_c.items()
    }
    Data2_all_c = {
        key: (value[-int(0.5 * n):] if check_matrix_or_vector(value) 
              else value)
        for key, value in Data_all_c.items()
    }

    train_data1_c = {
        key: (value[:int(0.8 * n/2)] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data1_all_c.items()
        }
    val_data1_c = {
        key: (value[-int(0.2 * n/2):] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data1_all_c.items()
        }
    

    train_data2_c = {
        key: (value[:int(0.8 * n/2)] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data2_all_c.items()
        }
    # val_data2_c = {
    #     key: (value[-int(0.2 * n):] if check_matrix_or_vector(value) 
    #     else value)
    #     for key, value in Data2_all_c.items()
    #     }
    polled_data_c = {
        key: np.concatenate((train_data1_c[key], train_data2_c[key]), axis=0) if train_data1_c[key].ndim > 1 else np.concatenate((train_data1_c[key], train_data2_c[key]))
        for key in train_data1_c
    }

    #--------------
    Data_all_ic = Data_all_c.copy()
    if 'X' in Data_all_ic:
        Data_all_ic['X'] = np.delete(Data_all_ic['X'], 2, axis=1)

    Data1_all_ic = {
        key: (value[:int(0.5 * n)] if check_matrix_or_vector(value) 
              else value)
        for key, value in Data_all_ic.items()
    }
    Data2_all_ic = {
        key: (value[-int(0.5 * n):] if check_matrix_or_vector(value) 
              else value)
        for key, value in Data_all_ic.items()
    }

    train_data1_ic = {
        key: (value[:int(0.8 * n/2)] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data1_all_ic.items()
        }
    # val_data1_c = {
    #     key: (value[-int(0.2 * n/2):] if check_matrix_or_vector(value) 
    #     else value)
    #     for key, value in Data1_all_ic.items()
    #     }
    
    train_data2_ic = {
        key: (value[:int(0.8 * n/2)] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data2_all_ic.items()
        }
    val_data2_ic = {
        key: (value[-int(0.2 * n/2):] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data2_all_ic.items()
        }
    polled_data_ic = {
        key: np.concatenate((train_data1_ic[key], train_data2_ic[key]), axis=0) if train_data1_ic[key].ndim > 1 else np.concatenate((train_data1_ic[key], train_data2_ic[key]))
        for key in train_data1_ic
    }



    # ----------------dnn----------------
    Est_dnn_g = g1_dnn(polled_data_c, train_data1_c, val_data1_c, tau, Dnn_layer1, Dnn_node1, Dnn_lr1, Dnn_epoch, patiences)
   
    g1_T_X_n = Est_dnn_g['g1_T_X_n']
    sigma_1_n1 = Est_dnn_g['sigma_1_n1']

    # ----------------dnn (-X1)----------------
    Est_dnn_g_X1 = g1_dnn(polled_data_ic, train_data2_ic, val_data2_ic, tau, Dnn_layer2, Dnn_node2, Dnn_lr2, Dnn_epoch, patiences)
   
    g1_T_X_n_X1 = Est_dnn_g_X1['g1_T_X_n']
    sigma_1_n1_X1 = Est_dnn_g_X1['sigma_1_n1']


    #%%-----------Test Statistics-------------
    sigma_n1 = np.sqrt(2 * sigma_1_n1 ** 2 + 2 * sigma_1_n1_X1 ** 2)

    I_T_T_n = Indicator_matrix(polled_data_c['T_O'], polled_data_c['T_O']) # n * n
    I_T_T_mean = np.mean(I_T_T_n, axis=0) # n
    T_w_n = np.sqrt(0.8 * n) * np.mean(polled_data_c['De'] * I_T_T_mean * (g1_T_X_n - g1_T_X_n_X1))
    
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
Dnn_node1 = 40
Dnn_lr1 = 3e-4

Dnn_layer2 = 2
Dnn_node2 = 40
Dnn_lr2 = 3e-4

Dnn_epoch = 1000
patiences = 10



results = Parallel(n_jobs=n_jobs)(
            delayed(single_simulation)(b, n, Dnn_layer1, Dnn_node1, Dnn_lr1, Dnn_layer2, Dnn_node2, Dnn_lr2, Dnn_epoch, patiences, tau, corr) 
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





#%% ---------------------------------------
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
size_DNN_path = os.path.join(output_folder, f"CoxI-size_DNN.csv")
result_size_DNN.to_csv(size_DNN_path, index=False, header=False) 


mask = np.abs(T_sigma_n1_B) <= 1.96

idx0 = np.where(mask)[0]

np.savetxt('file3.txt', idx0, fmt='%d')
