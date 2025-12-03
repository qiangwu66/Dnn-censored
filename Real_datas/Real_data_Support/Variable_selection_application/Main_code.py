# python 3.12.8
#%% ----------------------
import numpy as np
import random
import torch
from Survival_methods.DNN_iteration_g1 import g1_dnn
import pandas as pd
import os
from scipy.stats import norm


#%% ----------------------
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

#%% Data Processing
df = pd.read_csv('support_clean_death_as_event_all_x.csv') # read csv file

X = np.array(df[['male', 'age', 'sps', 'scoma', 'disease1', 'disease2', 'disease3']], dtype='float32')
De = np.array(df['delta'], dtype='float32')
Time = np.array(df['d.time'], dtype='float32')
Time = Time / np.max(Time)

X[:,1] = X[:,1] / np.max(X[:,1])
X[:,2] = X[:,2] / np.max(X[:,2])
X[:,3] = X[:,3] / np.max(X[:,3])

# random order
A = np.arange(len(Time))
np.random.shuffle(A)
X_R = X[A]
De_R = De[A]
Time_R = Time[A]
n = len(Time)
# -------training data1: 40%  validation data1: 20%  training data2: 40%-----------------------
# ---training data1: 3642
# X_R_train1 = X_R[np.arange(3642)]
# De_R_train1 = De_R[np.arange(3642)]
# Time_R_train1 = Time_R[np.arange(3642)]
# # ---training data2: 3642
# X_R_train2 = X_R[np.arange(3642, 7284)]
# De_R_train2 = De_R[np.arange(3642, 7284)]
# Time_R_train2 = Time_R[np.arange(3642, 7284)]
# # ---validation data1: 910
# X_R_valid1 = X_R[np.arange(7284, 8194)]
# De_R_valid1 = De_R[np.arange(7284, 8194)]
# Time_R_valid1 = Time_R[np.arange(7284, 8194)]
# # ---validation data2: 910
# X_R_valid1 = X_R[np.arange(8194, 9104)]
# De_R_valid1 = De_R[np.arange(8194, 9104)]
# Time_R_valid1 = Time_R[np.arange(8194, 9104)]


Data_all_c = {
        'X': np.array(X_R, dtype='float32'),
        'T_O': np.array(Time_R, dtype='float32'),
        'De': np.array(De_R, dtype='float32')
    }


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





# -------------------------------------------
tau = 1

Dnn_layer1 = 2
Dnn_node1 = 30
Dnn_lr1 = 1e-4



Dnn_epoch = 1000
patiences = 10
# ----------------dnn----------------
Est_dnn_g = g1_dnn(polled_data_c, train_data1_c, val_data1_c, tau, Dnn_layer1, Dnn_node1, Dnn_lr1, Dnn_epoch, patiences)
   
g1_T_X_n = Est_dnn_g['g1_T_X_n']
sigma_1_n1 = Est_dnn_g['sigma_1_n1']


Dnn_layer2 = 2
Dnn_lr2 = 1e-4
Dnn_node2_j = [30,30,30,30,30,30,35]

p_values = []
# ----------------dnn (-Xj)----------------
for j in range(X.shape[1]):
    #--------------
    Data_all_ic = Data_all_c.copy()
    if 'X' in Data_all_ic:
        Data_all_ic['X'] = np.delete(Data_all_ic['X'], j, axis=1)

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

    Est_dnn_g_X1 = g1_dnn(polled_data_ic, train_data2_ic, val_data2_ic, tau, Dnn_layer2, Dnn_node2_j[j], Dnn_lr2, Dnn_epoch, patiences)
   
    g1_T_X_n_X1 = Est_dnn_g_X1['g1_T_X_n']
    sigma_1_n1_X1 = Est_dnn_g_X1['sigma_1_n1']

    #%%-----------Test Statistics-------------
    sigma_n1 = np.sqrt(2 * sigma_1_n1 ** 2 + 2 * sigma_1_n1_X1 ** 2)
    
    I_T_T_n = Indicator_matrix(polled_data_c['T_O'], polled_data_c['T_O']) # n * n
    I_T_T_mean = np.mean(I_T_T_n, axis=0) # n
    T_w_n = np.sqrt(0.8 * n) * np.mean(polled_data_c['De'] * I_T_T_mean * (g1_T_X_n - g1_T_X_n_X1))

    T_sigma_n1 = T_w_n / sigma_n1
    print('T_sigma', np.array([T_sigma_n1]))

    p_values.append(2 * (1 - norm.cdf(abs(T_sigma_n1))))




output_folder = "Results_p_values"
os.makedirs(output_folder, exist_ok=True)

result_size_DNN = pd.DataFrame(
    np.array([p_values])
    ) 
size_DNN_path = os.path.join(output_folder, f"DNN_p_values.csv")
result_size_DNN.to_csv(size_DNN_path, index=False, header=False) 
