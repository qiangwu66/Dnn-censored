# python 3.12.8
#%% ----------------------
import numpy as np
import random
import torch
from Survival_methods.DNN_iteration_g1 import g1_dnn
from Survival_methods.AH_iteration import Estimates_AH
import pandas as pd
import os
from scipy.stats import norm


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

# random order
A = np.arange(len(Time))
np.random.shuffle(A)
X_R = X[A]
De_R = De[A]
Time_R = Time[A]

# -------training data1: 40%  validation data1: 20%  training data2: 40%-----------------------
# ---training data1: 3642
X_R_train1 = X_R[np.arange(3642)]
De_R_train1 = De_R[np.arange(3642)]
Time_R_train1 = Time_R[np.arange(3642)]
# ---training data2: 3642
X_R_train2 = X_R[np.arange(3642, 7284)]
De_R_train2 = De_R[np.arange(3642, 7284)]
Time_R_train2 = Time_R[np.arange(3642, 7284)]
# ---validation data1: 1820
X_R_valid1 = X_R[np.arange(7284, 9104)]
De_R_valid1 = De_R[np.arange(7284, 9104)]
Time_R_valid1 = Time_R[np.arange(7284, 9104)]

#----------------------------------------------------------
tau = 1
Dnn_layer1 = 2
Dnn_node1 = 50
Dnn_lr1 = 5e-4
Dnn_epoch = 1000
patiences = 10
m = 3
nodevec= np.array(np.linspace(0, tau, m + 2), dtype="float32")
t_nodes = np.array(np.linspace(0, tau, 101), dtype="float32")

h_beta = np.ones(X.shape[1])

#--------------------------------------------------------
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


train_data1 = {
        'X': X_R_train1,
        'De': De_R_train1,
        'T_O': Time_R_train1
        }

train_data2 = {
        'X': X_R_train2,
        'De': De_R_train2,
        'T_O': Time_R_train2
        }

val_data1 = {
        'X': X_R_valid1,
        'De': De_R_valid1,
        'T_O': Time_R_valid1
        }

polled_data = {
    key: np.concatenate((train_data1[key], train_data2[key]), axis=0) if train_data1[key].ndim > 1 else np.concatenate((train_data1[key], train_data2[key]))
    for key in train_data1
}

# ----------------dnn----------------
Est_dnn_g1 = g1_dnn(polled_data, train_data1,val_data1, tau, Dnn_layer1, Dnn_node1, Dnn_lr1, Dnn_epoch, patiences)

g1_T_X_n = Est_dnn_g1['g1_T_X_n']
sigma_1_n1 = Est_dnn_g1['sigma_1_n1']
# ----------------AH--------------
Est_AH_g0 = Estimates_AH(polled_data,train_data2, t_nodes, m, nodevec, tau, h_beta)

sigma_1_n2_AH = Est_AH_g0['sigma_1_n2_AH']
g0_T_X_n = Est_AH_g0['g0_T_X_n']


#%%-----------Test Statistics-------------
sigma_n1 = np.sqrt(2 * (sigma_1_n1 ** 2 + sigma_1_n2_AH ** 2))
I_T_T_n = Indicator_matrix(polled_data['T_O'], polled_data['T_O'])
I_T_T_mean = np.mean(I_T_T_n, axis=0) 
Weight = (I_T_T_mean + polled_data['X'] @ h_beta) / np.exp(g0_T_X_n)

T_w_n = np.sqrt(7284) * np.mean(polled_data['De'] * I_T_T_mean * (g1_T_X_n - g0_T_X_n))

T_sigma_n1 = T_w_n / sigma_n1


p_value = 2 * (1 - norm.cdf(abs(T_sigma_n1)))


# =================tables=======================
output_folder = "Results_p_values"
os.makedirs(output_folder, exist_ok=True)

result_size_DNN = pd.DataFrame(
    np.array([p_value])
    ) 
size_DNN_path = os.path.join(output_folder, f"AHlinear_p_value.csv")
result_size_DNN.to_csv(size_DNN_path, index=False, header=False) 
