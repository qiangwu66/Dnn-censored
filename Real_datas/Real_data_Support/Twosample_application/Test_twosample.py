# python 3.12.8
#%% ----------------------
import numpy as np
import random
import torch
from Survival_methods.DNN_iteration_g1 import g1_dnn
from Survival_methods.DNN_iteration_g2 import g2_dnn
import pandas as pd
import os
from scipy.stats import norm


#%% ----------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)   

set_seed(1)

def check_matrix_or_vector(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 2 or (value.ndim == 1 and value.size > 1):
            return True
    return False


#%% Data Processing
df = pd.read_csv('support_clean_death_as_event_all_x.csv') # read csv file
# split data
df_female = df[df["male"] == 0]
df_male = df[df["male"] == 1]

df_female = df_female.drop(columns=["male"])
df_male = df_male.drop(columns=["male"])

n1 = len(np.array(df_female['delta'], dtype='float32'))
n2 = len(np.array(df_male['delta'], dtype='float32'))




X_female = np.array(df_female[['age', 'sps', 'scoma', 'disease1', 'disease2', 'disease3']], dtype='float32')
X_female[:,0] = X_female[:,0] / np.max(X_female[:,0])
X_female[:,1] = X_female[:,1] / np.max(X_female[:,1])
X_female[:,2] = X_female[:,2] / np.max(X_female[:,2])
De_female = np.array(df_female['delta'], dtype='float32')
Time_female = np.array(df_female['d.time'], dtype='float32')
Time_female = Time_female / np.max(Time_female)


X_male = np.array(df_male[['age', 'sps', 'scoma', 'disease1', 'disease2', 'disease3']], dtype='float32')
X_male[:,0] = X_male[:,0] / np.max(X_male[:,0])
X_male[:,1] = X_male[:,1] / np.max(X_male[:,1])
X_male[:,2] = X_male[:,2] / np.max(X_male[:,2])
De_male = np.array(df_male['delta'], dtype='float32')
Time_male = np.array(df_male['d.time'], dtype='float32')
Time_male = Time_male / np.max(Time_male)



# ---training data1
X_R_train1 = X_female[:int(0.8 * len(Time_female))]
De_R_train1 = De_female[:int(0.8 * len(Time_female))]
Time_R_train1 = Time_female[:int(0.8 * len(Time_female))]
# ---validation data1
X_R_valid1 = X_female[int(0.8 * len(Time_female)):len(Time_female)]
De_R_valid1 = De_female[int(0.8 * len(Time_female)):len(Time_female)]
Time_R_valid1 = Time_female[int(0.8 * len(Time_female)):len(Time_female)]

# ---training data2
X_R_train2 = X_male[:int(0.8 * len(Time_male))]
De_R_train2 = De_male[:int(0.8 * len(Time_male))]
Time_R_train2 = Time_male[:int(0.8 * len(Time_male))]
# ---validation data2
X_R_valid2 = X_male[int(0.8 * len(Time_male)):len(Time_male)]
De_R_valid2 = De_male[int(0.8 * len(Time_male)):len(Time_male)]
Time_R_valid2 = Time_male[int(0.8 * len(Time_male)):len(Time_male)]


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

val_data2 = {
        'X': X_R_valid2,
        'De': De_R_valid2,
        'T_O': Time_R_valid2
        }

polled_data = {
    key: np.concatenate((train_data1[key], train_data2[key]), axis=0) if train_data1[key].ndim > 1 else np.concatenate((train_data1[key], train_data2[key]))
    for key in train_data1
}


tau = 1
Dnn_layer1 = 2
Dnn_node1 = 50
Dnn_lr1 = 2e-4

Dnn_layer2 = 2
Dnn_node2 = 50
Dnn_lr2 = 2e-4

Dnn_epoch = 1000
patiences = 10

# --------------training g2(t,X)----------------
Est_dnn_g2 = g2_dnn(polled_data, train_data2, val_data2, tau, Dnn_layer2, Dnn_node2, Dnn_lr2, Dnn_epoch, patiences)
g2_T_X_n = Est_dnn_g2['g2_T_X_n']
sigma2_ws = Est_dnn_g2['sigma2_ws']

# ---------------training g1(t,X)----------------------
Est_dnn_g1 = g1_dnn(polled_data, train_data1, val_data1, tau, Dnn_layer1, Dnn_node1, Dnn_lr1, Dnn_epoch, patiences, g2_T_X_n, sigma2_ws)


p_value_w1 = 2 * (1 - norm.cdf(abs(Est_dnn_g1['U_tau_w1'])))
p_value_w2 = 2 * (1 - norm.cdf(abs(Est_dnn_g1['U_tau_w2'])))
p_value_w3 = 2 * (1 - norm.cdf(abs(Est_dnn_g1['U_tau_w3'])))
p_value_w4 = 2 * (1 - norm.cdf(abs(Est_dnn_g1['U_tau_w4'])))


# =================tables=======================
output_folder = "Results_p_values"
os.makedirs(output_folder, exist_ok=True)

result_size_DNN = pd.DataFrame(
    np.array([p_value_w1, p_value_w2, p_value_w3, p_value_w4])
    ) 
size_DNN_path = os.path.join(output_folder, f"Twosample_p_values.csv")
result_size_DNN.to_csv(size_DNN_path, index=False, header=False) 

