
#%% ----------------------
import numpy as np
import random
import torch
from Survival_methods.DNN_iteration_g1 import g1_dnn
from Survival_methods.DNN_iteration_g2 import g2_dnn
import pandas as pd
import os
from scipy.stats import norm

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

X_variables = X_variables.astype('float32')


df_pre = X_variables[X_variables[:,5] == 0]
df_post = X_variables[X_variables[:,5] == 1]

X_pre = np.delete(df_pre, 5, axis=1)
X_post = np.delete(df_post, 5, axis=1)

Time_pre = Time[X_variables[:,5] == 0]
Time_post = Time[X_variables[:,5] == 1]

De_pre = De[X_variables[:,5] == 0]
De_post = De[X_variables[:,5] == 1]


n1 = len(De_pre)
n2 = len(De_post)


# ---training data1
X_R_train1 = X_pre[:int(0.8 * len(Time_pre))]
De_R_train1 = De_pre[:int(0.8 * len(Time_pre))]
Time_R_train1 = Time_pre[:int(0.8 * len(Time_pre))]
# ---validation data1
X_R_valid1 = X_pre[int(0.8 * len(Time_pre)):len(Time_pre)]
De_R_valid1 = De_pre[int(0.8 * len(Time_pre)):len(Time_pre)]
Time_R_valid1 = Time_pre[int(0.8 * len(Time_pre)):len(Time_pre)]

# ---training data2
X_R_train2 = X_post[:int(0.8 * len(Time_post))]
De_R_train2 = De_post[:int(0.8 * len(Time_post))]
Time_R_train2 = Time_post[:int(0.8 * len(Time_post))]
# ---validation data2
X_R_valid2 = X_post[int(0.8 * len(Time_post)):len(Time_post)]
De_R_valid2 = De_post[int(0.8 * len(Time_post)):len(Time_post)]
Time_R_valid2 = Time_post[int(0.8 * len(Time_post)):len(Time_post)]


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
Dnn_node1 = 30
Dnn_lr1 = 4e-4

Dnn_layer2 = 2
Dnn_node2 = 30
Dnn_lr2 = 4e-4

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

