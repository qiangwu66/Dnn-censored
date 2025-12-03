# python 3.12.8
#%% ----------------------
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os

from Survival_methods.DNN_iteration import g_dnn


results_folder = "results_folder"
os.makedirs(results_folder, exist_ok=True)

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

tau = 1
# random order
# A = np.arange(len(Time))
# np.random.shuffle(A)
# X_R = X[A]
# De_R = De[A]
# Time_R = Time[A]

# -------training data: 80% test data: 20%---------------------------------

# # ---training data 7284
# X_R_train = X_R[np.arange(7284)]
# De_R_train = De_R[np.arange(7284)]
# Time_R_train = Time_R[np.arange(7284)]
# # ---validation data 1820
# X_R_valid = X_R[np.arange(7284, 9104)]
# De_R_valid = De_R[np.arange(7284, 9104)]
# Time_R_valid = Time_R[np.arange(7284, 9104)]



# -------------------Fig-----------------------

X_age_Ave = np.mean(np.array(df['age'], dtype='float32')) / np.max(np.array(df['age'], dtype='float32'))
X_sps_Ave = np.mean(np.array(df['sps'], dtype='float32')) / np.max(np.array(df['sps'], dtype='float32'))
X_scoma_Ave = np.mean(np.array(df['scoma'], dtype='float32')) / np.max(np.array(df['scoma'], dtype='float32'))

X_fig = np.zeros((8,7))
X_fig[4:8,0] = 1
X_fig[:,1] = X_age_Ave 
X_fig[:,2] = X_sps_Ave 
X_fig[:,3] = X_scoma_Ave 
X_fig[0,4] = 1
X_fig[1,5] = 1
X_fig[2,6] = 1
X_fig[4,4] = 1
X_fig[5,5] = 1
X_fig[6,6] = 1


Time_d = np.array(df['d.time'], dtype='float32')
Time_m = Time_d / 30
t_fig_draw = np.array(np.linspace(0, np.max(Time_m), 30), dtype="float32") 

t_fig = t_fig_draw / np.max(Time_m)
np.save(os.path.join(results_folder, 't_fig_draw.npy'), t_fig_draw)

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



# ---------------DNN----------------------------
print('DNN')

Dnn_layer = 3
Dnn_node = 50
Dnn_epoch = 1000
Dnn_lr = 3e-4
patiences = 10
H_t_X_all = np.full((5, len(t_fig), X_fig.shape[0]), np.nan)

for k in range(5):
    # -------training data: 80% test data: 20%---------------------------------
    if k==0:
        # ---training data
        X_R_train = X[np.arange(7284)]
        De_R_train = De[np.arange(7284)]
        Time_R_train = Time[np.arange(7284)]
        # ---validation data
        X_R_valid = X[np.arange(7284, 9104)]
        De_R_valid = De[np.arange(7284, 9104)]
        Time_R_valid = Time[np.arange(7284, 9104)]
    elif k==1:
        # ---training data
        X_R_train = np.vstack((X[:5463,:], X[-1820:,:]))
        De_R_train = np.hstack((De[:5463], De[-1820:]))
        Time_R_train = np.hstack((Time[:5463], Time[-1820:]))
        # ---validation data
        X_R_valid = X[np.arange(5463, 7284)]
        De_R_valid = De[np.arange(5463, 7284)]
        Time_R_valid = Time[np.arange(5463, 7284)]
    elif k==2:
        # ---training data
        X_R_train = np.vstack((X[:3642,:], X[-3641:,:]))
        De_R_train = np.hstack((De[:3642], De[-3641:]))
        Time_R_train = np.hstack((Time[:3642], Time[-3641:]))
        # ---validation data
        X_R_valid = X[np.arange(3642, 5463)]
        De_R_valid = De[np.arange(3642, 5463)]
        Time_R_valid = Time[np.arange(3642, 5463)]
    elif k==3:
        # ---training data
        X_R_train = np.vstack((X[:1821,:], X[-5462:,:]))
        De_R_train = np.hstack((De[:1821], De[-5462:]))
        Time_R_train = np.hstack((Time[:1821], Time[-5462:]))
        # ---validation data
        X_R_valid = X[np.arange(1821, 3642)]
        De_R_valid = De[np.arange(1821, 3642)]
        Time_R_valid = Time[np.arange(1821, 3642)]
    else:
        # ---training data
        X_R_train = X[np.arange(1821, 9104)]
        De_R_train = De[np.arange(1821, 9104)]
        Time_R_train = Time[np.arange(1821, 9104)]
        # ---validation data
        X_R_valid = X[np.arange(1821)]
        De_R_valid = De[np.arange(1821)]
        Time_R_valid = Time[np.arange(1821)]


    Est_dnn = g_dnn(X_R_train, De_R_train, Time_R_train, X_R_valid, De_R_valid, Time_R_valid, X_fig, t_fig, Dnn_layer, Dnn_node, Dnn_lr, Dnn_epoch, patiences)

    H_t_X_all[k,:,:] = Est_dnn['H_T_X_fig'] 


np.save(os.path.join(results_folder, 'H_t_X_all.npy'), H_t_X_all)






#%% Draw Figure

import numpy as np
import matplotlib.pyplot as plt
import os
results_folder = "results_folder"


file_path = os.path.join(results_folder, "H_t_X_all.npy")    
H_t_X_all = np.load(file_path)
H_t_X = np.mean(H_t_X_all, axis=0)

file_path1 = os.path.join(results_folder, "t_fig_draw.npy")  
t_fig_draw = np.load(file_path1) 


fig1 = plt.figure(figsize=(8, 4)) 

ax1_1 = fig1.add_subplot(1, 4, 1)
ax1_1.set_title('ARF/MOSF', fontsize=10, loc='center')  
ax1_1.set_xlabel("t (months)", fontsize=10) 
ax1_1.set_ylabel(r"$H(t|X)$", fontsize=10)  
ax1_1.tick_params(axis='both', labelsize=10)  
ax1_1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5) 


ax1_2 = fig1.add_subplot(1, 4, 2)
ax1_2.set_title('Cancer', fontsize=10, loc='center') 
ax1_2.set_xlabel("t (months)", fontsize=10)  
ax1_2.tick_params(axis='both', labelsize=10)  
ax1_2.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5) 

ax1_3 = fig1.add_subplot(1, 4, 3)
ax1_3.set_title('Coma', fontsize=10, loc='center') 
ax1_3.set_xlabel("t (months)", fontsize=10)  
ax1_3.tick_params(axis='both', labelsize=10)  
ax1_3.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

ax1_4 = fig1.add_subplot(1, 4, 4)
ax1_4.set_title('COPD/CHF/Cirrhosis', fontsize=10, loc='center') 
ax1_4.set_xlabel("t (months)", fontsize=10)  
ax1_4.tick_params(axis='both', labelsize=10)  
ax1_4.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)



ax1_1.plot(t_fig_draw, H_t_X[:, 0], label='Female', color='orange', linestyle='--')
ax1_1.plot(t_fig_draw, H_t_X[:, 4], label='Male', color='blue',  linestyle=':')

ax1_2.plot(t_fig_draw, H_t_X[:, 1], label='Female', color='orange', linestyle='--')
ax1_2.plot(t_fig_draw, H_t_X[:, 5], label='Male', color='blue',linestyle=':')

ax1_3.plot(t_fig_draw, H_t_X[:, 2], label='Female', color='orange', linestyle='--')
ax1_3.plot(t_fig_draw, H_t_X[:, 6], label='Male', color='blue', linestyle=':')

ax1_4.plot(t_fig_draw, H_t_X[:, 3], label='Female', color='orange', linestyle='--')
ax1_4.plot(t_fig_draw, H_t_X[:, 7], label='Male', color='blue', linestyle=':')


# fig1.text(0.45, 1.05, 'Group', fontsize=12, va='center', ha='center')


fig1.legend(
    ['Female', 'Male'],  
    loc='center', bbox_to_anchor=(0.52, 0.92), fontsize=10
)


fig1.tight_layout(rect=[0, 0, 1, 0.88]) 

output_pdf = "Hazard_Functions.pdf"
fig1.savefig(output_pdf, dpi=600)
