# python 3.12.8
#%% -----------------------------------------
import numpy as np
import random
import torch
from data_generator import generate_Cox_1, generate_Cox_2_nocross
from Survival_methods.DNN_iteration_g1 import g1_dnn
from Survival_methods.DNN_iteration_g2 import g2_dnn
import pandas as pd
import os
from joblib import Parallel, delayed


results_folder = "results_folder"
os.makedirs(results_folder, exist_ok=True)

#%% ---------------------------------------
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


#%% ==============================================
def single_simulation(b, n, Dnn_layer1, Dnn_node1, Dnn_lr1, Dnn_layer2, Dnn_node2, Dnn_lr2, Dnn_epoch, patiences, tau, corr, c1):
    print(f'-------------n={n}, b={b}--------------')
    set_seed(500 + b)
    # --------------g1(t,x)-------------
    Data1_all = generate_Cox_1(n, corr, tau) 
    
    train_data1 = {
        key: (value[:int(0.8 * n)] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data1_all.items()
        }
    val_data1 = {
        key: (value[-int(0.2 * n):] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data1_all.items()
        }
    

    # --------------g2(t,x)-------------
    Data2_all = generate_Cox_2_nocross(n, corr, tau, c1) 
    
    train_data2 = {
        key: (value[:int(0.8 * n)] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data2_all.items()
        }
    val_data2 = {
        key: (value[-int(0.2 * n):] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data2_all.items()
        }

    polled_data = {
        key: np.concatenate((train_data1[key], train_data2[key]), axis=0) if train_data1[key].ndim > 1 else np.concatenate((train_data1[key], train_data2[key]))
        for key in train_data1
    }

    # ----------------training g2(t,X)----------------
    Est_dnn_g2 = g2_dnn(polled_data, train_data2, val_data2, tau, Dnn_layer2, Dnn_node2, Dnn_lr2, Dnn_epoch, patiences)
    g2_T_X_n = Est_dnn_g2['g2_T_X_n']
    sigma2_ws = Est_dnn_g2['sigma2_ws']

    # ----------------training g1(t,X)----------------------
    Est_dnn_g1 = g1_dnn(polled_data, train_data1, val_data1, tau, Dnn_layer1, Dnn_node1, Dnn_lr1, Dnn_epoch, patiences, g2_T_X_n, sigma2_ws)

    return {
        'U_tau_w1': Est_dnn_g1['U_tau_w1'],
        'U_tau_w2': Est_dnn_g1['U_tau_w2'],
        'U_tau_w3': Est_dnn_g1['U_tau_w3'],
        'U_tau_w4': Est_dnn_g1['U_tau_w4'],
    }

#%% ========================================
results = []
n_jobs = 1
B = 500


#%% ----------------------------------------
tau = 2 
corr = 0.5 
n = 800
Dnn_layer1 = 2
Dnn_node1 = 30
Dnn_lr1 = 2e-4

Dnn_layer2 = 2
Dnn_node2 = 30
Dnn_lr2 = 2e-4

Dnn_epoch = 1000
patiences = 10
c1 = 0 # c1 = 0, 0.125, 0.25, 0.5

results = Parallel(n_jobs=n_jobs)(
            delayed(single_simulation)(b, n, Dnn_layer1, Dnn_node1, Dnn_lr1, Dnn_layer2, Dnn_node2, Dnn_lr2, Dnn_epoch, patiences, tau, corr, c1) 
            for b in range(B)
        )

#%% =============================================
def process_results(results):
    U_tau_w1 = []
    U_tau_w2 = []
    U_tau_w3 = []
    U_tau_w4 = []
    

    for res in results:
        U_tau_w1.append(res["U_tau_w1"])
        U_tau_w2.append(res["U_tau_w2"])
        U_tau_w3.append(res["U_tau_w3"])
        U_tau_w4.append(res["U_tau_w4"])


    U_tau_w1 = np.array(U_tau_w1)
    U_tau_w2 = np.array(U_tau_w2)
    U_tau_w3 = np.array(U_tau_w3)
    U_tau_w4 = np.array(U_tau_w4)

    np.save(os.path.join(results_folder, 'U_tau_w1.npy'), U_tau_w1)
    np.save(os.path.join(results_folder, 'U_tau_w2.npy'), U_tau_w2)
    np.save(os.path.join(results_folder, 'U_tau_w3.npy'), U_tau_w3)
    np.save(os.path.join(results_folder, 'U_tau_w4.npy'), U_tau_w4)


process_results(results)



import numpy as np
import os
import pandas as pd


results_folder = "results_folder"

file_names = [
     'U_tau_w1.npy',
     'U_tau_w2.npy',
     'U_tau_w3.npy',
     'U_tau_w4.npy',
]

data = {}
for file_name in file_names:
    file_path = os.path.join(results_folder, file_name) 
    data[file_name] = np.load(file_path)


U_tau_w1_B = data['U_tau_w1.npy']
U_tau_w2_B = data['U_tau_w2.npy']
U_tau_w3_B = data['U_tau_w3.npy']
U_tau_w4_B = data['U_tau_w4.npy']


size1_Cox = np.mean(abs(U_tau_w1_B) > 1.96)
size2_Cox = np.mean(abs(U_tau_w2_B) > 1.96)
size3_Cox = np.mean(abs(U_tau_w3_B) > 1.96)
size4_Cox = np.mean(abs(U_tau_w4_B) > 1.96)


# =================tables=======================
output_folder = "Individual_DNN"
os.makedirs(output_folder, exist_ok=True)

result_size_DNN = pd.DataFrame(
    np.array([size1_Cox, size2_Cox, size3_Cox, size4_Cox])
    )
size_DNN_path = os.path.join(output_folder, f"size_DNN-{Dnn_layer1}-{Dnn_node1}-{Dnn_lr1}-{Dnn_layer2}-{Dnn_node2}-{Dnn_lr2}-c1-{c1}.csv")
result_size_DNN.to_csv(size_DNN_path, index=False, header=False)
