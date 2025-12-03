# python 3.12.8
#%% ----------------------
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from data_generator import generate_case_6
from Survival_methods.DNN_iteration import g_dnn
import pandas as pd
import os
from scipy.stats import norm
from joblib import Parallel, delayed  


results_folder = "results_folder"
os.makedirs(results_folder, exist_ok=True)

#%% -----------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)  

set_seed(1)
#%% -----------------------
tau = 2 
p = 3  
corr = 0.5 


def check_matrix_or_vector(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 2 or (value.ndim == 1 and value.size > 1):
            return True
    return False


#%% ========================
def single_simulation(b, n, Dnn_layer, Dnn_node, Dnn_lr, Dnn_epoch, tau, patiences):
    print(f'-------------n={n}, b={b}--------------')
    set_seed(20 + b)
    # --------------数据生成-------------
    Data_all = generate_case_6(n, corr, tau) 
    
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

    T_O_train = train_data['T_O']
    f_X_train = train_data['f_X']
    g_t_X_true_Cox = np.log(np.sqrt(T_O_train + 0.001) / 6) + f_X_train 
    g_t_X_true_AFT = np.log(2 * T_O_train * np.exp(-2 * f_X_train))
    g_t_X_true_AH = np.log(np.maximum((T_O_train + 0.001) / 10 + f_X_train, 1e-8))

    Est_dnn = g_dnn(train_data, val_data, tau, Dnn_layer, Dnn_node, Dnn_lr, Dnn_epoch, patiences, g_t_X_true_Cox, g_t_X_true_AFT, g_t_X_true_AH)



    return {
        'T_Sigma1_Cox': Est_dnn['T_Sigma1_Cox'],
        'T_Sigma2_Cox': Est_dnn['T_Sigma2_Cox'],
        'T_Sigma3_Cox': Est_dnn['T_Sigma3_Cox'],
        'T_Sigma4_Cox': Est_dnn['T_Sigma4_Cox'],
        'T_Sigma1_AFT': Est_dnn['T_Sigma1_AFT'],
        'T_Sigma2_AFT': Est_dnn['T_Sigma2_AFT'],
        'T_Sigma3_AFT': Est_dnn['T_Sigma3_AFT'],
        'T_Sigma4_AFT': Est_dnn['T_Sigma4_AFT'],
        'T_Sigma1_AH': Est_dnn['T_Sigma1_AH'],
        'T_Sigma2_AH': Est_dnn['T_Sigma2_AH'],
        'T_Sigma3_AH': Est_dnn['T_Sigma3_AH'],
        'T_Sigma4_AH': Est_dnn['T_Sigma4_AH'],
    }

#%% ======================
results = []  
n_jobs = 1 
B = 500 

n = 800
Dnn_layer = 1
Dnn_node = 40
Dnn_epoch = 1000
Dnn_lr = 4e-4
patiences = 10

results = Parallel(n_jobs=n_jobs)(
            delayed(single_simulation)(b, n, Dnn_layer, Dnn_node, Dnn_lr, Dnn_epoch, tau, patiences) 
            for b in range(B)
        )

#%% ======================
def process_results(results):
    T_Sigma_1_Cox = []
    T_Sigma_2_Cox = []
    T_Sigma_3_Cox = []
    T_Sigma_4_Cox = []
    T_Sigma_1_AFT = []
    T_Sigma_2_AFT = []
    T_Sigma_3_AFT = []
    T_Sigma_4_AFT = []
    T_Sigma_1_AH = []
    T_Sigma_2_AH = []
    T_Sigma_3_AH = []
    T_Sigma_4_AH = []
    

    for res in results:
        T_Sigma_1_Cox.append(res["T_Sigma1_Cox"])
        T_Sigma_2_Cox.append(res["T_Sigma2_Cox"])
        T_Sigma_3_Cox.append(res["T_Sigma3_Cox"])
        T_Sigma_4_Cox.append(res["T_Sigma4_Cox"])
        T_Sigma_1_AFT.append(res["T_Sigma1_AFT"])
        T_Sigma_2_AFT.append(res["T_Sigma2_AFT"])
        T_Sigma_3_AFT.append(res["T_Sigma3_AFT"])
        T_Sigma_4_AFT.append(res["T_Sigma4_AFT"])
        T_Sigma_1_AH.append(res["T_Sigma1_AH"])
        T_Sigma_2_AH.append(res["T_Sigma2_AH"])
        T_Sigma_3_AH.append(res["T_Sigma3_AH"])
        T_Sigma_4_AH.append(res["T_Sigma4_AH"])
        


    T_Sigma_1_Cox = np.array(T_Sigma_1_Cox)
    T_Sigma_2_Cox = np.array(T_Sigma_2_Cox)
    T_Sigma_3_Cox = np.array(T_Sigma_3_Cox)
    T_Sigma_4_Cox = np.array(T_Sigma_4_Cox)
    T_Sigma_1_AFT = np.array(T_Sigma_1_AFT)
    T_Sigma_2_AFT = np.array(T_Sigma_2_AFT)
    T_Sigma_3_AFT = np.array(T_Sigma_3_AFT)
    T_Sigma_4_AFT = np.array(T_Sigma_4_AFT)
    T_Sigma_1_AH = np.array(T_Sigma_1_AH)
    T_Sigma_2_AH = np.array(T_Sigma_2_AH)
    T_Sigma_3_AH = np.array(T_Sigma_3_AH)
    T_Sigma_4_AH = np.array(T_Sigma_4_AH)



    np.save(os.path.join(results_folder, 'T_Sigma_1_Cox.npy'), T_Sigma_1_Cox)
    np.save(os.path.join(results_folder, 'T_Sigma_2_Cox.npy'), T_Sigma_2_Cox)
    np.save(os.path.join(results_folder, 'T_Sigma_3_Cox.npy'), T_Sigma_3_Cox)
    np.save(os.path.join(results_folder, 'T_Sigma_4_Cox.npy'), T_Sigma_4_Cox)
    np.save(os.path.join(results_folder, 'T_Sigma_1_AFT.npy'), T_Sigma_1_AFT)
    np.save(os.path.join(results_folder, 'T_Sigma_2_AFT.npy'), T_Sigma_2_AFT)
    np.save(os.path.join(results_folder, 'T_Sigma_3_AFT.npy'), T_Sigma_3_AFT)
    np.save(os.path.join(results_folder, 'T_Sigma_4_AFT.npy'), T_Sigma_4_AFT)
    np.save(os.path.join(results_folder, 'T_Sigma_1_AH.npy'), T_Sigma_1_AH)
    np.save(os.path.join(results_folder, 'T_Sigma_2_AH.npy'), T_Sigma_2_AH)
    np.save(os.path.join(results_folder, 'T_Sigma_3_AH.npy'), T_Sigma_3_AH)
    np.save(os.path.join(results_folder, 'T_Sigma_4_AH.npy'), T_Sigma_4_AH)


process_results(results)








import numpy as np
import os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

n = 800
Dnn_layer = 1
Dnn_node = 40
Dnn_epoch = 1000
Dnn_lr = 4e-4
patiences = 10


results_folder = "results_folder"

file_names = [
     'T_Sigma_1_Cox.npy',
     'T_Sigma_2_Cox.npy',
     'T_Sigma_3_Cox.npy',
     'T_Sigma_4_Cox.npy',
     'T_Sigma_1_AFT.npy',
     'T_Sigma_2_AFT.npy',
     'T_Sigma_3_AFT.npy',
     'T_Sigma_4_AFT.npy',
     'T_Sigma_1_AH.npy',
     'T_Sigma_2_AH.npy',
     'T_Sigma_3_AH.npy',
     'T_Sigma_4_AH.npy',
]


data = {}
for file_name in file_names:
    file_path = os.path.join(results_folder, file_name)  
    data[file_name] = np.load(file_path)  


T_Sigma_1_Cox_B = data['T_Sigma_1_Cox.npy'] 
T_Sigma_2_Cox_B = data['T_Sigma_2_Cox.npy'] 
T_Sigma_3_Cox_B = data['T_Sigma_3_Cox.npy'] 
T_Sigma_4_Cox_B = data['T_Sigma_4_Cox.npy'] 
T_Sigma_1_AFT_B = data['T_Sigma_1_AFT.npy'] 
T_Sigma_2_AFT_B = data['T_Sigma_2_AFT.npy'] 
T_Sigma_3_AFT_B = data['T_Sigma_3_AFT.npy'] 
T_Sigma_4_AFT_B = data['T_Sigma_4_AFT.npy'] 
T_Sigma_1_AH_B = data['T_Sigma_1_AH.npy'] 
T_Sigma_2_AH_B = data['T_Sigma_2_AH.npy'] 
T_Sigma_3_AH_B = data['T_Sigma_3_AH.npy'] 
T_Sigma_4_AH_B = data['T_Sigma_4_AH.npy'] 



size1_Cox = np.mean(abs(T_Sigma_1_Cox_B) > 1.96)
size2_Cox = np.mean(abs(T_Sigma_2_Cox_B) > 1.96)
size3_Cox = np.mean(abs(T_Sigma_3_Cox_B) > 1.96)
size4_Cox = np.mean(abs(T_Sigma_4_Cox_B) > 1.96)

size1_AFT = np.mean(abs(T_Sigma_1_AFT_B) > 1.96)
size2_AFT = np.mean(abs(T_Sigma_2_AFT_B) > 1.96)
size3_AFT = np.mean(abs(T_Sigma_3_AFT_B) > 1.96)
size4_AFT = np.mean(abs(T_Sigma_4_AFT_B) > 1.96)

size1_AH = np.mean(abs(T_Sigma_1_AH_B) > 1.96)
size2_AH = np.mean(abs(T_Sigma_2_AH_B) > 1.96)
size3_AH = np.mean(abs(T_Sigma_3_AH_B) > 1.96)
size4_AH = np.mean(abs(T_Sigma_4_AH_B) > 1.96)



# =================tables=======================
output_folder = "Individual_DNN"
os.makedirs(output_folder, exist_ok=True)


result_size_DNN = pd.DataFrame(
    np.array([
        [size1_Cox, size2_Cox, size3_Cox, size4_Cox], 
        [size1_AFT, size2_AFT, size3_AFT, size4_AFT], 
        [size1_AH, size2_AH, size3_AH, size4_AH]
        ])
    ) 
size_DNN_path = os.path.join(output_folder, f"size_DNN-{n}-{Dnn_layer}-{Dnn_node}-{Dnn_lr}.csv")
result_size_DNN.to_csv(size_DNN_path, index=False, header=False)  








#%%--------------------------------------------------------------------------------------------------

fig1 = plt.figure(figsize=(8, 2))  
fig1.suptitle(r"QQ-Plots under AH model", fontsize=10)  


ax1_1 = fig1.add_subplot(1, 4, 1, aspect='equal')  
ax1_2 = fig1.add_subplot(1, 4, 2, aspect='equal')
ax1_3 = fig1.add_subplot(1, 4, 3, aspect='equal')
ax1_4 = fig1.add_subplot(1, 4, 4, aspect='equal')

res1 = stats.probplot(T_Sigma_1_AH_B, dist="norm", plot=ax1_1)
res2 = stats.probplot(T_Sigma_2_AH_B, dist="norm", plot=ax1_2)
res3 = stats.probplot(T_Sigma_3_AH_B, dist="norm", plot=ax1_3)
res4 = stats.probplot(T_Sigma_4_AH_B, dist="norm", plot=ax1_4)


x_min = -3
x_max = 3
y_min = -3
y_max = 3


for ax in [ax1_1, ax1_2, ax1_3, ax1_4]:
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


ax1_1.set_xlabel("Theoretical Quantiles", fontsize=6)
ax1_1.set_title(r'$(\rho,\gamma)=(0,0)$', fontsize=6, loc='center')
ax1_1.set_ylabel('Sample Quantiles', fontsize=6)
ax1_1.tick_params(axis='both', labelsize=6)

ax1_2.set_xlabel("Theoretical Quantiles", fontsize=6)
ax1_2.set_title(r'$(\rho,\gamma)=(1,0)$', fontsize=6, loc='center')
ax1_2.set_ylabel('Sample Quantiles', fontsize=6)
ax1_2.tick_params(axis='both', labelsize=6)

ax1_3.set_xlabel("Theoretical Quantiles", fontsize=6)
ax1_3.set_title(r'$(\rho,\gamma)=(0.5,0.5)$', fontsize=6, loc='center')
ax1_3.set_ylabel('Sample Quantiles', fontsize=6)
ax1_3.tick_params(axis='both', labelsize=6)

ax1_4.set_xlabel("Theoretical Quantiles", fontsize=6)
ax1_4.set_title(r'$(\rho,\gamma)=(1,1)$', fontsize=6, loc='center')
ax1_4.set_ylabel('Sample Quantiles', fontsize=6)
ax1_4.tick_params(axis='both', labelsize=6)


plt.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.2, wspace=0.1)  


fig1.savefig("qq_plot_AH.pdf", dpi=600)
