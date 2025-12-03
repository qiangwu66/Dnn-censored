# python 3.12.8
#%% ---------------------
import numpy as np
import random
import torch
from data_generator import generate_AFT_1
import matplotlib.pyplot as plt
import os


results_folder = "results_folder"
os.makedirs(results_folder, exist_ok=True)

#%% -----------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed) 

set_seed(1)
n = 2
corr = 0.5
tau = 2


t_fig = np.array(np.linspace(0, tau, 30), dtype="float32")
Data1 = generate_AFT_1(n, corr, tau)
X = Data1['X']
f_X = Data1['f_X']
np.save(os.path.join(results_folder, 't_fig.npy'), t_fig)
np.save(os.path.join(results_folder, 'X_covariates.npy'), X)


for k in range(n):
    S1_t_X = np.exp(- t_fig ** 2 * np.exp(- 2 * f_X[k]))
    S2_t_X = np.exp(- t_fig ** 1.5 * np.exp(- 1.5 * f_X[k]))
    fig1 = plt.figure() 
    fig1.suptitle("AFT", fontsize=10)  
    ax1_1 = fig1.add_subplot(1, 1, 1)
    formatted_row = np.array2string(
        X[k], 
        separator=',', 
        formatter={'float_kind': lambda x: f'{x:.3f}'} 
    )
    ax1_1.set_title(f'$X = {formatted_row}$', fontsize=8, loc='center')
    ax1_1.set_xlabel("t", fontsize=8) 
    ax1_1.set_ylabel('Conditional Survival Function', fontsize=8) 
    ax1_1.tick_params(axis='both', labelsize=6) 
    ax1_1.plot(t_fig, S1_t_X, color='red', label=r'$S_1(t~|~X)$', linestyle='-')
    ax1_1.plot(t_fig, S2_t_X, color='green', label=r'$S_2(t~|~X)$', linestyle='--')
    ax1_1.legend(loc='best', fontsize=6)
    file_name = os.path.join("Individual_DNN", f'AFT_cross_fig_{k}.pdf') 
    fig1.savefig(file_name, dpi=600, bbox_inches='tight')  