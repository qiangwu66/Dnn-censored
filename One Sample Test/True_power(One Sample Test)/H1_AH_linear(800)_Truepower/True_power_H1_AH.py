# python 3.12.8
#%% ----------------------
import numpy as np
import numpy.random as ndm
import torch
from data_generator import generate_case_6
import pandas as pd
from scipy.stats import norm



#%% ----------------------
def set_seed(seed):
    np.random.seed(seed)  
    torch.manual_seed(seed)  

set_seed(1)
#%% -----------------------
tau = 2 
p = 3  
corr = 0.5 


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

def uniform_data(n, u1, u2):
    a = ndm.rand(n)
    b = (u2 - u1) * a + u1
    return b

#%% ------------------------------------------------------
Data_N = generate_case_6(2000, corr, tau)

De_N = Data_N['De']
f_X_N = Data_N['f_X']
Y_N = Data_N['T_O']
Y_N = np.maximum(Y_N, 1e-4)


g_t_X_true_Cox = np.log(np.sqrt(Y_N + 0.001) / 6) + f_X_N
St_true_X_Cox = np.exp(-((Y_N + 0.001) ** (3 / 2) / 9 - 0.001 ** (3 / 2) / 9) * np.exp(f_X_N))

g_t_X_true_AFT = np.log(2 * Y_N * np.exp(-2 * f_X_N))
St_true_X_AFT = np.exp(-(Y_N * np.exp(-f_X_N)) ** 2)

g_t_X_true_AH = np.log(np.maximum((Y_N + 0.001 / 10) + f_X_N, 1e-8)) 
St_true_X_AH = np.exp(- ((Y_N + 0.001) ** 2 - 0.001 ** 2) / 20 - Y_N * f_X_N)



Data_1 = generate_case_6(5000, corr, tau)
X_gene = Data_1['X'] 
f_X_gene = Data_1['f_X']

C_gene = np.minimum(ndm.exponential(2, 5000), tau)

St_X_gene_AH = np.exp(- ((Y_N[:,np.newaxis] + 0.001) ** 2 - 0.001 ** 2) / 20 - Y_N[:,np.newaxis] * f_X_gene)

E_S_N = np.mean(St_X_gene_AH, axis=1) 
E_I_C_T =  np.mean(Indicator_matrix(C_gene, Y_N), axis=0) 


W_1 = E_S_N * E_I_C_T 

W_2 = E_S_N * E_I_C_T * St_true_X_AH 

W_3 = E_S_N * E_I_C_T * np.sqrt(St_true_X_AH * (1-St_true_X_AH)) 

W_4 = E_S_N * E_I_C_T * St_true_X_AH * (1-St_true_X_AH) 


mu_H0_Cox_H1_AH_1 = np.mean(De_N * W_1 * (g_t_X_true_AH - g_t_X_true_Cox))
mu_H0_Cox_H1_AH_2 = np.mean(De_N * W_2 * (g_t_X_true_AH - g_t_X_true_Cox))
mu_H0_Cox_H1_AH_3 = np.mean(De_N * W_3 * (g_t_X_true_AH - g_t_X_true_Cox))
mu_H0_Cox_H1_AH_4 = np.mean(De_N * W_4 * (g_t_X_true_AH - g_t_X_true_Cox))

mu_H0_AFT_H1_AH_1 = np.mean(De_N * W_1 * (g_t_X_true_AH - g_t_X_true_AFT))
mu_H0_AFT_H1_AH_2 = np.mean(De_N * W_2 * (g_t_X_true_AH - g_t_X_true_AFT))
mu_H0_AFT_H1_AH_3 = np.mean(De_N * W_3 * (g_t_X_true_AH - g_t_X_true_AFT))
mu_H0_AFT_H1_AH_4 = np.mean(De_N * W_4 * (g_t_X_true_AH - g_t_X_true_AFT))



T_nodes  = uniform_data(5000, 0, tau)
I_T_T_nodes = Indicator_matrix(Y_N, T_nodes) 

g_T_nodes_X_Cox = np.log(np.sqrt(T_nodes.reshape(-1, 1) + 0.001) / 6) + f_X_N.reshape(1, -1) 

g_T_nodes_X_AFT = np.log(2 * T_nodes.reshape(-1, 1) * np.exp(-2 * f_X_N.reshape(1, -1))) 
g_T_nodes_X_AH = np.log(np.maximum((T_nodes.reshape(-1, 1) + 0.001) / 10 + f_X_N.reshape(1, -1), 1e-8)) 

S_T_nodes_X_gene_AH = np.exp(- (T_nodes[:,np.newaxis] + 0.001) ** 2 / 20 - T_nodes[:,np.newaxis] * f_X_gene) 

E_S_T_nodes = np.mean(S_T_nodes_X_gene_AH, axis=1) 

E_I_C_T_noeds =  np.mean(Indicator_matrix(C_gene, T_nodes), axis=0) 

S_T_nodes_X_N = np.exp(- ((T_nodes[:,np.newaxis] + 0.001) ** 2 - 0.001 ** 2) / 20 - T_nodes[:,np.newaxis] * f_X_N) 


W_T_nodes_1 =  np.tile((E_S_T_nodes * E_I_C_T_noeds)[:, np.newaxis], (1, 2000)) 

W_T_nodes_2 =  np.tile((E_S_T_nodes * E_I_C_T_noeds)[:, np.newaxis], (1, 2000)) * S_T_nodes_X_N 

W_T_nodes_3 =  np.tile((E_S_T_nodes * E_I_C_T_noeds)[:, np.newaxis], (1, 2000)) * np.sqrt(S_T_nodes_X_N * (1 - S_T_nodes_X_N)) 

W_T_nodes_4 =  np.tile((E_S_T_nodes * E_I_C_T_noeds)[:, np.newaxis], (1, 2000)) * S_T_nodes_X_N * (1 - S_T_nodes_X_N) 




Psi_1_AH = De_N * W_1 - tau * np.mean(I_T_T_nodes.T * np.exp(g_T_nodes_X_AH) * W_T_nodes_1, axis=0) 
Psi_2_AH = De_N * W_2 - tau * np.mean(I_T_T_nodes.T * np.exp(g_T_nodes_X_AH) * W_T_nodes_2, axis=0) 
Psi_3_AH = De_N * W_3 - tau * np.mean(I_T_T_nodes.T * np.exp(g_T_nodes_X_AH) * W_T_nodes_3, axis=0) 
Psi_4_AH = De_N * W_4 - tau * np.mean(I_T_T_nodes.T * np.exp(g_T_nodes_X_AH) * W_T_nodes_4, axis=0) 



Omega_1_Cox = Psi_1_AH + De_N * W_1 * (g_t_X_true_AH - g_t_X_true_Cox) 
Omega_2_Cox = Psi_2_AH + De_N * W_2 * (g_t_X_true_AH - g_t_X_true_Cox) 
Omega_3_Cox = Psi_3_AH + De_N * W_3 * (g_t_X_true_AH - g_t_X_true_Cox) 
Omega_4_Cox = Psi_4_AH + De_N * W_4 * (g_t_X_true_AH - g_t_X_true_Cox) 


Omega_1_AH = Psi_1_AH + De_N * W_1 * (g_t_X_true_AH - g_t_X_true_AFT) 
Omega_2_AH = Psi_2_AH + De_N * W_2 * (g_t_X_true_AH - g_t_X_true_AFT) 
Omega_3_AH = Psi_3_AH + De_N * W_3 * (g_t_X_true_AH - g_t_X_true_AFT) 
Omega_4_AH = Psi_4_AH + De_N * W_4 * (g_t_X_true_AH - g_t_X_true_AFT) 




sigma_H0_Cox_H1_AH_1 = np.sqrt(np.mean((Omega_1_Cox - np.mean(Omega_1_Cox)) ** 2))
sigma_H0_Cox_H1_AH_2 = np.sqrt(np.mean((Omega_2_Cox - np.mean(Omega_2_Cox)) ** 2))
sigma_H0_Cox_H1_AH_3 = np.sqrt(np.mean((Omega_3_Cox - np.mean(Omega_3_Cox)) ** 2))
sigma_H0_Cox_H1_AH_4 = np.sqrt(np.mean((Omega_4_Cox - np.mean(Omega_4_Cox)) ** 2))


sigma_H0_AFT_H1_AH_1 = np.sqrt(np.mean((Omega_1_AH - np.mean(Omega_1_AH)) ** 2))
sigma_H0_AFT_H1_AH_2 = np.sqrt(np.mean((Omega_2_AH - np.mean(Omega_2_AH)) ** 2))
sigma_H0_AFT_H1_AH_3 = np.sqrt(np.mean((Omega_3_AH - np.mean(Omega_3_AH)) ** 2))
sigma_H0_AFT_H1_AH_4 = np.sqrt(np.mean((Omega_4_AH - np.mean(Omega_4_AH)) ** 2))



sigma_H1_AH_1 = np.sqrt(np.mean(Psi_1_AH ** 2))
sigma_H1_AH_2 = np.sqrt(np.mean(Psi_2_AH ** 2))
sigma_H1_AH_3 = np.sqrt(np.mean(Psi_3_AH ** 2))
sigma_H1_AH_4 = np.sqrt(np.mean(Psi_4_AH ** 2))



Power_Cox_1 = 1 - norm.cdf((sigma_H1_AH_1*1.96 - np.sqrt(800*0.8)*mu_H0_Cox_H1_AH_1) / sigma_H0_AFT_H1_AH_1) + norm.cdf((-sigma_H1_AH_1*1.96 - np.sqrt(800*0.8)*mu_H0_Cox_H1_AH_1) / sigma_H0_AFT_H1_AH_1)
Power_Cox_2 = 1 - norm.cdf((sigma_H1_AH_2*1.96 - np.sqrt(800*0.8)*mu_H0_Cox_H1_AH_2) / sigma_H0_AFT_H1_AH_2) + norm.cdf((-sigma_H1_AH_2*1.96 - np.sqrt(800*0.8)*mu_H0_Cox_H1_AH_2) / sigma_H0_AFT_H1_AH_2)
Power_Cox_3 = 1 - norm.cdf((sigma_H1_AH_3*1.96 - np.sqrt(800*0.8)*mu_H0_Cox_H1_AH_3) / sigma_H0_AFT_H1_AH_3) + norm.cdf((-sigma_H1_AH_3*1.96 - np.sqrt(800*0.8)*mu_H0_Cox_H1_AH_3) / sigma_H0_AFT_H1_AH_3)
Power_Cox_4 = 1 - norm.cdf((sigma_H1_AH_4*1.96 - np.sqrt(800*0.8)*mu_H0_Cox_H1_AH_4) / sigma_H0_AFT_H1_AH_4) + norm.cdf((-sigma_H1_AH_4*1.96 - np.sqrt(800*0.8)*mu_H0_Cox_H1_AH_4) / sigma_H0_AFT_H1_AH_4)


Power_AFT_1 = 1 - norm.cdf((sigma_H1_AH_1*1.96 - np.sqrt(800*0.8)*mu_H0_AFT_H1_AH_1) / sigma_H0_AFT_H1_AH_1) + norm.cdf((-sigma_H1_AH_1*1.96 - np.sqrt(800*0.8)*mu_H0_AFT_H1_AH_1) / sigma_H0_AFT_H1_AH_1)
Power_AFT_2 = 1 - norm.cdf((sigma_H1_AH_2*1.96 - np.sqrt(800*0.8)*mu_H0_AFT_H1_AH_2) / sigma_H0_AFT_H1_AH_2) + norm.cdf((-sigma_H1_AH_2*1.96 - np.sqrt(800*0.8)*mu_H0_AFT_H1_AH_2) / sigma_H0_AFT_H1_AH_2)
Power_AFT_3 = 1 - norm.cdf((sigma_H1_AH_3*1.96 - np.sqrt(800*0.8)*mu_H0_AFT_H1_AH_3) / sigma_H0_AFT_H1_AH_3) + norm.cdf((-sigma_H1_AH_3*1.96 - np.sqrt(800*0.8)*mu_H0_AFT_H1_AH_3) / sigma_H0_AFT_H1_AH_3)
Power_AFT_4 = 1 - norm.cdf((sigma_H1_AH_4*1.96 - np.sqrt(800*0.8)*mu_H0_AFT_H1_AH_4) / sigma_H0_AFT_H1_AH_4) + norm.cdf((-sigma_H1_AH_4*1.96 - np.sqrt(800*0.8)*mu_H0_AFT_H1_AH_4) / sigma_H0_AFT_H1_AH_4)



result_size_DNN_new = pd.DataFrame(
    np.array([
        [Power_Cox_1, Power_Cox_2, Power_Cox_3, Power_Cox_4], 
        [Power_AFT_1, Power_AFT_2, Power_AFT_3, Power_AFT_4]
        ])
    ) 
size_DNN_path_new = "result_True_power_H1_AH.csv"  
result_size_DNN_new.to_csv(size_DNN_path_new, index=False, header=False)  


print('sigma', np.array([sigma_H1_AH_1, sigma_H1_AH_2, sigma_H1_AH_3, sigma_H1_AH_4]))