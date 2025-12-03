# python 3.12.8
#%% ----------------------
import numpy as np
import random
import numpy.random as ndm
import torch
from data_generator import generate_AFT_1, generate_AFT_2_cross
import pandas as pd
from scipy.stats import norm



#%% ---------- -------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)

set_seed(3000)
#%% ----------------------
tau = 2 
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

#%% --------------- power-------------------------

# N1
Data_N1 = generate_AFT_1(2000, corr, tau)
De_N1 = Data_N1['De']
f_X_N1 = Data_N1['f_X']
Y_N1 = Data_N1['T_O']

# N2
Data_N2 = generate_AFT_2_cross(2000, corr, tau)
De_N2 = Data_N2['De']
f_X_N2 = Data_N2['f_X']
Y_N2 = Data_N2['T_O']


# g1,S1
g1_true_N1 = np.log(2 * Y_N1) - 2 * f_X_N1 
g1_true_N2 = np.log(2 * Y_N2) - 2 * f_X_N2 

S1_true_N1 = np.exp(- Y_N1 ** 2 * np.exp(- 2 * f_X_N1)) 
S1_true_N2 = np.exp(- Y_N2 ** 2 * np.exp(- 2 * f_X_N2)) 

f1_true_N1 = np.exp(g1_true_N1) * S1_true_N1 
f1_true_N2 = np.exp(g1_true_N2) * S1_true_N2 


# g2,S2
g2_true_N1 = np.log(1.5 * Y_N1 ** 0.5) - 1.5 * f_X_N1 
g2_true_N2 = np.log(1.5 * Y_N2 ** 0.5) - 1.5 * f_X_N2 

S2_true_N1 = np.exp(- Y_N1 ** 1.5 * np.exp(- 1.5 * f_X_N1)) 
S2_true_N2 = np.exp(- Y_N2 ** 1.5 * np.exp(- 1.5 * f_X_N2)) 

f2_true_N1 = np.exp(g2_true_N1) * S2_true_N1 
f2_true_N2 = np.exp(g2_true_N2) * S2_true_N2 


Data_gene = generate_AFT_1(5000, corr, tau)
X_gene = Data_gene['X'] 
f_X_gene = Data_gene['f_X']
C_gene = Data_gene['De']


S1_t_X_gene_N1 = np.exp(- Y_N1[:,np.newaxis] ** 2 * np.exp(- 2 * f_X_gene))

S1_t_X_gene_N2 = np.exp(- Y_N2[:,np.newaxis] ** 2 * np.exp(- 2 * f_X_gene))



E_S_N1 = np.mean(S1_t_X_gene_N1, axis=1) 
E_I_C_T_N1 =  np.mean(Indicator_matrix(C_gene, Y_N1), axis=0) 

E_S_N2 = np.mean(S1_t_X_gene_N2, axis=1) 
E_I_C_T_N2 =  np.mean(Indicator_matrix(C_gene, Y_N2), axis=0) 


W_1_N1 = E_S_N1 * E_I_C_T_N1 
W_2_N1 = E_S_N1 * E_I_C_T_N1 * S1_true_N1 
W_3_N1 = E_S_N1 * E_I_C_T_N1 * np.sqrt(S1_true_N1 * (1-S1_true_N1)) 
W_4_N1 = E_S_N1 * E_I_C_T_N1 * S1_true_N1 * (1-S1_true_N1) 


W_1_N2 = E_S_N2 * E_I_C_T_N2 
W_2_N2 = E_S_N2 * E_I_C_T_N2 * S1_true_N2 
W_3_N2 = E_S_N2 * E_I_C_T_N2 * np.sqrt(S1_true_N2 * (1-S1_true_N2)) 
W_4_N2 = E_S_N2 * E_I_C_T_N2 * S1_true_N2 * (1-S1_true_N2) 



Eta_1 = (np.mean(De_N1 * W_1_N1 * (g1_true_N1 - g2_true_N1)) + np.mean(De_N2 * W_1_N2 * (g1_true_N2 - g2_true_N2))) / 2
Eta_2 = (np.mean(De_N1 * W_2_N1 * (g1_true_N1 - g2_true_N1)) + np.mean(De_N2 * W_2_N2 * (g1_true_N2 - g2_true_N2))) / 2
Eta_3 = (np.mean(De_N1 * W_3_N1 * (g1_true_N1 - g2_true_N1)) + np.mean(De_N2 * W_3_N2 * (g1_true_N2 - g2_true_N2))) / 2
Eta_4 = (np.mean(De_N1 * W_4_N1 * (g1_true_N1 - g2_true_N1)) + np.mean(De_N2 * W_4_N2 * (g1_true_N2 - g2_true_N2))) / 2



T_nodes  = uniform_data(5000, 0, tau)
E_I_C_T_nodes =  np.mean(Indicator_matrix(C_gene, T_nodes), axis=0) 


I_T_T_nodes_N1 = Indicator_matrix(Y_N1, T_nodes) 
g1_T_nodes_X_N1 = np.log(2 * T_nodes.reshape(-1, 1)) - 2 * f_X_N1.reshape(1, -1) 
g1_T_nodes_X_N2 = np.log(2 * T_nodes.reshape(-1, 1)) - 2 * f_X_N2.reshape(1, -1) 
S1_T_nodes_X_gene = np.exp(- T_nodes[:,np.newaxis] ** 2 * np.exp(- 2 * f_X_gene))
E_S1_T_nodes = np.mean(S1_T_nodes_X_gene, axis=1)  
S1_T_nodes_X_N1 = np.exp(- T_nodes[:,np.newaxis] ** 2 * np.exp(- 2 * f_X_N1)) 
S1_T_nodes_X_N2 = np.exp(- T_nodes[:,np.newaxis] ** 2 * np.exp(- 2 * f_X_N2)) 
f1_T_nodes_X_N1 = np.exp(g1_T_nodes_X_N1) * S1_T_nodes_X_N1 
f1_T_nodes_X_N2 = np.exp(g1_T_nodes_X_N2) * S1_T_nodes_X_N2 


I_T_T_nodes_N2 = Indicator_matrix(Y_N2, T_nodes) 
g2_T_nodes_X_N1 = np.log(1.5 * T_nodes.reshape(-1, 1) ** 0.5) - 1.5 * f_X_N1.reshape(1, -1) 
g2_T_nodes_X_N2 = np.log(1.5 * T_nodes.reshape(-1, 1) ** 0.5) - 1.5 * f_X_N2.reshape(1, -1) 
S2_T_nodes_X_gene = np.exp(- T_nodes[:,np.newaxis] ** 1.5 * np.exp(- 1.5 * f_X_gene))
E_S2_T_nodes = np.mean(S2_T_nodes_X_gene, axis=1)  
S2_T_nodes_X_N1 = np.exp(- T_nodes[:,np.newaxis] ** 1.5 * np.exp(- 1.5 * f_X_N1)) 
S2_T_nodes_X_N2 = np.exp(- T_nodes[:,np.newaxis] ** 1.5 * np.exp(- 1.5 * f_X_N2)) 
f2_T_nodes_X_N1 = np.exp(g2_T_nodes_X_N1) * S2_T_nodes_X_N1 
f2_T_nodes_X_N2 = np.exp(g2_T_nodes_X_N2) * S2_T_nodes_X_N2 




W_T_nodes_1_N1 =  np.tile((E_S1_T_nodes * E_I_C_T_nodes)[:, np.newaxis], (1, 2000)) 
W_T_nodes_2_N1 =  np.tile((E_S1_T_nodes * E_I_C_T_nodes)[:, np.newaxis], (1, 2000)) * S1_T_nodes_X_N1 
W_T_nodes_3_N1 =  np.tile((E_S1_T_nodes * E_I_C_T_nodes)[:, np.newaxis], (1, 2000)) * np.sqrt(S1_T_nodes_X_N1 * (1 - S1_T_nodes_X_N1)) 
W_T_nodes_4_N1 =  np.tile((E_S1_T_nodes * E_I_C_T_nodes)[:, np.newaxis], (1, 2000)) * S1_T_nodes_X_N1 * (1 - S1_T_nodes_X_N1) 


W_T_nodes_1_N2 =  np.tile((E_S1_T_nodes * E_I_C_T_nodes)[:, np.newaxis], (1, 2000)) 
W_T_nodes_2_N2 =  np.tile((E_S1_T_nodes * E_I_C_T_nodes)[:, np.newaxis], (1, 2000)) * S1_T_nodes_X_N2 
W_T_nodes_3_N2 =  np.tile((E_S1_T_nodes * E_I_C_T_nodes)[:, np.newaxis], (1, 2000)) * np.sqrt(S1_T_nodes_X_N2 * (1 - S1_T_nodes_X_N2)) 
W_T_nodes_4_N2 =  np.tile((E_S1_T_nodes * E_I_C_T_nodes)[:, np.newaxis], (1, 2000)) * S1_T_nodes_X_N2 * (1 - S1_T_nodes_X_N2) 




Psi_1_N1 = De_N1 * W_1_N1 - tau * np.mean(I_T_T_nodes_N1.T * np.exp(g1_T_nodes_X_N1) * W_T_nodes_1_N1, axis=0) 
Psi_2_N1 = De_N1 * W_2_N1 - tau * np.mean(I_T_T_nodes_N1.T * np.exp(g1_T_nodes_X_N1) * W_T_nodes_2_N1, axis=0) 
Psi_3_N1 = De_N1 * W_3_N1 - tau * np.mean(I_T_T_nodes_N1.T * np.exp(g1_T_nodes_X_N1) * W_T_nodes_3_N1, axis=0) 
Psi_4_N1 = De_N1 * W_4_N1 - tau * np.mean(I_T_T_nodes_N1.T * np.exp(g1_T_nodes_X_N1) * W_T_nodes_4_N1, axis=0) 

Psi_1_N2 = De_N2 * W_1_N2 - tau * np.mean(I_T_T_nodes_N2.T * np.exp(g2_T_nodes_X_N2) * W_T_nodes_1_N2, axis=0) 
Psi_2_N2 = De_N2 * W_2_N2 - tau * np.mean(I_T_T_nodes_N2.T * np.exp(g2_T_nodes_X_N2) * W_T_nodes_2_N2, axis=0) 
Psi_3_N2 = De_N2 * W_3_N2 - tau * np.mean(I_T_T_nodes_N2.T * np.exp(g2_T_nodes_X_N2) * W_T_nodes_3_N2, axis=0) 
Psi_4_N2 = De_N2 * W_4_N2 - tau * np.mean(I_T_T_nodes_N2.T * np.exp(g2_T_nodes_X_N2) * W_T_nodes_4_N2, axis=0) 



Theta_1_N1 = De_N1 * (1 + f2_true_N1 / f1_true_N1) * W_1_N1 - tau * np.mean(I_T_T_nodes_N1.T * np.exp(g1_T_nodes_X_N1) * (1 + f2_T_nodes_X_N1 / f1_T_nodes_X_N1) * W_T_nodes_1_N1, axis=0) 
Theta_2_N1 = De_N1 * (1 + f2_true_N1 / f1_true_N1) * W_2_N1 - tau * np.mean(I_T_T_nodes_N1.T * np.exp(g1_T_nodes_X_N1) * (1 + f2_T_nodes_X_N1 / f1_T_nodes_X_N1) * W_T_nodes_2_N1, axis=0) 
Theta_3_N1 = De_N1 * (1 + f2_true_N1 / f1_true_N1) * W_3_N1 - tau * np.mean(I_T_T_nodes_N1.T * np.exp(g1_T_nodes_X_N1) * (1 + f2_T_nodes_X_N1 / f1_T_nodes_X_N1) * W_T_nodes_3_N1, axis=0) 
Theta_4_N1 = De_N1 * (1 + f2_true_N1 / f1_true_N1) * W_4_N1 - tau * np.mean(I_T_T_nodes_N1.T * np.exp(g1_T_nodes_X_N1) * (1 + f2_T_nodes_X_N1 / f1_T_nodes_X_N1) * W_T_nodes_4_N1, axis=0) 

Theta_1_N2 = De_N2 * (1 + f1_true_N2 / f2_true_N2) * W_1_N2 - tau * np.mean(I_T_T_nodes_N2.T * np.exp(g2_T_nodes_X_N2) * (1 + f1_T_nodes_X_N2 / f2_T_nodes_X_N2) * W_T_nodes_1_N2, axis=0) 
Theta_2_N2 = De_N2 * (1 + f1_true_N2 / f2_true_N2) * W_2_N2 - tau * np.mean(I_T_T_nodes_N2.T * np.exp(g2_T_nodes_X_N2) * (1 + f1_T_nodes_X_N2 / f2_T_nodes_X_N2) * W_T_nodes_2_N2, axis=0) 
Theta_3_N2 = De_N2 * (1 + f1_true_N2 / f2_true_N2) * W_3_N2 - tau * np.mean(I_T_T_nodes_N2.T * np.exp(g2_T_nodes_X_N2) * (1 + f1_T_nodes_X_N2 / f2_T_nodes_X_N2) * W_T_nodes_3_N2, axis=0) 
Theta_4_N2 = De_N2 * (1 + f1_true_N2 / f2_true_N2) * W_4_N2 - tau * np.mean(I_T_T_nodes_N2.T * np.exp(g2_T_nodes_X_N2) * (1 + f1_T_nodes_X_N2 / f2_T_nodes_X_N2) * W_T_nodes_4_N2, axis=0) 





sigma_1w_1 = np.sqrt(np.mean(Psi_1_N1 ** 2))
sigma_1w_2 = np.sqrt(np.mean(Psi_2_N1 ** 2))
sigma_1w_3 = np.sqrt(np.mean(Psi_3_N1 ** 2))
sigma_1w_4 = np.sqrt(np.mean(Psi_4_N1 ** 2))

sigma_2w_1 = np.sqrt(np.mean(Psi_1_N2 ** 2))
sigma_2w_2 = np.sqrt(np.mean(Psi_2_N2 ** 2))
sigma_2w_3 = np.sqrt(np.mean(Psi_3_N2 ** 2))
sigma_2w_4 = np.sqrt(np.mean(Psi_4_N2 ** 2))






Omega_1_N1 = Theta_1_N1 + De_N1 * W_1_N1 * (g1_true_N1 - g2_true_N1) 
Omega_2_N1 = Theta_2_N1 + De_N1 * W_2_N1 * (g1_true_N1 - g2_true_N1) 
Omega_3_N1 = Theta_3_N1 + De_N1 * W_3_N1 * (g1_true_N1 - g2_true_N1) 
Omega_4_N1 = Theta_4_N1 + De_N1 * W_4_N1 * (g1_true_N1 - g2_true_N1) 

sigma_1w_g1g2_1 = np.sqrt(np.mean((Omega_1_N1 - np.mean(Omega_1_N1)) ** 2))
sigma_1w_g1g2_2 = np.sqrt(np.mean((Omega_2_N1 - np.mean(Omega_2_N1)) ** 2))
sigma_1w_g1g2_3 = np.sqrt(np.mean((Omega_3_N1 - np.mean(Omega_3_N1)) ** 2))
sigma_1w_g1g2_4 = np.sqrt(np.mean((Omega_4_N1 - np.mean(Omega_4_N1)) ** 2))

Omega_1_N2 = Theta_1_N2 + De_N2 * W_1_N2 * (g1_true_N2 - g2_true_N2) 
Omega_2_N2 = Theta_2_N2 + De_N2 * W_2_N2 * (g1_true_N2 - g2_true_N2) 
Omega_3_N2 = Theta_3_N2 + De_N2 * W_3_N2 * (g1_true_N2 - g2_true_N2) 
Omega_4_N2 = Theta_4_N2 + De_N2 * W_4_N2 * (g1_true_N2 - g2_true_N2) 

sigma_2w_g1g2_1 = np.sqrt(np.mean((Omega_1_N2 - np.mean(Omega_1_N2)) ** 2))
sigma_2w_g1g2_2 = np.sqrt(np.mean((Omega_2_N2 - np.mean(Omega_2_N2)) ** 2))
sigma_2w_g1g2_3 = np.sqrt(np.mean((Omega_3_N2 - np.mean(Omega_3_N2)) ** 2))
sigma_2w_g1g2_4 = np.sqrt(np.mean((Omega_4_N2 - np.mean(Omega_4_N2)) ** 2))




Power_1 = 1 - norm.cdf((np.sqrt(2 * (sigma_1w_1 ** 2 + sigma_2w_1 ** 2)) * 1.96 - np.sqrt(1600 * 0.8) * Eta_1) / np.sqrt((1/2) * (sigma_1w_g1g2_1 ** 2 + sigma_2w_g1g2_1 ** 2))) + norm.cdf((-np.sqrt(2 * (sigma_1w_1 ** 2 + sigma_2w_1 ** 2)) * 1.96 - np.sqrt(1600 * 0.8) * Eta_1) / np.sqrt((1/2) * (sigma_1w_g1g2_1 ** 2 + sigma_2w_g1g2_1 ** 2)))

Power_2 = 1 - norm.cdf((np.sqrt(2 * (sigma_1w_2 ** 2 + sigma_2w_2 ** 2)) * 1.96 - np.sqrt(1600 * 0.8) * Eta_2) / np.sqrt((1/2) * (sigma_1w_g1g2_2 ** 2 + sigma_2w_g1g2_2 ** 2))) + norm.cdf((-np.sqrt(2 * (sigma_1w_2 ** 2 + sigma_2w_2 ** 2)) * 1.96 - np.sqrt(1600 * 0.8) * Eta_2) / np.sqrt((1/2) * (sigma_1w_g1g2_2 ** 2 + sigma_2w_g1g2_2 ** 2)))

Power_3 = 1 - norm.cdf((np.sqrt(2 * (sigma_1w_3 ** 2 + sigma_2w_3 ** 2)) * 1.96 - np.sqrt(1600 * 0.8) * Eta_3) / np.sqrt((1/2) * (sigma_1w_g1g2_3 ** 2 + sigma_2w_g1g2_3 ** 2))) + norm.cdf((-np.sqrt(2 * (sigma_1w_3 ** 2 + sigma_2w_3 ** 2)) * 1.96 - np.sqrt(1600 * 0.8) * Eta_3) / np.sqrt((1/2) * (sigma_1w_g1g2_3 ** 2 + sigma_2w_g1g2_3 ** 2)))


Power_4 = 1 - norm.cdf((np.sqrt(2 * (sigma_1w_4 ** 2 + sigma_2w_4 ** 2)) * 1.96 - np.sqrt(1600 * 0.8) * Eta_4) / np.sqrt((1/2) * (sigma_1w_g1g2_4 ** 2 + sigma_2w_g1g2_4 ** 2))) + norm.cdf((-np.sqrt(2 * (sigma_1w_4 ** 2 + sigma_2w_4 ** 2)) * 1.96 - np.sqrt(1600 * 0.8) * Eta_4) / np.sqrt((1/2) * (sigma_1w_g1g2_4 ** 2 + sigma_2w_g1g2_4 ** 2)))



result_size_DNN_new = pd.DataFrame(
    np.array([Power_1, Power_2, Power_3, Power_4])
    ) 
size_DNN_path_new = f"result_True_power_AFT_cross.csv"  
result_size_DNN_new.to_csv(size_DNN_path_new, index=False, header=False) 

