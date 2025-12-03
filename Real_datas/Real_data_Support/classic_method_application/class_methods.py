# python 3.12.8
#%% ----------------------
import numpy as np
import random
import torch
from scipy.stats import norm
from B_spline import *
import pandas as pd
import os
from lifelines import CoxPHFitter

from Survival_methods.CoxPH_iteration import Est_Coxph
from Survival_methods.AH_iteration import Est_AH


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



t_nodes = np.array(np.linspace(0, 1, 101), dtype="float32")
t_nodes = np.maximum(t_nodes, 1e-4)
m_CoxPH = 4
nodevec_CoxPH = np.array(np.linspace(0, tau, m_CoxPH + 2), dtype="float32")






Omega_b = np.max(Time) * np.exp(np.max(X @ np.ones(7)))

nodes_AFT = 4
nodevec_AFT = np.array(np.linspace(0, Omega_b, nodes_AFT + 2), dtype="float32")  
t_x_nodes = np.array(np.linspace(0, Omega_b, 501), dtype="float32")  


m_AH = 4
nodevec_AH = np.array(np.linspace(0, tau, m_AH + 2), dtype="float32")

bs_no = 200

Beta_bs_Cox = np.full((bs_no, X.shape[1]), np.nan)
Beta_bs_AFT = np.full((bs_no, X.shape[1]), np.nan)
Beta_bs_AH = np.full((bs_no, X.shape[1]), np.nan)

for k in range(bs_no):
    bootstrap_index_k = np.random.choice(range(X.shape[0]), size=1000, replace=True)
    train_data_bs_k = {
            'X': np.array(X[bootstrap_index_k, :], dtype='float32'),
            'Time': np.array(Time[bootstrap_index_k], dtype='float32'),
            'Delta': np.array(De[bootstrap_index_k], dtype='float32'),
        }
    
    #-------------------------------------------
    Beta_bs_Cox[k] = Est_Coxph(train_data_bs_k['X'], train_data_bs_k['Time'], train_data_bs_k['Delta'], t_nodes, m_CoxPH, nodevec_CoxPH, tau)

    # ------------------------------------------
    columns = [f'feature_{i+1}' for i in range(train_data_bs_k['X'].shape[1])]
    data = pd.DataFrame(train_data_bs_k['X'], columns=columns)
    data['duration'] = np.maximum(train_data_bs_k['Time'], 1e-4)
    data['event'] = train_data_bs_k['Delta']

    cph = CoxPHFitter()
    cph.fit(data, duration_col='duration', event_col='event')
    beta_coefs = cph.summary[['coef']].values[:,0]
    Beta_bs_AFT[k] = beta_coefs

    # ------------------------------------------
    Beta_bs_AH[k] = Est_AH(train_data_bs_k['X'], train_data_bs_k['Time'], train_data_bs_k['Delta'], t_nodes, m_AH, nodevec_AH, tau)
    



means_Cox = np.mean(Beta_bs_Cox, axis=0) 
stds_Cox = np.std(Beta_bs_Cox, axis=0, ddof=1)
standard_errors_Cox = stds_Cox / np.sqrt(len(De)) 
z_stats_Cox = means_Cox/ standard_errors_Cox
p_values_Cox = 2 * (1 - norm.cdf(np.abs(z_stats_Cox)))


means_AFT = np.mean(Beta_bs_AFT, axis=0) 
stds_AFT = np.std(Beta_bs_AFT, axis=0, ddof=1)
standard_errors_AFT = stds_AFT / np.sqrt(len(De)) 
z_stats_AFT = means_AFT/ standard_errors_AFT
p_values_AFT = 2 * (1 - norm.cdf(np.abs(z_stats_AFT)))


means_AH = np.mean(Beta_bs_AH, axis=0) 
stds_AH = np.std(Beta_bs_AH, axis=0, ddof=1)
standard_errors_AH = stds_AH / np.sqrt(len(De)) 
z_stats_AH = means_AH/ standard_errors_AH
p_values_AH = 2 * (1 - norm.cdf(np.abs(z_stats_AH)))



# =================IBS-Table=======================
output_folder = "Result_Est_class"
os.makedirs(output_folder, exist_ok=True)


Est_p = {
    "Beta_Cox": means_Cox,
    "stds_Cox": stds_Cox,
    "p_values_Cox": p_values_Cox,
    "Beta_AFT": means_AFT,
    "stds_AFT": stds_AFT,
    "p_values_AFT": p_values_AFT,
    "Beta_AH": means_AH,
    "stds_AH": stds_AH,
    "p_values_AH": p_values_AH,
}
result_IBS = pd.DataFrame(Est_p)
ibs_path = os.path.join(output_folder, "est_class.csv")
result_IBS.to_csv(ibs_path, index=False)