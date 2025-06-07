import numpy as np
import scipy.optimize as spo
from B_spline import B_S
from lifelines import CoxPHFitter
import pandas as pd



def Indicator_matrix(a, b):
    a = np.array(a)
    b = np.array(b)
    I_M = (a[:, np.newaxis] >= b).astype(int)
    return I_M




def SAFT_C_est(De, t_x_nodes, beta_X_train, Y_beta_X_train, I_t_x_nodes_Y_X_train, nodes_num, node_vec, Omega_b):
    def CF(c):
        Loss = De * (B_S(nodes_num, Y_beta_X_train, node_vec) @ c + beta_X_train) - (Omega_b / len(t_x_nodes)) * I_t_x_nodes_Y_X_train @ np.exp(B_S(nodes_num, t_x_nodes, node_vec) @ c)
        return -Loss.mean()
    result = spo.minimize(CF, np.ones(nodes_num+4), method='SLSQP')
    return result['x']



def Estimates_AFT(polled_data, train_data, t_nodes, m, nodevec, tau):
    T_O_train = train_data['T_O']
    X_train = train_data['X']
    De_train = train_data['De']

    T_O_all = polled_data['T_O']
    X_all = polled_data['X']
    d = X_all.shape[1]

    columns = [f'feature_{i+1}' for i in range(d)]
    data = pd.DataFrame(train_data['X'], columns=columns)
    data['duration'] = np.maximum(train_data['T_O'], 1e-4)
    data['event'] = train_data['De']
    cph = CoxPHFitter()
    cph.fit(data, duration_col='duration', event_col='event')

    beta_coefs = cph.summary[['coef']].values[:,0]
    Omega_b = np.max(T_O_all) * np.exp(np.max(X_all @ beta_coefs))

    nodes_num = int((0.8 * len(T_O_train)) ** (1 / 4))
    node_vec = np.array(np.linspace(0, Omega_b, nodes_num + 2), dtype="float32") 
    
    t_x_nodes = np.array(np.linspace(0, Omega_b, 501), dtype="float32")
    beta_X_train = np.array(train_data['X'] @ beta_coefs, dtype='float32') 
    Y_beta_X_train = np.array(train_data['T_O'] * np.exp(beta_X_train), dtype="float32") 
    I_t_x_nodes_Y_X_train = Indicator_matrix(Y_beta_X_train, t_x_nodes) 
    c_saft = SAFT_C_est(train_data['De'], t_x_nodes, beta_X_train, Y_beta_X_train, I_t_x_nodes_Y_X_train, nodes_num, node_vec, Omega_b)

    # np.exp(B_S(nodes_num, t_x_nodes, node_vec) @ c_saft)
    beta_X_all = np.array(X_all @ beta_coefs, dtype='float32') 
    Y_beta_X_all = np.array(T_O_all * np.exp(beta_X_all), dtype="float32")

    g0_T_X_n = B_S(nodes_num, Y_beta_X_all, node_vec) @ c_saft + beta_X_all

    # ----------------------------------------- 
    I_T_T_n = Indicator_matrix(T_O_train, T_O_train)

    I_T_T_mean = np.mean(I_T_T_n, axis=0)
    W_1_n = I_T_T_mean


    int_Y_exp_1 = (Omega_b / len(t_x_nodes)) * I_t_x_nodes_Y_X_train @ np.exp(B_S(nodes_num, t_x_nodes, node_vec) @ c_saft)
    sigma_1_n2 = np.sqrt(np.mean((De_train * W_1_n - int_Y_exp_1) ** 2)) 

    return{
        "g0_T_X_n": g0_T_X_n,
        'sigma_1_n2_AFT': sigma_1_n2,
    }

