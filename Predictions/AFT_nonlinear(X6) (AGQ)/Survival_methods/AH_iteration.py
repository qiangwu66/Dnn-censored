import numpy as np
import scipy.optimize as spo
from B_spline import B_S

def Indicator_matrix(a, b):
    a = np.array(a)
    b = np.array(b)
    I_M = (a[:, np.newaxis] >= b).astype(int)
    return I_M


def Surv_pred(m, ts, nodevec, coefs, t_nodes, X_test, tau):
    d = X_test.shape[1]
    Beta_X = X_test @ coefs[:d] 
    T_Beta_X = np.outer(ts, Beta_X) 
    lambda_t_nodes = B_S(m, t_nodes, nodevec) @ coefs[d:(d+m+4)]   
    S_t_X = np.exp(- np.repeat((Indicator_matrix(ts, t_nodes) @ lambda_t_nodes * (tau / len(t_nodes)))[:, np.newaxis], X_test.shape[0], axis=1) - T_Beta_X) 
    return S_t_X


def Est_AH(train_data, t_nodes, m, nodevec, tau):
    X_train = train_data['X']
    Y_train = train_data['T_O']
    De_train = train_data['De']
    d = X_train.shape[1]
    def CF(*args):
        c = args[0]
        Beta_X = X_train @ c[:d] 
        lambda_t_nodes = B_S(m, t_nodes, nodevec) @ c[d:(d+m+4)]  
        lambda_Y = B_S(m, Y_train, nodevec) @ c[d:(d+m+4)] 
        Loss = - np.mean(De_train * np.log(lambda_Y + Beta_X + 1e-5) - 
                         Indicator_matrix(Y_train, t_nodes) @ lambda_t_nodes * (tau / len(t_nodes)) - Y_train * Beta_X)
        return Loss

    initial_c = 0.1 * np.ones(d+m+4)
    bounds = [(None, None)] * d + [(0, None)] * (m+4)  
    result = spo.minimize(CF, initial_c, method='SLSQP', bounds=bounds)
    return result['x']




def Surv_AH(train_data, test_data, t_nodes, m, nodevec, tau, s_k, t_fig):
    coefs = Est_AH(train_data, t_nodes, m, nodevec, tau)
    X_test = test_data['X']
    Y_test = test_data['T_O'] 
    S_t_X_AH_fig = Surv_pred(m, t_fig, nodevec, coefs, t_nodes, X_test, tau) 
    S_t_X_AH_IBS = Surv_pred(m, s_k, nodevec, coefs, t_nodes, X_test, tau)  
    S_t_X_AH_DMS = Surv_pred(m, Y_test, nodevec, coefs, t_nodes, X_test, tau)  
    return{
        "S_t_X_AH_fig": S_t_X_AH_fig,
        "S_t_X_AH_IBS": S_t_X_AH_IBS,
        "S_t_X_AH_DMS": S_t_X_AH_DMS
    }

