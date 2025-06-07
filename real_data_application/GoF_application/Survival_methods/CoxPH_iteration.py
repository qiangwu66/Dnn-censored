import numpy as np
import scipy.optimize as spo
from B_spline import B_S

def Indicator_matrix(a, b):
    a = np.array(a)
    b = np.array(b)
    I_M = (a[:, np.newaxis] >= b).astype(int)
    return I_M


def Est_Coxph(train_data, t_nodes, m, nodevec, tau):
    X_train = train_data['X']
    Y_train = train_data['T_O']
    De_train = train_data['De']
    d = X_train.shape[1]
    def COF(*args):
        c = args[0]
        Beta_X = X_train @ c[:d]
        lambda_t_nodes = B_S(m, t_nodes, nodevec) @ c[d:(d+m+4)]  
        lambda_Y = B_S(m, Y_train, nodevec) @ c[d:(d+m+4)]
        Loss = - np.mean(De_train * (np.log(lambda_Y + 1e-5) + Beta_X)- 
                         Indicator_matrix(Y_train, t_nodes) @ lambda_t_nodes * (tau / len(t_nodes)) * np.exp(Beta_X))
        return Loss


    initial_c = 0.1 * np.ones(d+m+4)

    bounds = [(None, None)] * d + [(0, None)] * (m+4)

    result = spo.minimize(COF, initial_c, method='SLSQP', bounds=bounds)
    return result['x']



def Estimates_Coxph(polled_data, train_data, t_nodes, m, nodevec, tau):
    T_O_train = train_data['T_O']
    X_train = train_data['X']
    De_train = train_data['De']

    T_O_all = polled_data['T_O']
    X_all = polled_data['X']
    De_all = polled_data['De']

    coefs = Est_Coxph(train_data, t_nodes, m, nodevec, tau)
    d = X_all.shape[1]

    Beta_X_all = X_all @ coefs[:d] 
    lambda_t_all = B_S(m, T_O_all, nodevec) @ coefs[d:(d+m+4)]
    g0_T_X_n = np.log(lambda_t_all + 1e-4) + Beta_X_all


    lambda_t_nodes = B_S(m, t_nodes, nodevec) @ coefs[d:(d+m+4)]

    # -----------------------------------------
    Beta_X_train = X_train @ coefs[:d] 
    I_T_T_n = Indicator_matrix(T_O_train, T_O_train)

    I_T_T_mean = np.mean(I_T_T_n, axis=0)
    W_1_n = I_T_T_mean

    I_T_t_nodes_n = Indicator_matrix(T_O_train, t_nodes)

    int_Y_exp_1 = (tau / len(t_nodes)) * (I_T_t_nodes_n @ (lambda_t_nodes * np.mean(I_T_t_nodes_n, axis = 0))) * np.exp(Beta_X_train)

    sigma_1_n2 = np.sqrt(np.mean((De_train * W_1_n - int_Y_exp_1) ** 2)) 

    

    return{
        "g0_T_X_n": g0_T_X_n,
        'sigma_1_n2_coxph': sigma_1_n2,
    }

