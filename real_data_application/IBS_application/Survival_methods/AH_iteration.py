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
    Beta_X = X_test @ coefs[:d] # 200
    T_Beta_X = np.outer(ts, Beta_X) # len(ts) * 200
    lambda_t_nodes = B_S(m, t_nodes, nodevec) @ coefs[d:(d+m+4)]  # len(t_nodes) 
    S_t_X = np.exp(- np.repeat((Indicator_matrix(ts, t_nodes) @ lambda_t_nodes * (tau / len(t_nodes)))[:, np.newaxis], X_test.shape[0], axis=1) - T_Beta_X) # len(ts) * 200
    return S_t_X


def Est_AH(X_train, Y_train, De_train, t_nodes, m, nodevec, tau):
    d = X_train.shape[1]
    def CF(*args):
        c = args[0]
        Beta_X = X_train @ c[:d] # n
        lambda_t_nodes = B_S(m, t_nodes, nodevec) @ c[d:(d+m+4)]  # len(t_nodes)
        lambda_Y = B_S(m, Y_train, nodevec) @ c[d:(d+m+4)] # n
        Loss = - np.mean(De_train * np.log(lambda_Y + Beta_X + 1e-5) - 
                         Indicator_matrix(Y_train, t_nodes) @ lambda_t_nodes * (tau / len(t_nodes)) - Y_train * Beta_X)
        return Loss

    # 初始化参数
    initial_c = 0.1 * np.ones(d+m+4)
    # 设置参数的边界：所有参数默认无约束，最后 (m+4) 个参数非负
    bounds = [(None, None)] * d + [(0, None)] * (m+4)  # [(None, None)] 表示无约束，(0, None) 表示非负
    # 优化
    result = spo.minimize(CF, initial_c, method='SLSQP', bounds=bounds)
    return result['x']




def Surv_AH(X_train, Y_train, De_train, X_test, t_nodes, m, nodevec, tau, s_k):
    coefs = Est_AH(X_train, Y_train, De_train, t_nodes, m, nodevec, tau)
    S_t_X_AH_IBS = Surv_pred(m, s_k, nodevec, coefs, t_nodes, X_test, tau) 
    return{
        "S_t_X_AH_IBS": S_t_X_AH_IBS
    }

