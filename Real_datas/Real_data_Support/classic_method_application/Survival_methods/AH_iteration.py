import numpy as np
import scipy.optimize as spo
from B_spline import B_S

def Indicator_matrix(a, b):
    a = np.array(a)
    b = np.array(b)
    I_M = (a[:, np.newaxis] >= b).astype(int)
    return I_M

def Est_AH(X_train, Y_train, De_train, t_nodes, m, nodevec, tau):
    d = X_train.shape[1]
    def CF(*args):
        c = args[0]
        Beta_X = X_train @ c[:d] # n
        lambda_t_nodes = B_S(m, t_nodes, nodevec) @ c[d:(d+m+4)]  # len(t_nodes)
        lambda_Y = B_S(m, Y_train, nodevec) @ c[d:(d+m+4)] # n
        Loss = - np.mean(De_train * np.log(np.maximum(lambda_Y + Beta_X, 1e-5)) - 
                         Indicator_matrix(Y_train, t_nodes) @ lambda_t_nodes * (tau / len(t_nodes)) - Y_train * Beta_X)
        return Loss

    # 初始化参数
    initial_c = 0.1 * np.ones(d+m+4)
    # 设置参数的边界：所有参数默认无约束，最后 (m+4) 个参数非负
    bounds = [(None, None)] * d + [(0, None)] * (m+4)  # [(None, None)] 表示无约束，(0, None) 表示非负
    # 优化
    result = spo.minimize(CF, initial_c, method='SLSQP', bounds=bounds)
    return result['x'][:d]
