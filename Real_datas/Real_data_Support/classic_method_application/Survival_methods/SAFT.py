# Nan Bin's method
import numpy as np
import scipy.optimize as spo
from B_spline import B_S


def Indicator_matrix(a, b):
    a = np.array(a)
    b = np.array(b)
    I_M = (a[:, np.newaxis] >= b).astype(int)
    return I_M

def SAFT_est(De, t_x_nodes, nodes_num, node_vec, Omega_b, Y_train, X_train):
    d = X_train.shape[1]
    def CF(c):
        beta_X_train = X_train @ c[:d]
        Y_beta_X_train = np.array(Y_train * np.exp(beta_X_train), dtype="float32")
        I_t_x_nodes_Y_X_train = Indicator_matrix(Y_beta_X_train, t_x_nodes)
        Loss = De * (B_S(nodes_num, Y_beta_X_train, node_vec) @ c[d:(d+ nodes_num+4)] + beta_X_train) - (Omega_b / len(t_x_nodes)) * I_t_x_nodes_Y_X_train @ np.exp(B_S(nodes_num, t_x_nodes, node_vec) @ c[d:(d+ nodes_num+4)])
        return -Loss.mean()

    result = spo.minimize(CF, np.ones(d+nodes_num+4), method='SLSQP')
    return result['x'][:d]
