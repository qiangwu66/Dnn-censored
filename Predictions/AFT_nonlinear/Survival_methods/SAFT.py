# Nan Bin's method
import numpy as np
import scipy.optimize as spo
from B_spline import B_S

def SAFT_C_est(De, t_x_nodes, beta_X_train, Y_beta_X_train, I_t_x_nodes_Y_X_train, nodes_num, node_vec, Omega_b):
    def CF(c):
        Loss = De * (B_S(nodes_num, Y_beta_X_train, node_vec) @ c + beta_X_train) - (Omega_b / len(t_x_nodes)) * I_t_x_nodes_Y_X_train @ np.exp(B_S(nodes_num, t_x_nodes, node_vec) @ c)
        return -Loss.mean()

    result = spo.minimize(CF, np.ones(nodes_num+4), method='SLSQP')
    return result['x']


