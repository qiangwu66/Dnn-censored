import numpy as np
import scipy.optimize as spo
from B_spline import B_S

def Indicator_matrix(a, b):
    a = np.array(a)
    b = np.array(b)
    I_M = (a[:, np.newaxis] >= b).astype(int)
    return I_M



def Est_Coxph(X_train, Y_train, De_train, t_nodes, m, nodevec, tau):
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
    return result['x'][:d]

