import numpy as np
import scipy.optimize as spo
from B_spline import B_S

def Indicator_matrix(a, b):
    a = np.array(a)
    b = np.array(b)
    I_M = (a[:, np.newaxis] >= b).astype(int)
    return I_M



def Beta_t(m, ts, nodevec, coefs):
    Beta_ts = np.zeros((len(ts),7))
    Beta_ts[:,0] = B_S(m, ts, nodevec) @ coefs[0:(m+4)] 
    Beta_ts[:,1] = B_S(m, ts, nodevec) @ coefs[(m+4):(2 *(m+4))] 
    Beta_ts[:,2] = B_S(m, ts, nodevec) @ coefs[(2*(m+4)):(3*(m+4))] 
    Beta_ts[:,3] = B_S(m, ts, nodevec) @ coefs[(3*(m+4)):(4*(m+4))] 
    Beta_ts[:,4] = B_S(m, ts, nodevec) @ coefs[(4*(m+4)):(5*(m+4))] 
    Beta_ts[:,5] = B_S(m, ts, nodevec) @ coefs[(5*(m+4)):(6*(m+4))] 
    Beta_ts[:,6] = B_S(m, ts, nodevec) @ coefs[(6*(m+4)):(7*(m+4))]
    return Beta_ts


def Surv_pred(m, ts, nodevec, coefs, t_nodes, X_test, tau):
    Beta_t_nodes = Beta_t(m, t_nodes, nodevec, coefs[0:7*(m+4)]) 
    BX_t_nodes = Beta_t_nodes @ X_test.T 
    lambda_t_nodes = B_S(m, t_nodes, nodevec) @ coefs[(7*(m+4)):(8*(m+4))] 
    S_t_X = np.exp(- Indicator_matrix(ts, t_nodes) @ np.diag(lambda_t_nodes) @ np.exp(BX_t_nodes) * (tau / len(t_nodes)))
    return S_t_X




def Est_Coxvarying(X_train, Y_train, De_train, t_nodes, m, nodevec, tau):
    def CF(*args):
        c = args[0]
        Beta_Y = Beta_t(m, Y_train, nodevec, c[0:7*(m+4)]) 
        BX_Y = np.sum(Beta_Y * X_train, axis=1) 
        Beta_t_nodes = Beta_t(m, t_nodes, nodevec, c[0:7*(m+4)])  
        BX_t_nodes = X_train @ Beta_t_nodes.T 
        lambda_t_nodes = B_S(m, t_nodes, nodevec) @ c[(7*(m+4)):(8*(m+4))] 
        lambda_Y = B_S(m, Y_train, nodevec) @ c[(7*(m+4)):(8*(m+4))] 
        Loss = - np.mean(De_train * (np.log(lambda_Y + 1e-5) + BX_Y) - 
                         (Indicator_matrix(Y_train, t_nodes) * np.exp(BX_t_nodes)) @ lambda_t_nodes * (tau / len(t_nodes)))
        return Loss

    initial_c = 0.1 * np.ones(8*(m+4))

    bounds = [(None, None)] * (7*(m+4)) + [(0, None)] * (m+4)

    result = spo.minimize(CF, initial_c, method='SLSQP', bounds=bounds)

    return result['x']


def Surv_Coxvarying(X_train, Y_train, De_train, X_test, t_nodes, m, nodevec, tau, s_k):
    coefs = Est_Coxvarying(X_train, Y_train, De_train, t_nodes, m, nodevec, tau)
    S_t_X_Coxvary_IBS = Surv_pred(m, s_k, nodevec, coefs, t_nodes, X_test, tau)

    return{
        "S_t_X_Coxvary_IBS": S_t_X_Coxvary_IBS
    }

