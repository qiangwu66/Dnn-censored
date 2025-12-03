import numpy as np
import scipy.optimize as spo
from B_spline import B_S

def Indicator_matrix(a, b):
    a = np.array(a)
    b = np.array(b)
    I_M = (a[:, np.newaxis] >= b).astype(int)
    return I_M



def Beta_t(m, ts, nodevec, coefs):
    Beta_ts = np.zeros((len(ts),6))
    Beta_ts[:,0] = B_S(m, ts, nodevec) @ coefs[0:(m+4)] 
    Beta_ts[:,1] = B_S(m, ts, nodevec) @ coefs[(m+4):(2 *(m+4))] 
    Beta_ts[:,2] = B_S(m, ts, nodevec) @ coefs[(2*(m+4)):(3*(m+4))]
    Beta_ts[:,3] = B_S(m, ts, nodevec) @ coefs[(3*(m+4)):(4*(m+4))]
    Beta_ts[:,4] = B_S(m, ts, nodevec) @ coefs[(4*(m+4)):(5*(m+4))]
    Beta_ts[:,5] = B_S(m, ts, nodevec) @ coefs[(5*(m+4)):(6*(m+4))]
    return Beta_ts


def Surv_pred(m, ts, nodevec, coefs, t_nodes, X_test, tau):
    Beta_t_nodes = Beta_t(m, t_nodes, nodevec, coefs[0:6*(m+4)]) 
    BX_t_nodes = Beta_t_nodes @ X_test.T 
    lambda_t_nodes = B_S(m, t_nodes, nodevec) @ coefs[(6*(m+4)):(7*(m+4))] 
    S_t_X = np.exp(- Indicator_matrix(ts, t_nodes) @ np.diag(lambda_t_nodes) @ np.exp(BX_t_nodes) * (tau / len(t_nodes)))
    return S_t_X




def Est_Coxvarying(train_data, t_nodes, m, nodevec, tau):
    X_train = train_data['X']
    Y_train = train_data['T_O']
    De_train = train_data['De']
    
    def CF(*args):
        c = args[0]
        Beta_Y = Beta_t(m, Y_train, nodevec, c[0:6*(m+4)]) 
        BX_Y = np.sum(Beta_Y * X_train, axis=1) 
        Beta_t_nodes = Beta_t(m, t_nodes, nodevec, c[0:6*(m+4)]) 
        BX_t_nodes = X_train @ Beta_t_nodes.T  
        lambda_t_nodes = B_S(m, t_nodes, nodevec) @ c[(6*(m+4)):(7*(m+4))] 
        lambda_Y = B_S(m, Y_train, nodevec) @ c[(6*(m+4)):(7*(m+4))] 
        Loss = - np.mean(De_train * (np.log(lambda_Y + 1e-5) + BX_Y) - 
                         (Indicator_matrix(Y_train, t_nodes) * np.exp(BX_t_nodes)) @ lambda_t_nodes * (tau / len(t_nodes)))
        return Loss


    initial_c = np.ones(7*(m+4))

    bounds = [(None, None)] * (6*(m+4)) + [(0, None)] * (m+4) 

    result = spo.minimize(CF, initial_c, method='SLSQP', bounds=bounds)

    return result['x']




def Surv_Coxvarying(train_data, test_data, t_nodes, m, nodevec, tau, n1, s_k, t_fig):
    coefs = Est_Coxvarying(train_data, t_nodes, m, nodevec, tau)
    X_test = test_data['X']
    Y_test = test_data['T_O'] 
    S_t_X_Coxvary_fig = Surv_pred(m, t_fig, nodevec, coefs, t_nodes, X_test, tau) 
    S_t_X_Coxvary_IBS = Surv_pred(m, s_k, nodevec, coefs, t_nodes, X_test, tau)  
    S_t_X_Coxvary_DMS = Surv_pred(m, Y_test, nodevec, coefs, t_nodes, X_test, tau) 

    return{
        "S_t_X_Coxvary_fig": S_t_X_Coxvary_fig,
        "S_t_X_Coxvary_IBS": S_t_X_Coxvary_IBS,
        "S_t_X_Coxvary_DMS": S_t_X_Coxvary_DMS
    }

