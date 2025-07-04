import numpy as np
import numpy.random as ndm


def generate_case_2(n, corr, tau):
    Z1 = ndm.binomial(1, 0.5, n)
    Z2 = ndm.binomial(1, 0.25, n)
    mean = np.zeros(4)
    cov = np.identity(4) * (1-corr) + np.ones((4, 4)) * corr
    X0 = ndm.multivariate_normal(mean, cov, n)
    X1 = (np.clip(X0, -2, 2) + 2) / 4 # 归一化[0,1]
    f_X = (Z1 + Z2) ** 2 + np.log(np.sin(X1[:,0] * np.pi) + 4 * X1[:,1] * X1[:,2] + np.exp(X1[:,3]) + 2)
    Y = ndm.rand(n) 
    T = np.maximum((- 21 * np.log(Y) * np.exp(-f_X)) ** (2/3) - 0.001, 1e-4)
    C = np.minimum(ndm.exponential(2, n), tau)
    De = (T <= C)
    T_O = np.minimum(T, C)
    Z1 = Z1.reshape(n, 1)
    Z2 = Z2.reshape(n, 1)
    X = np.hstack((Z1, Z2, X1))
    return {
        'Z1': np.array(Z1, dtype='float32'),
        'Z2': np.array(Z2, dtype='float32'),
        'X1': np.array(X1, dtype='float32'),
        'X': np.array(X, dtype='float32'),
        'T': np.array(T, dtype='float32'),
        'T_O': np.array(T_O, dtype='float32'),
        'De': np.array(De, dtype='float32'),
        'f_X': np.array(f_X, dtype='float32')
    }