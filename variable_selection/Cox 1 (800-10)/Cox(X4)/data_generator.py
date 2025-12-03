#%% ----------------
import numpy as np
import numpy.random as ndm

def generate_Cox_1(n, rho, tau):
    def ar1_cov(d, rho, sigma2):
        idx = np.arange(d)
        D = np.abs(idx[:, None] - idx[None, :])
        return (sigma2 * (rho ** D)).astype(float)

    d = 10
    sigma2 = 1.0
    mu = np.zeros(d)

    Sigma = ar1_cov(d, rho, sigma2)

    X1 = np.random.multivariate_normal(mean=mu, cov=Sigma, size=n)
    X = np.clip(X1, -3, 3)

    f_X = 1.5 * (X[:,0] + X[:,1] + X[:,2] + X[:,3] + X[:,4])
    Y = ndm.rand(n)
    T = np.maximum((- np.log(Y) * np.exp(-f_X)) ** (2/3) - 0.001, 1e-4)
    C = np.minimum(ndm.exponential(2, n), tau)
    De = (T <= C)
    T_O = np.minimum(T, C)

    return {
        'X': np.array(X, dtype='float32'),
        'T': np.array(T, dtype='float32'),
        'T_O': np.array(T_O, dtype='float32'),
        'De': np.array(De, dtype='float32')
    }

