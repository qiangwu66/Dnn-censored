import numpy as np
import scipy.optimize as spo
from B_spline import B_S

def Indicator_matrix(a, b):
    a = np.array(a)
    b = np.array(b)
    I_M = (a[:, np.newaxis] >= b).astype(int)
    return I_M


import numpy as np
import scipy.optimize as spo

def Beta_t(m, ts, nodevec, coefs, d):
    """
    计算在时间点 ts 上，每个协变量的时间变系数 beta_j(t)。
    返回形状: (len(ts), d)
    - m: 样条阶数相关参数（与你的 B_S 一致）
    - ts: 时间点数组，形如 (T,)
    - nodevec: 样条结点
    - coefs: 展平的一维参数，长度应为 d*(m+4)
    - d: 协变量维数
    """
    B = B_S(m, ts, nodevec)              # 形状 (T, m+4)
    k = m + 4
    # 将 coefs 重塑为 (d, k)，每行对应一个协变量的样条系数
    C = np.reshape(coefs[:d * k], (d, k))
    # 对每个协变量 j: beta_j(t) = B @ C[j]
    # 合并向量化：Beta_ts = B @ C^T -> 形状 (T, d)
    Beta_ts = B @ C.T
    return Beta_ts  # (T, d)

def Surv_pred(m, ts, nodevec, coefs, t_nodes, X_test, tau, d):
    """
    返回在时间网格 ts 上、对测试集每个样本的生存概率 S(t|X)。
    - coefs: 长度为 (d+1)*(m+4)，前 d*k 为 beta(t) 的系数，最后 k 为 lambda(t) 的系数
    """
    k = m + 4
    # beta(t_nodes): (len(t_nodes), d)
    Beta_t_nodes = Beta_t(m, t_nodes, nodevec, coefs[0:d * k], d)
    # 线性预测项：BX(t_nodes) = X_test @ beta(t_nodes)^T -> 形状 (n_test, len(t_nodes))
    BX_t_nodes = X_test @ Beta_t_nodes.T

    # 基准风险 λ(t_nodes): (len(t_nodes),)
    lambda_t_nodes = B_S(m, t_nodes, nodevec) @ coefs[d * k:(d + 1) * k]

    # 积分近似：S(t|X) = exp( - sum_{u<=t} λ(u)*exp(BX(u)) Δu )
    # 你的实现用 Indicator_matrix(ts, t_nodes) 选择 u<=t，并用对角矩阵乘 λ，再乘 exp(BX)
    # 为了向量化，避免巨型对角阵，可逐步等价改写：
    # step 1: W = exp(BX_t_nodes) -> (n_test, n_nodes)
    W = np.exp(BX_t_nodes)

    # step 2: 对每个样本，元素乘 λ： (n_test, n_nodes)
    WL = W * lambda_t_nodes  # 广播

    # step 3: 计算到每个 ts 的累积和。先构造 I(ts, t_nodes): (len(ts), len(t_nodes))
    I = Indicator_matrix(ts, t_nodes).astype(float)
    # H(ts, X) = I @ WL^T -> (len(ts), n_test)
    H = I @ WL.T

    # 步长 Δu
    delta = tau / len(t_nodes)
    S_t_X = np.exp(- H * delta)  # (len(ts), n_test)
    return S_t_X

def Est_Coxvarying(X_train, Y_train, De_train, t_nodes, m, nodevec, tau):
    """
    估计通用维度 d 的时间变系数与基准风险样条系数。
    返回展平的参数 coefs，长度 (d+1)*(m+4)
    """
    n, d = X_train.shape
    k = m + 4

    B_Y = B_S(m, Y_train, nodevec)          # (n, k)
    B_nodes = B_S(m, t_nodes, nodevec)      # (n_nodes, k)
    I_Y_nodes = Indicator_matrix(Y_train, t_nodes)  # (n, n_nodes)
    n_nodes = len(t_nodes)
    delta = tau / n_nodes

    def CF(c):
        # 解析参数
        beta_coef = c[:d * k].reshape(d, k)          # (d, k)
        lambda_coef = c[d * k:(d + 1) * k]           # (k,)

        # 1) 事件时刻的线性项 BX_Y
        # Beta_Y: (n, d) = B_Y @ beta_coef^T
        Beta_Y = B_Y @ beta_coef.T
        BX_Y = np.sum(Beta_Y * X_train, axis=1)      # (n,)

        # 2) t_nodes 上的线性项，每个样本对每个节点
        # Beta_t_nodes: (n_nodes, d) = B_nodes @ beta_coef^T
        Beta_t_nodes = B_nodes @ beta_coef.T         # (n_nodes, d)
        # BX_t_nodes: (n, n_nodes) = X_train @ Beta_t_nodes^T
        BX_t_nodes = X_train @ Beta_t_nodes.T

        # 3) 基准风险
        lambda_t_nodes = B_nodes @ lambda_coef       # (n_nodes,)
        lambda_Y = B_Y @ lambda_coef                 # (n,)

        # 4) Cox 偏对数似然的离散化近似（与你原 Loss 形式一致）
        # Loss = - mean[ De*(log λ(Y)+BX_Y) - (I(Y, t_nodes) * exp(BX)) @ λ(t_nodes) * Δ ]
        term1 = De_train * (np.log(lambda_Y + 1e-8) + BX_Y)             # (n,)
        term2 = (I_Y_nodes * np.exp(BX_t_nodes)) @ lambda_t_nodes       # (n,)
        Loss = - np.mean(term1 - term2 * delta)
        return Loss

    # 初始值与边界
    initial_c = 0.1 * np.ones((d + 1) * k)
    # beta 系数无界，lambda 系数 >= 0
    bounds = [(None, None)] * (d * k) + [(0, None)] * k

    result = spo.minimize(CF, initial_c, method='SLSQP', bounds=bounds)
    return result['x']  # 长度 (d+1)*k

def Surv_Coxvarying(X_train, Y_train, De_train, X_test, t_nodes, m, nodevec, tau, s_k):
    """
    训练并在时间网格 s_k 上预测生存函数。
    返回 dict: {"S_t_X_Coxvary_IBS": S(ts | X_test)} 形状 (len(s_k), n_test)
    """
    d = X_train.shape[1]
    coefs = Est_Coxvarying(X_train, Y_train, De_train, t_nodes, m, nodevec, tau)
    S_t_X_Coxvary_IBS = Surv_pred(m, s_k, nodevec, coefs, t_nodes, X_test, tau, d)
    return {"S_t_X_Coxvary_IBS": S_t_X_Coxvary_IBS}