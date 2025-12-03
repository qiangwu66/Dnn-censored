# python 3.12.8
#%% ----------- 导入包 -----------
import numpy as np
import random
import torch
from data_generator import generate_Cox_1, generate_Cox_2_nocross
from Survival_methods.DNN_iteration_g1 import g1_dnn
from Survival_methods.DNN_iteration_g2 import g2_dnn
import pandas as pd
import os
from joblib import Parallel, delayed  # 引入并行工具

# 确保目标文件夹 "result_data" 存在
results_folder = "results_folder"
os.makedirs(results_folder, exist_ok=True)

#%% ---------- 设置种子 -------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)  # 为CPU设置随机种子

set_seed(1)



#%% 两个向量逐个元素比较大小，返回0-1矩阵 len(a)*len(b)
# def Indicator_matrix(a, b):
#     a = np.array(a)
#     b = np.array(b)
#     I_M = (a[:, np.newaxis] >= b).astype(int)
#     return I_M

def check_matrix_or_vector(value):
    # 检查是否是 NumPy 数组
    if isinstance(value, np.ndarray):
        # 检查是否是矩阵（2D）或向量（1D）且长度大于1
        if value.ndim == 2 or (value.ndim == 1 and value.size > 1):
            return True
    return False


#%% =========== 定义并行任务函数 =============
def single_simulation(b, n, Dnn_layer1, Dnn_node1, Dnn_lr1, Dnn_layer2, Dnn_node2, Dnn_lr2, Dnn_epoch, patiences, tau, corr, c1):
    print(f'-------------n={n}, b={b}--------------')
    set_seed(500 + b)
    # --------------g1(t,x)数据生成-------------
    Data1_all = generate_Cox_1(n, corr, tau) 
    
    train_data1 = {
        key: (value[:int(0.8 * n)] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data1_all.items()
        }
    val_data1 = {
        key: (value[-int(0.2 * n):] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data1_all.items()
        }
    

    # --------------g2(t,x)数据生成-------------
    Data2_all = generate_Cox_2_nocross(n, corr, tau, c1) 
    
    train_data2 = {
        key: (value[:int(0.8 * n)] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data2_all.items()
        }
    val_data2 = {
        key: (value[-int(0.2 * n):] if check_matrix_or_vector(value) 
        else value)
        for key, value in Data2_all.items()
        }


    # 合并所有的训练数据
    polled_data = {
        key: np.concatenate((train_data1[key], train_data2[key]), axis=0) if train_data1[key].ndim > 1 else np.concatenate((train_data1[key], train_data2[key]))
        for key in train_data1
    }

    # ----------------先训练g2(t,X)----------------
    Est_dnn_g2 = g2_dnn(polled_data, train_data2, val_data2, tau, Dnn_layer2, Dnn_node2, Dnn_lr2, Dnn_epoch, patiences)
    g2_T_X_n = Est_dnn_g2['g2_T_X_n']
    sigma2_ws = Est_dnn_g2['sigma2_ws']

    # -----------------再训练g1(t,X)----------------------
    Est_dnn_g1 = g1_dnn(polled_data, train_data1, val_data1, tau, Dnn_layer1, Dnn_node1, Dnn_lr1, Dnn_epoch, patiences, g2_T_X_n, sigma2_ws)

    # 返回结果
    return {
        'U_tau_w1': Est_dnn_g1['U_tau_w1'],
        'U_tau_w2': Est_dnn_g1['U_tau_w2'],
        'U_tau_w3': Est_dnn_g1['U_tau_w3'],
        'U_tau_w4': Est_dnn_g1['U_tau_w4'],
    }

#%% =========== 并行模拟 ===========
results = []  # 用于存储 n= 400 的结果
n_jobs = 1 # 设置并行核数
B = 500 # 模拟次数 


#%% ----------- 设置参数 ------------
tau = 2 # 研究终止时间
corr = 0.5  # 相关系数
n = 800
Dnn_layer1 = 2
Dnn_node1 = 40
Dnn_lr1 = 4e-4

Dnn_layer2 = 2
Dnn_node2 = 40
Dnn_lr2 = 4e-4

Dnn_epoch = 1000
patiences = 10
c1 = 0.25 # c1 = 0, 0.125, 0.25, 0.5

results = Parallel(n_jobs=n_jobs)(
            delayed(single_simulation)(b, n, Dnn_layer1, Dnn_node1, Dnn_lr1, Dnn_layer2, Dnn_node2, Dnn_lr2, Dnn_epoch, patiences, tau, corr, c1) 
            for b in range(B)
        )

#%% =========== 后处理 ===========
def process_results(results):
    U_tau_w1 = []
    U_tau_w2 = []
    U_tau_w3 = []
    U_tau_w4 = []
    

    for res in results:
        U_tau_w1.append(res["U_tau_w1"])
        U_tau_w2.append(res["U_tau_w2"])
        U_tau_w3.append(res["U_tau_w3"])
        U_tau_w4.append(res["U_tau_w4"])


    # 转换为 numpy 数组
    U_tau_w1 = np.array(U_tau_w1)
    U_tau_w2 = np.array(U_tau_w2)
    U_tau_w3 = np.array(U_tau_w3)
    U_tau_w4 = np.array(U_tau_w4)

    # 保存文件到 result_data 文件夹
    np.save(os.path.join(results_folder, 'U_tau_w1.npy'), U_tau_w1)
    np.save(os.path.join(results_folder, 'U_tau_w2.npy'), U_tau_w2)
    np.save(os.path.join(results_folder, 'U_tau_w3.npy'), U_tau_w3)
    np.save(os.path.join(results_folder, 'U_tau_w4.npy'), U_tau_w4)


# 处理 n=400 的结果
process_results(results)


#%% 运行结果
import numpy as np
import os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


#%% 加载模拟结果
# 文件夹路径
results_folder = "results_folder"

# 文件名列表（包含 DMS、IBS、St 和其他文件）
file_names = [
     'U_tau_w1.npy',
     'U_tau_w2.npy',
     'U_tau_w3.npy',
     'U_tau_w4.npy',
]

# 遍历文件名列表并加载文件
data = {}
for file_name in file_names:
    file_path = os.path.join(results_folder, file_name)  # 拼接完整路径
    data[file_name] = np.load(file_path)  # 加载 .npy 文件

# 将数据分配到变量中
U_tau_w1_B = data['U_tau_w1.npy'] # B
U_tau_w2_B = data['U_tau_w2.npy'] # B
U_tau_w3_B = data['U_tau_w3.npy'] # B
U_tau_w4_B = data['U_tau_w4.npy'] # B

#%% 计算size/power结果
size1_Cox = np.mean(abs(U_tau_w1_B) > 1.96)
size2_Cox = np.mean(abs(U_tau_w2_B) > 1.96)
size3_Cox = np.mean(abs(U_tau_w3_B) > 1.96)
size4_Cox = np.mean(abs(U_tau_w4_B) > 1.96)


# =================tables=======================
# 确保目标文件夹 "tables" 存在
output_folder = "Individual_DNN"
os.makedirs(output_folder, exist_ok=True)

# size1 是一个NumPy向量
result_size_DNN = pd.DataFrame(
    np.array([size1_Cox, size2_Cox, size3_Cox, size4_Cox])
    ) # 将矩阵转换为 DataFrame
# 保存为 CSV 文件
size_DNN_path = os.path.join(output_folder, f"size_DNN-{Dnn_layer1}-{Dnn_node1}-{Dnn_lr1}-{Dnn_layer2}-{Dnn_node2}-{Dnn_lr2}-c1-{c1}.csv")
result_size_DNN.to_csv(size_DNN_path, index=False, header=False)  # 不保存索引和列名
print(f"size_DNN 数据已保存到 {size_DNN_path}")
