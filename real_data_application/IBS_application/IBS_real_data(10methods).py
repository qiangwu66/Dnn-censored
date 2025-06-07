# python 3.12.8
#%% ----------------------
import numpy as np
import random
import torch

from B_spline import *
import pandas as pd
import os
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d

import torchtuples as tt
from pycox.models import CoxCC

from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime

from pycox.models import CoxPH

from pycox.models import DeepHitSingle

from Survival_methods.CoxPH_iteration import Surv_CoxPH
from Survival_methods.AH_iteration import Surv_AH

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv 

from lifelines import LogNormalAFTFitter

from Survival_methods.SAFT import SAFT_C_est

import time

results_folder = "results_folder"
os.makedirs(results_folder, exist_ok=True)

#%% ----------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)   

set_seed(1)

#%% Data Processing
df = pd.read_csv('support_clean_death_as_event_all_x.csv') # read csv file

X = np.array(df[['male', 'age', 'sps', 'scoma', 'disease1', 'disease2', 'disease3']], dtype='float32')
De = np.array(df['delta'], dtype='float32')
Time = np.array(df['d.time'], dtype='float32')
Time = Time / np.max(Time)

X[:,1] = X[:,1] / np.max(X[:,1])
X[:,2] = X[:,2] / np.max(X[:,2])
X[:,3] = X[:,3] / np.max(X[:,3])

tau = 1
# random order
A = np.arange(len(Time))
np.random.shuffle(A)
X_R = X[A]
De_R = De[A]
Time_R = Time[A]

# -------training data: 64%  validation data: 16%  test data: 20%---------------------------------
# ---training data 5828
X_R_train = X_R[np.arange(5828)]
De_R_train = De_R[np.arange(5828)]
Time_R_train = Time_R[np.arange(5828)]
# ---validation data 1456
X_R_valid = X_R[np.arange(5828, 7284)]
De_R_valid = De_R[np.arange(5828, 7284)]
Time_R_valid = Time_R[np.arange(5828, 7284)]
# ---test data 1820
X_R_test = X_R[np.arange(7284, 9104)]
De_R_test = De_R[np.arange(7284, 9104)]
Time_R_test = Time_R[np.arange(7284, 9104)]


#%%
def Indicator_matrix(a, b):
    a = np.array(a)
    b = np.array(b)
    I_M = (a[:, np.newaxis] >= b).astype(int)
    return I_M

def check_matrix_or_vector(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 2 or (value.ndim == 1 and value.size > 1):
            return True
    return False


#%% Kaplan-Meier
kmf = KaplanMeierFitter()
kmf.fit(Time_R_test, event_observed = 1 - De_R_test)
G_T_i = np.maximum(kmf.predict(Time_R_test).values, np.min(kmf.predict(Time_R_test).values[kmf.predict(Time_R_test).values > 0])) 

# G_s_k 
original_times = kmf.survival_function_.index.values[1:]  
original_survival_probs = kmf.survival_function_["KM_estimate"].values[1:]  

s_k = np.linspace(Time_R_test.min(), Time_R_test.max(), 100, endpoint=False)
s_k = np.array([float(f"{x:.4g}") for x in s_k]) 
s_k = np.maximum(s_k, 1e-4)

interpolator = interp1d(original_times, original_survival_probs, kind="previous", fill_value="extrapolate")
G_s_k = np.maximum(interpolator(s_k), np.min(interpolator(s_k)[interpolator(s_k) > 0]))


I_T_i_s_k = Indicator_matrix(s_k, Time_R_test)
I_T_i_s_k_D_1 = Indicator_matrix(s_k, Time_R_test) * np.tile(De_R_test, (len(s_k), 1)) 

t_nodes = np.array(np.linspace(0, 1, 101), dtype="float32")
t_nodes = np.maximum(t_nodes, 1e-4)



# ----------------Cox Linear-------------------------
start_time_CoxLinear = time.time()

columns = [f'feature_{i+1}' for i in range(X_R_train.shape[1])]
data = pd.DataFrame(X_R_train, columns=columns)
data['duration'] = np.maximum(Time_R_train, 1e-4)
data['event'] = De_R_train

cph = CoxPHFitter()
cph.fit(data, duration_col='duration', event_col='event')

# IBS_CoxLinear
CoxPH_IBS = cph.predict_survival_function(X_R_test, times=s_k)
IBS_CoxLinear = np.nanmean(CoxPH_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - CoxPH_IBS) ** 2 * (1 -I_T_i_s_k) / np.tile(G_s_k, (len(Time_R_test), 1)).T)

end_time_CoxLinear = time.time()
run_time_CoxLinear = end_time_CoxLinear - start_time_CoxLinear


# ---------------CoxPH(Full)---------------------------
start_time_CoxPH = time.time()

m_CoxPH = 5
nodevec_CoxPH = np.array(np.linspace(0, tau, m_CoxPH + 2), dtype="float32")

# IBS_CoxPH(full)
S_CoxPH = Surv_CoxPH(X_R_train, Time_R_train, De_R_train, X_R_test, t_nodes, m_CoxPH, nodevec_CoxPH, tau, s_k)
S_t_X_CoxPH_IBS = S_CoxPH["S_t_X_CoxPH_IBS"] # 100*200
IBS_CoxPH = np.nanmean(S_t_X_CoxPH_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - S_t_X_CoxPH_IBS) ** 2 * (1 - I_T_i_s_k) / np.tile(G_s_k, (len(Time_R_test), 1)).T)

end_time_CoxPH = time.time()
run_time_CoxPH = end_time_CoxPH - start_time_CoxPH



# ---------------Cox-CC---------------------------
start_time_CoxCC = time.time()

y_train = (Time_R_train, De_R_train)
y_val = (Time_R_valid, De_R_valid)
val = tt.tuplefy(X_R_valid, y_val)
in_features = X_R_train.shape[1]
num_nodes = [32, 32]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False
net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                          dropout, output_bias=output_bias)
model = CoxCC(net, tt.optim.Adam)
batch_size = 256
model.optimizer.set_lr(0.01)
epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True
log = model.fit(X_R_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val.repeat(10).cat())

model.partial_log_likelihood(*val).mean()
_ = model.compute_baseline_hazards()
surv = model.predict_surv_df(X_R_test)

# IBS_Cox-CC
Survial_CoxCC_IBS = surv.apply(lambda col: np.interp(s_k, surv.index, col), axis=0).to_numpy()
IBS_CoxCC = np.nanmean(Survial_CoxCC_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - Survial_CoxCC_IBS) ** 2 *(1 - I_T_i_s_k) / np.tile(G_s_k, (len(Time_R_test), 1)).T)

end_time_CoxCC = time.time()
run_time_CoxCC = end_time_CoxCC - start_time_CoxCC



# ---------------Cox-Time---------------------------
start_time_CoxTime = time.time()

labtrans = CoxTime.label_transform()
y_train_CoxTime = labtrans.fit_transform(*y_train)
y_val_CoxTime = labtrans.transform(*y_val)
val_CoxTime = tt.tuplefy(X_R_valid, y_val_CoxTime)
in_features = X_R_train.shape[1]
num_nodes = [32, 32]
out_features = 1
batch_norm = True
dropout = 0.1
net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)
batch_size = 256
model.optimizer.set_lr(0.01)
epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True
log = model.fit(X_R_train, y_train_CoxTime, batch_size, epochs, callbacks, verbose,
                val_data=val_CoxTime.repeat(10).cat())
model.partial_log_likelihood(*val_CoxTime).mean()
_ = model.compute_baseline_hazards()
surv_CoxTime = model.predict_surv_df(X_R_test) 

# IBS_Cox-Time
Survial_CoxTime_IBS = surv_CoxTime.apply(lambda col: np.interp(s_k, surv_CoxTime.index, col), axis=0).to_numpy() 
IBS_CoxTime = np.nanmean(Survial_CoxTime_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - Survial_CoxTime_IBS)** 2 * (1 - I_T_i_s_k) / np.tile(G_s_k, (len(Time_R_test), 1)).T)

end_time_CoxTime = time.time()
run_time_CoxTime = end_time_CoxTime - start_time_CoxTime


# ---------------DeepSurv---------------------------
start_time_DeepSurv = time.time()

in_features = X_R_train.shape[1]
num_nodes = [32, 32]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False
net = tt.practical.MLPVanilla(in_features, num_nodes, out_features,
                              batch_norm,dropout, output_bias=output_bias)
model = CoxPH(net, tt.optim.Adam)
batch_size = 256
model.optimizer.set_lr(0.01)
epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True
log = model.fit(X_R_train, y_train, batch_size, epochs, callbacks, verbose, 
                val_data=val, val_batch_size=batch_size)
model.partial_log_likelihood(*val).mean()
_ = model.compute_baseline_hazards()
surv_DeepSurv = model.predict_surv_df(X_R_test) 

# IBS_DeepSurv
Survial_DeepSurv_IBS = surv_DeepSurv.apply(lambda col: np.interp(s_k, surv_DeepSurv.index, col), axis=0).to_numpy() 
IBS_DeepSurv = np.nanmean(Survial_DeepSurv_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 -Survial_DeepSurv_IBS) ** 2 * (1 - I_T_i_s_k) / np.tile(G_s_k, (len(Time_R_test), 1)).T)

end_time_DeepSurv = time.time()
run_time_DeepSurv = end_time_DeepSurv - start_time_DeepSurv



# ---------------DeepHit---------------------------
start_time_DeepHit = time.time()

num_durations = 10
labtrans = DeepHitSingle.label_transform(num_durations)
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train_DeepHit = labtrans.fit_transform(*y_train)
y_val_DeepHit = labtrans.transform(*y_val)
val_DeepHit = tt.tuplefy(X_R_valid, y_val_DeepHit)
in_features = X_R_train.shape[1]
num_nodes = [32, 32]
out_features = labtrans.out_features
batch_norm = True
dropout = 0.1
net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
batch_size = 256
model.optimizer.set_lr(0.01)
epochs = 100
callbacks = [tt.callbacks.EarlyStopping()]
log = model.fit(X_R_train,  y_train_DeepHit, batch_size, epochs, callbacks, val_data=val_DeepHit)
surv_DeepHit = model.predict_surv_df(X_R_test)

# IBS_DeepHit
Survial_DeepHit_IBS = surv_DeepHit.apply(lambda col: np.interp(s_k, surv_DeepHit.index, col), axis=0).to_numpy()
IBS_DeepHit = np.nanmean(Survial_DeepHit_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - Survial_DeepHit_IBS)** 2 * (1 - I_T_i_s_k) / np.tile(G_s_k, (len(Time_R_test), 1)).T)

end_time_DeepHit = time.time()
run_time_DeepHit = end_time_DeepHit - start_time_DeepHit




# ---------------RSF---------------------------
start_time_RSF = time.time()

columns = [f'feature_{i+1}' for i in range(X_R_train.shape[1])] 
X_train_DF = pd.DataFrame(X_R_train, columns=columns)
X_R_test_DF = pd.DataFrame(X_R_test, columns=columns)
y_train_DF = pd.DataFrame({
"event": De_R_train.astype(bool),
"time": Time_R_train
})
y_test_DF = pd.DataFrame({
"event": De_R_test.astype(bool),
"time": Time_R_test
})

y_train_DF = Surv.from_dataframe("event", "time", y_train_DF)
y_test_DF = Surv.from_dataframe("event", "time", y_test_DF)

rsf = RandomSurvivalForest(
    n_estimators=1000, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=20)
rsf.fit(X_train_DF, y_train_DF)

surv_original = rsf.predict_survival_function(X_R_test_DF, return_array=True)

# IBS_RSF
Survial_RSF_IBS = np.zeros((len(s_k), len(X_R_test))) 
for i, surv_func in enumerate(surv_original):
    interp_func = interp1d(rsf.unique_times_, surv_func, kind="previous", bounds_error= False, fill_value=(1.0, 0.0))
    Survial_RSF_IBS[:, i] = interp_func(s_k)
IBS_RSF = np.nanmean(Survial_RSF_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - Survial_RSF_IBS) ** 2 * (1 -I_T_i_s_k) / np.tile(G_s_k, (len(Time_R_test), 1)).T)

end_time_RSF = time.time()
run_time_RSF = end_time_RSF - start_time_RSF



# ----------------PAFT-------------------------
start_time_PAFT = time.time()

columns = [f'feature_{i+1}' for i in range(X_R_train.shape[1])] 
data = pd.DataFrame(X_R_train, columns=columns)
data['duration'] = np.maximum(Time_R_train, 1e-4)
data['event'] = De_R_train
new_data = pd.DataFrame(X_R_test, columns=[f'feature_{i+1}' for i in range(X_R_test.shape[1])])

aft_model = LogNormalAFTFitter()
aft_model.fit(data, duration_col='duration', event_col='event')

# IBS_PAFT
S_t_X_PAFT_IBS = aft_model.predict_survival_function(new_data, times=s_k) 
IBS_PAFT = np.nanmean(S_t_X_PAFT_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - S_t_X_PAFT_IBS) ** 2 * (1 -I_T_i_s_k) / np.tile(G_s_k, (len(Time_R_test), 1)).T)

end_time_PAFT = time.time()
run_time_PAFT = end_time_PAFT - start_time_PAFT



# ----------------SAFT-------------------------
start_time_SAFT = time.time()

beta_coefs = cph.summary[['coef']].values[:,0]
Omega_b = np.max(np.concatenate([Time_R_train, Time_R_test])) * np.exp(np.max(np.concatenate([X_R_train @beta_coefs, X_R_test @ beta_coefs])))

nodes_num = 5
node_vec = np.array(np.linspace(0, Omega_b, nodes_num + 2), dtype="float32")  
t_x_nodes = np.array(np.linspace(0, Omega_b, 501), dtype="float32")  
beta_X_train = np.array(X_R_train @ beta_coefs, dtype='float32')  
Y_beta_X_train = np.array(Time_R_train * np.exp(beta_X_train), dtype="float32") 
I_t_x_nodes_Y_X_train = Indicator_matrix(Y_beta_X_train, t_x_nodes)
c_saft = SAFT_C_est(De_R_train, t_x_nodes, beta_X_train, Y_beta_X_train, I_t_x_nodes_Y_X_train, nodes_num, node_vec,Omega_b)
beta_X_R_test = np.array(X_R_test @ beta_coefs, dtype='float32')

# IBS_SAFT
S_t_X_SAFT_IBS = []
for t in s_k:
    t_beta_X_R_test = np.array(t * np.exp(beta_X_R_test), dtype="float32") 
    S_t_X_SAFT_IBS.append(np.exp(- (Omega_b/len(t_x_nodes)) * Indicator_matrix(t_beta_X_R_test, t_x_nodes) @ np.exp(B_S(nodes_num, t_x_nodes, node_vec) @ c_saft)))
S_t_X_SAFT_IBS = np.array(S_t_X_SAFT_IBS) # 100*200
IBS_SAFT = np.nanmean(S_t_X_SAFT_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - S_t_X_SAFT_IBS) ** 2 * (1 -I_T_i_s_k) / np.tile(G_s_k, (len(Time_R_test), 1)).T)

end_time_SAFT = time.time()
run_time_SAFT = end_time_SAFT - start_time_SAFT



# ---------------AH---------------------------
start_time_AH = time.time()

m_AH = 5
nodevec_AH = np.array(np.linspace(0, tau, m_AH + 2), dtype="float32")
S_AH = Surv_AH(X_R_train, Time_R_train, De_R_train, X_R_test, t_nodes, m_AH, nodevec_AH, tau, s_k)

# IBS_AH
S_t_X_AH_IBS = S_AH["S_t_X_AH_IBS"]
IBS_AH = np.nanmean(S_t_X_AH_IBS ** 2 * I_T_i_s_k_D_1 / np.tile(G_T_i, (len(s_k), 1)) + (1 - S_t_X_AH_IBS) ** 2 * (1 -I_T_i_s_k) / np.tile(G_s_k, (len(Time_R_test), 1)).T)


end_time_AH = time.time()
run_time_AH = end_time_AH - start_time_AH

# =================IBS-Table=======================
output_folder = "IBS_12methods"
os.makedirs(output_folder, exist_ok=True)


dic_IBS = {
    "IBS_Cox_Linear": np.array([IBS_CoxLinear, run_time_CoxLinear]),
    "IBS_CoxPH(Full)": np.array([IBS_CoxPH, run_time_CoxPH]),
    "IBS_Cox_CC": np.array([IBS_CoxCC, run_time_CoxCC]),
    "IBS_Cox_Time": np.array([IBS_CoxTime, run_time_CoxTime]),
    "IBS_DeepSurv": np.array([IBS_DeepSurv, run_time_DeepSurv]),
    "IBS_DeepHit": np.array([IBS_DeepHit, run_time_DeepHit]),
    "IBS_RSF": np.array([IBS_RSF, run_time_RSF]),
    "IBS_PAFT": np.array([IBS_PAFT, run_time_PAFT]),
    "IBS_SAFT": np.array([IBS_SAFT, run_time_SAFT]),
    "IBS_AH": np.array([IBS_AH, run_time_AH]),
}
result_IBS = pd.DataFrame(dic_IBS)
ibs_path = os.path.join(output_folder, "IBS_realdata_10methods.csv")
result_IBS.to_csv(ibs_path, index=False)

