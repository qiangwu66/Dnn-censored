
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

Set_n = np.array([400, 800]) 

results_folder = "results_folder"


file_names = [

    "n400_DMS_Cox_Linear.npy",
    "n800_DMS_Cox_Linear.npy",
    "n400_DMS_CoxPH.npy",
    "n800_DMS_CoxPH.npy",
    "n400_DMS_DNN.npy",
    "n800_DMS_DNN.npy",
    "n400_DMS_Cox_CC.npy",
    "n800_DMS_Cox_CC.npy",
    "n400_DMS_Cox_Time.npy",
    "n800_DMS_Cox_Time.npy",
    "n400_DMS_DeepSurv.npy",
    "n800_DMS_DeepSurv.npy",
    "n400_DMS_DeepHit.npy",
    "n800_DMS_DeepHit.npy",
    "n400_DMS_AH.npy",
    "n800_DMS_AH.npy",
    "n400_DMS_RSF.npy",
    "n800_DMS_RSF.npy",
    "n400_DMS_PAFT.npy",
    "n800_DMS_PAFT.npy",
    "n400_DMS_SAFT.npy",
    "n800_DMS_SAFT.npy",
    "n400_DMS_Coxvary.npy",
    "n800_DMS_Coxvary.npy",

    "n400_IBS_Cox_Linear.npy",
    "n800_IBS_Cox_Linear.npy",
    "n400_IBS_CoxPH.npy",
    "n800_IBS_CoxPH.npy",
    "n400_IBS_DNN.npy",
    "n800_IBS_DNN.npy",
    "n400_IBS_Cox_CC.npy",
    "n800_IBS_Cox_CC.npy",
    "n400_IBS_Cox_Time.npy",
    "n800_IBS_Cox_Time.npy",
    "n400_IBS_DeepSurv.npy",
    "n800_IBS_DeepSurv.npy",
    "n400_IBS_DeepHit.npy",
    "n800_IBS_DeepHit.npy",
    "n400_IBS_AH.npy",
    "n800_IBS_AH.npy",
    "n400_IBS_RSF.npy",
    "n800_IBS_RSF.npy",
    "n400_IBS_PAFT.npy",
    "n800_IBS_PAFT.npy",
    "n400_IBS_SAFT.npy",
    "n800_IBS_SAFT.npy",
    "n400_IBS_Coxvary.npy",
    "n800_IBS_Coxvary.npy",
   
    "n400_run_time_Cox_Linear.npy",
    "n800_run_time_Cox_Linear.npy",
    "n400_run_time_CoxPH.npy",
    "n800_run_time_CoxPH.npy",
    "n400_run_time_DNN.npy",
    "n800_run_time_DNN.npy",
    "n400_run_time_Cox_CC.npy",
    "n800_run_time_Cox_CC.npy",
    "n400_run_time_Cox_Time.npy",
    "n800_run_time_Cox_Time.npy",
    "n400_run_time_DeepSurv.npy",
    "n800_run_time_DeepSurv.npy",
    "n400_run_time_DeepHit.npy",
    "n800_run_time_DeepHit.npy",
    "n400_run_time_AH.npy",
    "n800_run_time_AH.npy",
    "n400_run_time_RSF.npy",
    "n800_run_time_RSF.npy",
    "n400_run_time_PAFT.npy",
    "n800_run_time_PAFT.npy",
    "n400_run_time_SAFT.npy",
    "n800_run_time_SAFT.npy",
    "n400_run_time_Coxvary.npy",
    "n800_run_time_Coxvary.npy",

    "n400_S_t_X_Cox_Linear.npy",
    "n800_S_t_X_Cox_Linear.npy",
    "n400_S_t_X_CoxPH.npy",
    "n800_S_t_X_CoxPH.npy",
    "n400_S_t_X_CoxPH.npy",
    "n800_S_t_X_CoxPH.npy",
    "n400_S_t_X_DNN.npy",
    "n800_S_t_X_DNN.npy",
    "n400_S_t_X_Cox_CC.npy",
    "n800_S_t_X_Cox_CC.npy",
    "n400_S_t_X_Cox_Time.npy",
    "n800_S_t_X_Cox_Time.npy",
    "n400_S_t_X_DeepSurv.npy",
    "n800_S_t_X_DeepSurv.npy",
    "n400_S_t_X_DeepHit.npy",
    "n800_S_t_X_DeepHit.npy",
    "n400_S_t_X_AH.npy",
    "n800_S_t_X_AH.npy",
    "n400_S_t_X_RSF.npy",
    "n800_S_t_X_RSF.npy",
    "n400_S_t_X_PAFT.npy",
    "n800_S_t_X_PAFT.npy",
    "n400_S_t_X_SAFT.npy",
    "n800_S_t_X_SAFT.npy",
    "n400_S_t_X_Coxvary.npy",
    "n800_S_t_X_Coxvary.npy",

    "St_true_X_fig.npy",
    "t_fig.npy",
]


data = {}
for file_name in file_names:
    file_path = os.path.join(results_folder, file_name) 
    data[file_name] = np.load(file_path) 


DMS_Cox_Linear_400_B = data["n400_DMS_Cox_Linear.npy"]
DMS_Cox_Linear_800_B = data["n800_DMS_Cox_Linear.npy"]
DMS_CoxPH_400_B = data["n400_DMS_CoxPH.npy"]
DMS_CoxPH_800_B = data["n800_DMS_CoxPH.npy"]
DMS_DNN_400_B = data["n400_DMS_DNN.npy"]
DMS_DNN_800_B = data["n800_DMS_DNN.npy"]
DMS_Cox_CC_400_B = data["n400_DMS_Cox_CC.npy"]
DMS_Cox_CC_800_B = data["n800_DMS_Cox_CC.npy"]
DMS_Cox_Time_400_B = data["n400_DMS_Cox_Time.npy"]
DMS_Cox_Time_800_B = data["n800_DMS_Cox_Time.npy"]
DMS_DeepSurv_400_B = data["n400_DMS_DeepSurv.npy"]
DMS_DeepSurv_800_B = data["n800_DMS_DeepSurv.npy"]
DMS_DeepHit_400_B = data["n400_DMS_DeepHit.npy"]
DMS_DeepHit_800_B = data["n800_DMS_DeepHit.npy"]
DMS_AH_400_B = data["n400_DMS_AH.npy"]
DMS_AH_800_B = data["n800_DMS_AH.npy"]
DMS_RSF_400_B = data["n400_DMS_RSF.npy"]
DMS_RSF_800_B = data["n800_DMS_RSF.npy"]
DMS_PAFT_400_B = data["n400_DMS_PAFT.npy"]
DMS_PAFT_800_B = data["n800_DMS_PAFT.npy"]
DMS_SAFT_400_B = data["n400_DMS_SAFT.npy"]
DMS_SAFT_800_B = data["n800_DMS_SAFT.npy"]
DMS_Coxvary_400_B = data["n400_DMS_Coxvary.npy"]
DMS_Coxvary_800_B = data["n800_DMS_Coxvary.npy"]


IBS_Cox_Linear_400_B = data["n400_IBS_Cox_Linear.npy"]
IBS_Cox_Linear_800_B = data["n800_IBS_Cox_Linear.npy"]
IBS_CoxPH_400_B = data["n400_IBS_CoxPH.npy"]
IBS_CoxPH_800_B = data["n800_IBS_CoxPH.npy"]
IBS_DNN_400_B = data["n400_IBS_DNN.npy"]
IBS_DNN_800_B = data["n800_IBS_DNN.npy"]
IBS_Cox_CC_400_B = data["n400_IBS_Cox_CC.npy"]
IBS_Cox_CC_800_B = data["n800_IBS_Cox_CC.npy"]
IBS_Cox_Time_400_B = data["n400_IBS_Cox_Time.npy"]
IBS_Cox_Time_800_B = data["n800_IBS_Cox_Time.npy"]
IBS_DeepSurv_400_B = data["n400_IBS_DeepSurv.npy"]
IBS_DeepSurv_800_B = data["n800_IBS_DeepSurv.npy"]
IBS_DeepHit_400_B = data["n400_IBS_DeepHit.npy"]
IBS_DeepHit_800_B = data["n800_IBS_DeepHit.npy"]
IBS_AH_400_B = data["n400_IBS_AH.npy"]
IBS_AH_800_B = data["n800_IBS_AH.npy"]
IBS_RSF_400_B = data["n400_IBS_RSF.npy"]
IBS_RSF_800_B = data["n800_IBS_RSF.npy"]
IBS_PAFT_400_B = data["n400_IBS_PAFT.npy"]
IBS_PAFT_800_B = data["n800_IBS_PAFT.npy"]
IBS_SAFT_400_B = data["n400_IBS_SAFT.npy"]
IBS_SAFT_800_B = data["n800_IBS_SAFT.npy"]
IBS_Coxvary_400_B = data["n400_IBS_Coxvary.npy"]
IBS_Coxvary_800_B = data["n800_IBS_Coxvary.npy"]


run_time_Cox_Linear_400_B = data["n400_run_time_Cox_Linear.npy"]
run_time_Cox_Linear_800_B = data["n800_run_time_Cox_Linear.npy"]
run_time_CoxPH_400_B = data["n400_run_time_CoxPH.npy"]
run_time_CoxPH_800_B = data["n800_run_time_CoxPH.npy"]
run_time_DNN_400_B = data["n400_run_time_DNN.npy"]
run_time_DNN_800_B = data["n800_run_time_DNN.npy"]
run_time_Cox_CC_400_B = data["n400_run_time_Cox_CC.npy"]
run_time_Cox_CC_800_B = data["n800_run_time_Cox_CC.npy"]
run_time_Cox_Time_400_B = data["n400_run_time_Cox_Time.npy"]
run_time_Cox_Time_800_B = data["n800_run_time_Cox_Time.npy"]
run_time_DeepSurv_400_B = data["n400_run_time_DeepSurv.npy"]
run_time_DeepSurv_800_B = data["n800_run_time_DeepSurv.npy"]
run_time_DeepHit_400_B = data["n400_run_time_DeepHit.npy"]
run_time_DeepHit_800_B = data["n800_run_time_DeepHit.npy"]
run_time_AH_400_B = data["n400_run_time_AH.npy"]
run_time_AH_800_B = data["n800_run_time_AH.npy"]
run_time_RSF_400_B = data["n400_run_time_RSF.npy"]
run_time_RSF_800_B = data["n800_run_time_RSF.npy"]
run_time_PAFT_400_B = data["n400_run_time_PAFT.npy"]
run_time_PAFT_800_B = data["n800_run_time_PAFT.npy"]
run_time_SAFT_400_B = data["n400_run_time_SAFT.npy"]
run_time_SAFT_800_B = data["n800_run_time_SAFT.npy"]
run_time_Coxvary_400_B = data["n400_run_time_Coxvary.npy"]
run_time_Coxvary_800_B = data["n800_run_time_Coxvary.npy"]


St_Cox_Linear_X_400_B = data["n400_S_t_X_Cox_Linear.npy"]
St_Cox_Linear_X_800_B = data["n800_S_t_X_Cox_Linear.npy"]
St_CoxPH_X_400_B = data["n400_S_t_X_CoxPH.npy"]
St_CoxPH_X_800_B = data["n800_S_t_X_CoxPH.npy"]
St_DNN_X_400_B = data["n400_S_t_X_DNN.npy"]
St_DNN_X_800_B = data["n800_S_t_X_DNN.npy"]
St_Cox_CC_X_400_B = data["n400_S_t_X_Cox_CC.npy"]
St_Cox_CC_X_800_B = data["n800_S_t_X_Cox_CC.npy"]
St_Cox_Time_X_400_B = data["n400_S_t_X_Cox_Time.npy"]
St_Cox_Time_X_800_B = data["n800_S_t_X_Cox_Time.npy"]
St_DeepSurv_X_400_B = data["n400_S_t_X_DeepSurv.npy"]
St_DeepSurv_X_800_B = data["n800_S_t_X_DeepSurv.npy"]
St_DeepHit_X_400_B = data["n400_S_t_X_DeepHit.npy"]
St_DeepHit_X_800_B = data["n800_S_t_X_DeepHit.npy"]
St_AH_X_400_B = data["n400_S_t_X_AH.npy"]
St_AH_X_800_B = data["n800_S_t_X_AH.npy"]
St_RSF_X_400_B = data["n400_S_t_X_RSF.npy"]
St_RSF_X_800_B = data["n800_S_t_X_RSF.npy"]
St_PAFT_X_400_B = data["n400_S_t_X_PAFT.npy"]
St_PAFT_X_800_B = data["n800_S_t_X_PAFT.npy"]
St_SAFT_X_400_B = data["n400_S_t_X_SAFT.npy"]
St_SAFT_X_800_B = data["n800_S_t_X_SAFT.npy"]
St_Coxvary_X_400_B = data["n400_S_t_X_Coxvary.npy"]
St_Coxvary_X_800_B = data["n800_S_t_X_Coxvary.npy"]


St_true_X_fig = data["St_true_X_fig.npy"]
t_fig = data["t_fig.npy"]




DMS_Cox_Linears = np.array([np.nanmean(DMS_Cox_Linear_400_B), np.nanmean(DMS_Cox_Linear_800_B)])
DMS_Cox_Linears_sd= np.array([np.sqrt(np.nanmean((DMS_Cox_Linear_400_B-np.nanmean(DMS_Cox_Linear_400_B))**2)), np.sqrt(np.nanmean((DMS_Cox_Linear_800_B-np.nanmean(DMS_Cox_Linear_800_B))**2))])
DMS_CoxPHs = np.array([np.nanmean(DMS_CoxPH_400_B), np.nanmean(DMS_CoxPH_800_B)])
DMS_CoxPHs_sd= np.array([np.sqrt(np.nanmean((DMS_CoxPH_400_B-np.nanmean(DMS_CoxPH_400_B))**2)), np.sqrt(np.nanmean((DMS_CoxPH_800_B-np.nanmean(DMS_CoxPH_800_B))**2))])
DMS_DNNs = np.array([np.nanmean(DMS_DNN_400_B), np.nanmean(DMS_DNN_800_B)])
DMS_DNNs_sd= np.array([np.sqrt(np.nanmean((DMS_DNN_400_B-np.nanmean(DMS_DNN_400_B))**2)), np.sqrt(np.nanmean((DMS_DNN_800_B-np.nanmean(DMS_DNN_800_B))**2))])
DMS_Cox_CCs = np.array([np.nanmean(DMS_Cox_CC_400_B), np.nanmean(DMS_Cox_CC_800_B)])
DMS_Cox_CCs_sd= np.array([np.sqrt(np.nanmean((DMS_Cox_CC_400_B-np.nanmean(DMS_Cox_CC_400_B))**2)), np.sqrt(np.nanmean((DMS_Cox_CC_800_B-np.nanmean(DMS_Cox_CC_800_B))**2))])
DMS_Cox_Times = np.array([np.nanmean(DMS_Cox_Time_400_B), np.nanmean(DMS_Cox_Time_800_B)])
DMS_Cox_Times_sd= np.array([np.sqrt(np.nanmean((DMS_Cox_Time_400_B-np.nanmean(DMS_Cox_Time_400_B))**2)), np.sqrt(np.nanmean((DMS_Cox_Time_800_B-np.nanmean(DMS_Cox_Time_800_B))**2))])
DMS_DeepSurvs = np.array([np.nanmean(DMS_DeepSurv_400_B), np.nanmean(DMS_DeepSurv_800_B)])
DMS_DeepSurvs_sd= np.array([np.sqrt(np.nanmean((DMS_DeepSurv_400_B-np.nanmean(DMS_DeepSurv_400_B))**2)), np.sqrt(np.nanmean((DMS_DeepSurv_800_B-np.nanmean(DMS_DeepSurv_800_B))**2))])
DMS_DeepHits = np.array([np.nanmean(DMS_DeepHit_400_B), np.nanmean(DMS_DeepHit_800_B)])
DMS_DeepHits_sd= np.array([np.sqrt(np.nanmean((DMS_DeepHit_400_B-np.nanmean(DMS_DeepHit_400_B))**2)), np.sqrt(np.nanmean((DMS_DeepHit_800_B-np.nanmean(DMS_DeepHit_800_B))**2))])
DMS_AHs = np.array([np.nanmean(DMS_AH_400_B), np.nanmean(DMS_AH_800_B)])
DMS_AHs_sd= np.array([np.sqrt(np.nanmean((DMS_AH_400_B-np.nanmean(DMS_AH_400_B))**2)), np.sqrt(np.nanmean((DMS_AH_800_B-np.nanmean(DMS_AH_800_B))**2))])
DMS_RSFs = np.array([np.nanmean(DMS_RSF_400_B), np.nanmean(DMS_RSF_800_B)])
DMS_RSFs_sd= np.array([np.sqrt(np.nanmean((DMS_RSF_400_B-np.nanmean(DMS_RSF_400_B))**2)), np.sqrt(np.nanmean((DMS_RSF_800_B-np.nanmean(DMS_RSF_800_B))**2))])
DMS_PAFTs = np.array([np.nanmean(DMS_PAFT_400_B), np.nanmean(DMS_PAFT_800_B)])
DMS_PAFTs_sd= np.array([np.sqrt(np.nanmean((DMS_PAFT_400_B-np.nanmean(DMS_PAFT_400_B))**2)), np.sqrt(np.nanmean((DMS_PAFT_800_B-np.nanmean(DMS_PAFT_800_B))**2))])
DMS_SAFTs = np.array([np.nanmean(DMS_SAFT_400_B), np.nanmean(DMS_SAFT_800_B)])
DMS_SAFTs_sd= np.array([np.sqrt(np.nanmean((DMS_SAFT_400_B-np.nanmean(DMS_SAFT_400_B))**2)), np.sqrt(np.nanmean((DMS_SAFT_800_B-np.nanmean(DMS_SAFT_800_B))**2))])
DMS_Coxvarys = np.array([np.nanmean(DMS_Coxvary_400_B), np.nanmean(DMS_Coxvary_800_B)])
DMS_Coxvarys_sd= np.array([np.sqrt(np.nanmean((DMS_Coxvary_400_B-np.nanmean(DMS_Coxvary_400_B))**2)), np.sqrt(np.nanmean((DMS_Coxvary_800_B-np.nanmean(DMS_Coxvary_800_B))**2))])


IBS_Cox_Linears = np.array([np.nanmean(IBS_Cox_Linear_400_B), np.nanmean(IBS_Cox_Linear_800_B)])
IBS_Cox_Linears_sd= np.array([np.sqrt(np.nanmean((IBS_Cox_Linear_400_B-np.nanmean(IBS_Cox_Linear_400_B))**2)), np.sqrt(np.nanmean((IBS_Cox_Linear_800_B-np.nanmean(IBS_Cox_Linear_800_B))**2))])
IBS_CoxPHs = np.array([np.nanmean(IBS_CoxPH_400_B), np.nanmean(IBS_CoxPH_800_B)])
IBS_CoxPHs_sd= np.array([np.sqrt(np.nanmean((IBS_CoxPH_400_B-np.nanmean(IBS_CoxPH_400_B))**2)), np.sqrt(np.nanmean((IBS_CoxPH_800_B-np.nanmean(IBS_CoxPH_800_B))**2))])
IBS_DNNs = np.array([np.nanmean(IBS_DNN_400_B), np.nanmean(IBS_DNN_800_B)])
IBS_DNNs_sd= np.array([np.sqrt(np.nanmean((IBS_DNN_400_B-np.nanmean(IBS_DNN_400_B))**2)), np.sqrt(np.nanmean((IBS_DNN_800_B-np.nanmean(IBS_DNN_800_B))**2))])
IBS_Cox_CCs = np.array([np.nanmean(IBS_Cox_CC_400_B), np.nanmean(IBS_Cox_CC_800_B)])
IBS_Cox_CCs_sd= np.array([np.sqrt(np.nanmean((IBS_Cox_CC_400_B-np.nanmean(IBS_Cox_CC_400_B))**2)), np.sqrt(np.nanmean((IBS_Cox_CC_800_B-np.nanmean(IBS_Cox_CC_800_B))**2))])
IBS_Cox_Times = np.array([np.nanmean(IBS_Cox_Time_400_B), np.nanmean(IBS_Cox_Time_800_B)])
IBS_Cox_Times_sd= np.array([np.sqrt(np.nanmean((IBS_Cox_Time_400_B-np.nanmean(IBS_Cox_Time_400_B))**2)), np.sqrt(np.nanmean((IBS_Cox_Time_800_B-np.nanmean(IBS_Cox_Time_800_B))**2))])
IBS_DeepSurvs = np.array([np.nanmean(IBS_DeepSurv_400_B), np.nanmean(IBS_DeepSurv_800_B)])
IBS_DeepSurvs_sd= np.array([np.sqrt(np.nanmean((IBS_DeepSurv_400_B-np.nanmean(IBS_DeepSurv_400_B))**2)), np.sqrt(np.nanmean((IBS_DeepSurv_800_B-np.nanmean(IBS_DeepSurv_800_B))**2))])
IBS_DeepHits = np.array([np.nanmean(IBS_DeepHit_400_B), np.nanmean(IBS_DeepHit_800_B)])
IBS_DeepHits_sd= np.array([np.sqrt(np.nanmean((IBS_DeepHit_400_B-np.nanmean(IBS_DeepHit_400_B))**2)), np.sqrt(np.nanmean((IBS_DeepHit_800_B-np.nanmean(IBS_DeepHit_800_B))**2))])
IBS_AHs = np.array([np.nanmean(IBS_AH_400_B), np.nanmean(IBS_AH_800_B)])
IBS_AHs_sd= np.array([np.sqrt(np.nanmean((IBS_AH_400_B-np.nanmean(IBS_AH_400_B))**2)), np.sqrt(np.nanmean((IBS_AH_800_B-np.nanmean(IBS_AH_800_B))**2))])
IBS_RSFs = np.array([np.nanmean(IBS_RSF_400_B), np.nanmean(IBS_RSF_800_B)])
IBS_RSFs_sd= np.array([np.sqrt(np.nanmean((IBS_RSF_400_B-np.nanmean(IBS_RSF_400_B))**2)), np.sqrt(np.nanmean((IBS_RSF_800_B-np.nanmean(IBS_RSF_800_B))**2))])
IBS_PAFTs = np.array([np.nanmean(IBS_PAFT_400_B), np.nanmean(IBS_PAFT_800_B)])
IBS_PAFTs_sd= np.array([np.sqrt(np.nanmean((IBS_PAFT_400_B-np.nanmean(IBS_PAFT_400_B))**2)), np.sqrt(np.nanmean((IBS_PAFT_800_B-np.nanmean(IBS_PAFT_800_B))**2))])
IBS_SAFTs = np.array([np.nanmean(IBS_SAFT_400_B), np.nanmean(IBS_SAFT_800_B)])
IBS_SAFTs_sd= np.array([np.sqrt(np.nanmean((IBS_SAFT_400_B-np.nanmean(IBS_SAFT_400_B))**2)), np.sqrt(np.nanmean((IBS_SAFT_800_B-np.nanmean(IBS_SAFT_800_B))**2))])
IBS_Coxvarys = np.array([np.nanmean(IBS_Coxvary_400_B), np.nanmean(IBS_Coxvary_800_B)])
IBS_Coxvarys_sd= np.array([np.sqrt(np.nanmean((IBS_Coxvary_400_B-np.nanmean(IBS_Coxvary_400_B))**2)), np.sqrt(np.nanmean((IBS_Coxvary_800_B-np.nanmean(IBS_Coxvary_800_B))**2))])


run_time_Cox_Linears = np.array([np.nanmean(run_time_Cox_Linear_400_B), np.nanmean(run_time_Cox_Linear_800_B)])
run_time_CoxPHs = np.array([np.nanmean(run_time_CoxPH_400_B), np.nanmean(run_time_CoxPH_800_B)])
run_time_DNNs = np.array([np.nanmean(run_time_DNN_400_B), np.nanmean(run_time_DNN_800_B)])
run_time_Cox_CCs = np.array([np.nanmean(run_time_Cox_CC_400_B), np.nanmean(run_time_Cox_CC_800_B)])
run_time_Cox_Times = np.array([np.nanmean(run_time_Cox_Time_400_B), np.nanmean(run_time_Cox_Time_800_B)])
run_time_DeepSurvs = np.array([np.nanmean(run_time_DeepSurv_400_B), np.nanmean(run_time_DeepSurv_800_B)])
run_time_DeepHits = np.array([np.nanmean(run_time_DeepHit_400_B), np.nanmean(run_time_DeepHit_800_B)])
run_time_AHs = np.array([np.nanmean(run_time_AH_400_B), np.nanmean(run_time_AH_800_B)])
run_time_RSFs = np.array([np.nanmean(run_time_RSF_400_B), np.nanmean(run_time_RSF_800_B)])
run_time_PAFTs = np.array([np.nanmean(run_time_PAFT_400_B), np.nanmean(run_time_PAFT_800_B)])
run_time_SAFTs = np.array([np.nanmean(run_time_SAFT_400_B), np.nanmean(run_time_SAFT_800_B)])
run_time_Coxvarys = np.array([np.nanmean(run_time_Coxvary_400_B), np.nanmean(run_time_Coxvary_800_B)])

# =================tables=======================

output_folder = "tables"
os.makedirs(output_folder, exist_ok=True)

dic_DMS = {
    "n": Set_n,
    "DMS_DNN": np.array(DMS_DNNs),
    "DMS_Cox_Linear": np.array(DMS_Cox_Linears),
    "DMS_CoxPH": np.array(DMS_CoxPHs),
    "DMS_Coxvary": np.array(DMS_Coxvarys),
    "DMS_Cox_CC": np.array(DMS_Cox_CCs),
    "DMS_Cox_Time": np.array(DMS_Cox_Times),
    "DMS_DeepSurv": np.array(DMS_DeepSurvs),
    "DMS_DeepHit": np.array(DMS_DeepHits),
    "DMS_RSF": np.array(DMS_RSFs),
    "DMS_PAFT": np.array(DMS_PAFTs),
    "DMS_SAFT": np.array(DMS_SAFTs),
    "DMS_AH": np.array(DMS_AHs),
}
result_DMS = pd.DataFrame(dic_DMS)
DMS_path = os.path.join(output_folder, "DMS_AHlinear.csv")
result_DMS.to_csv(DMS_path, index=False)

dic_IBS = {
    "n": Set_n,
    "IBS_DNN": np.array(IBS_DNNs),
    "IBS_Cox_Linear": np.array(IBS_Cox_Linears),
    "IBS_CoxPH": np.array(IBS_CoxPHs),
    "IBS_Coxvary": np.array(IBS_Coxvarys),
    "IBS_Cox_CC": np.array(IBS_Cox_CCs),
    "IBS_Cox_Time": np.array(IBS_Cox_Times),
    "IBS_DeepSurv": np.array(IBS_DeepSurvs),
    "IBS_DeepHit": np.array(IBS_DeepHits),
    "IBS_RSF": np.array(IBS_RSFs),
    "IBS_PAFT": np.array(IBS_PAFTs),
    "IBS_SAFT": np.array(IBS_SAFTs),
    "IBS_AH": np.array(IBS_AHs),
}
result_IBS = pd.DataFrame(dic_IBS)
ibs_path = os.path.join(output_folder, "IBS_AHlinear.csv")
result_IBS.to_csv(ibs_path, index=False)


dic_run_time = {
    "n": Set_n,
    "run_time_DNN": np.array(run_time_DNNs),
    "run_time_Cox_Linear": np.array(run_time_Cox_Linears),
    "run_time_CoxPH": np.array(run_time_CoxPHs),
    "run_time_Coxvary": np.array(run_time_Coxvarys),
    "run_time_Cox_CC": np.array(run_time_Cox_CCs),
    "run_time_Cox_Time": np.array(run_time_Cox_Times),
    "run_time_DeepSurv": np.array(run_time_DeepSurvs),
    "run_time_DeepHit": np.array(run_time_DeepHits), 
    "run_time_RSF": np.array(run_time_RSFs),
    "run_time_PAFT": np.array(run_time_PAFTs),
    "run_time_SAFT": np.array(run_time_SAFTs),
    "run_time_AH": np.array(run_time_AHs),
}
result_run_time = pd.DataFrame(dic_run_time)
run_time_path = os.path.join(output_folder, "run_time_AHlinear.csv")
result_run_time.to_csv(run_time_path, index=False)


# =================pictures=======================
subfolder = "pictures"

St_Cox_Linear_X_400 = np.nanmean(St_Cox_Linear_X_400_B, axis=0)
St_Cox_Linear_X_800 = np.nanmean(St_Cox_Linear_X_800_B, axis=0)
St_CoxPH_X_400 = np.nanmean(St_CoxPH_X_400_B, axis=0)
St_CoxPH_X_800 = np.nanmean(St_CoxPH_X_800_B, axis=0)
St_DNN_X_400 = np.nanmean(St_DNN_X_400_B, axis=0)
St_DNN_X_800 = np.nanmean(St_DNN_X_800_B, axis=0)
St_Cox_CC_X_400 = np.nanmean(St_Cox_CC_X_400_B, axis=0)
St_Cox_CC_X_800 = np.nanmean(St_Cox_CC_X_800_B, axis=0)
St_Cox_Time_X_400 = np.nanmean(St_Cox_Time_X_400_B, axis=0)
St_Cox_Time_X_800 = np.nanmean(St_Cox_Time_X_800_B, axis=0)
St_DeepSurv_X_400 = np.nanmean(St_DeepSurv_X_400_B, axis=0)
St_DeepSurv_X_800 = np.nanmean(St_DeepSurv_X_800_B, axis=0)
St_DeepHit_X_400 = np.nanmean(St_DeepHit_X_400_B, axis=0)
St_DeepHit_X_800 = np.nanmean(St_DeepHit_X_800_B, axis=0)
St_AH_X_400 = np.nanmean(St_AH_X_400_B, axis=0)
St_AH_X_800 = np.nanmean(St_AH_X_800_B, axis=0)
St_RSF_X_400 = np.nanmean(St_RSF_X_400_B, axis=0)
St_RSF_X_800 = np.nanmean(St_RSF_X_800_B, axis=0)
St_PAFT_X_400 = np.nanmean(St_PAFT_X_400_B, axis=0)
St_PAFT_X_800 = np.nanmean(St_PAFT_X_800_B, axis=0)
St_SAFT_X_400 = np.nanmean(St_SAFT_X_400_B, axis=0)
St_SAFT_X_800 = np.nanmean(St_SAFT_X_800_B, axis=0)
St_Coxvary_X_400 = np.nanmean(St_Coxvary_X_400_B, axis=0)
St_Coxvary_X_800 = np.nanmean(St_Coxvary_X_800_B, axis=0)


df = pd.read_csv('X_test.csv')
X = np.array(df, dtype='float32')


for k in range(10):
    formatted_row = np.array2string(
            X[k], 
            separator=',',
            formatter={'float_kind': lambda x: f'{x:.3f}'}
        )
    fig1 = plt.figure() 
    fig1.suptitle(f'AH (linear), $X = {formatted_row}$', fontsize=8)
    # -----n=400, X-----
    ax1_1 = fig1.add_subplot(1, 2, 1)
    ax1_1.set_title('n=400', fontsize=8, loc='center') 
    ax1_1.set_xlabel("t", fontsize=8)  
    ax1_1.set_ylabel('Conditional Survival Function', fontsize=8)  
    ax1_1.tick_params(axis='both', labelsize=6)  
    # -----n=800, X-----
    ax1_2 = fig1.add_subplot(1, 2, 2)
    ax1_2.set_title('n=800', fontsize=8, loc='center') 
    ax1_2.set_xlabel("t", fontsize=8)  
    ax1_2.tick_params(axis='both', labelsize=6)  
    # n=400
    ax1_1.plot(t_fig, St_true_X_fig[:, k], color='black', label='True', linestyle='-')
    ax1_1.plot(t_fig, St_DNN_X_400[k], color='green', label='Proposed', linestyle='--')
    # ax1_1.plot(t_fig, St_Cox_Linear_X_400[:, k], color='red', label='CoxPH(Partial)', linestyle=':')
    # ax1_1.plot(t_fig, St_CoxPH_X_400[:, k], color='blue', label='CoxPH(Full)', linestyle=':')
    # ax1_1.plot(t_fig, St_Coxvary_X_400[:, k], color='orange', label='Cox-Varying', linestyle=':')
    # ax1_1.plot(t_fig, St_Cox_CC_X_400[:, k], color='chocolate', label='Cox-CC', linestyle='--')
    # ax1_1.plot(t_fig, St_Cox_Time_X_400[:, k], color='gray', label='Cox-Time', linestyle='--')
    # ax1_1.plot(t_fig, St_DeepSurv_X_400[:, k], color='purple', label='DeepSurv', linestyle='--')
    # ax1_1.plot(t_fig, St_DeepHit_X_400[:, k], color='gold', label='DeepHit', linestyle='--')
    # ax1_1.plot(t_fig, St_RSF_X_400[:, k], color= 'pink', label='RSF', linestyle='--')
    # ax1_1.plot(t_fig, St_PAFT_X_400[:, k], color='red', label='PAFT', linestyle=':')
    # ax1_1.plot(t_fig, St_SAFT_X_400[:, k], color='blue', label='SAFT', linestyle='--')
    ax1_1.plot(t_fig, St_AH_X_400[:, k], color='red', label='AH', linestyle='--')
    ax1_1.legend(loc='best', fontsize=6)
    # n=800
    ax1_2.plot(t_fig, St_true_X_fig[:, k], color='black', label='True', linestyle='-')
    ax1_2.plot(t_fig, St_DNN_X_800[k], color='green', label='Proposed', linestyle='--')
    # ax1_2.plot(t_fig, St_Cox_Linear_X_800[:, k], color='red', label='CoxPH(Partial)', linestyle=':')
    # ax1_2.plot(t_fig, St_CoxPH_X_800[:, k], color='blue', label='CoxPH(Full)', linestyle=':')
    # ax1_2.plot(t_fig, St_Coxvary_X_800[:, k], color='orange', label='Cox-Varying', linestyle=':')
    # ax1_2.plot(t_fig, St_Cox_CC_X_800[:, k], color='chocolate', label='Cox-CC', linestyle='--')
    # ax1_2.plot(t_fig, St_Cox_Time_X_800[:, k], color='gray', label='Cox-Time', linestyle='--')
    # ax1_2.plot(t_fig, St_DeepSurv_X_800[:, k], color='purple', label='DeepSurv', linestyle='--')
    # ax1_2.plot(t_fig, St_DeepHit_X_800[:, k], color='gold', label='DeepHit', linestyle='--')
    # ax1_2.plot(t_fig, St_RSF_X_800[:, k], color= 'pink', label='RSF', linestyle='--')
    # ax1_2.plot(t_fig, St_PAFT_X_800[:, k], color='red', label='PAFT', linestyle=':')
    # ax1_2.plot(t_fig, St_SAFT_X_800[:, k], color='blue', label='SAFT', linestyle='--')
    ax1_2.plot(t_fig, St_AH_X_800[:, k], color='red', label='AH', linestyle='--')
    ax1_2.legend(loc='best', fontsize=6)
    file_name = os.path.join(subfolder, f'Case6_fig_{k}.jpeg')  
    fig1.savefig(file_name, dpi=600, bbox_inches='tight') 