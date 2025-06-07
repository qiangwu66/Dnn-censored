import numpy as np
import numpy.random as ndm
import pandas as pd
import os

results_folder = "results_folder"
os.makedirs(results_folder, exist_ok=True)


def generate_Cox_1(n, corr, tau):
    Z1 = ndm.binomial(1, 0.5, n)
    Z2 = ndm.binomial(1, 0.25, n)
    mean = np.zeros(4)
    cov = np.identity(4) * (1-corr) + np.ones((4, 4)) * corr
    X0 = ndm.multivariate_normal(mean, cov, n)
    X1 = (np.clip(X0, -2, 2) + 2) / 4
    f_X = Z1 + Z2 + X1[:,0] + X1[:,1] / 2 + X1[:,2] / 3 +  X1[:,3] / 4
    Y = ndm.rand(n)
    T = np.maximum((- 9 * np.log(Y) * np.exp(-f_X)) ** (2/3) - 0.001, 1e-4)
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



import numpy as np
import pandas as pd
import os

def save_data_for_r(n_values, B=500, corr=0.5, tau=2, results_folder="."):
    for n in n_values:
        all_data = [] 

        for b in range(B):
            np.random.seed(500 + b)  
            

            data = generate_Cox_1(n, corr, tau)
            

            n_40 = int(n * 0.4) 
            n_50 = int(n * 0.5) 
            n_90 = int(n * 0.9) 


            df_40 = pd.DataFrame({
                'X1': data['Z1'].flatten()[:n_40],
                'X2': data['Z2'].flatten()[:n_40],
                'X3': data['X1'][:n_40, 0],
                'X4': data['X1'][:n_40, 1],
                'X5': data['X1'][:n_40, 2],
                'X6': data['X1'][:n_40, 3],
                'Y': data['T_O'][:n_40],
                'De': data['De'][:n_40]
            })

            df_50_90 = pd.DataFrame({
                'X1': data['Z1'].flatten()[n_50:n_90],
                'X2': data['Z2'].flatten()[n_50:n_90],
                'X3': data['X1'][n_50:n_90, 0],
                'X4': data['X1'][n_50:n_90, 1],
                'X5': data['X1'][n_50:n_90, 2],
                'X6': data['X1'][n_50:n_90, 3],
                'Y': data['T_O'][n_50:n_90],
                'De': data['De'][n_50:n_90]
            })


            df_combined = pd.concat([df_40, df_50_90], axis=0)
            df_combined['Run'] = b + 1  
            all_data.append(df_combined)


        all_data_combined = pd.concat(all_data, axis=0)


        csv_filename = f"generated_data_n{n}_Cox1.csv"
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)  
        all_data_combined.to_csv(os.path.join(results_folder, csv_filename), index=False)


save_data_for_r(n_values=[1600], results_folder=results_folder)


