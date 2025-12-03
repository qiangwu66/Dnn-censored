import torch
import numpy as np
import math


def Indicator_matrix(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    I_M = (a.unsqueeze(1) >= b.unsqueeze(0)).int()

    return I_M



def g1_dnn(polled_data, train_data, val_data, tau, n_layer, n_node, n_lr, n_epoch, patiences, g2_T_X_n, sigma2_ws):

    X_all = torch.tensor(polled_data['X'], dtype=torch.float32)
    De_all = torch.tensor(polled_data['De'], dtype=torch.float32)
    T_O_all = torch.tensor(polled_data['T_O'], dtype=torch.float32)

    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    De_train = torch.tensor(train_data['De'], dtype=torch.float32)
    T_O_train = torch.tensor(train_data['T_O'], dtype=torch.float32)

    X_val = torch.tensor(val_data['X'], dtype=torch.float32)
    De_val = torch.tensor(val_data['De'], dtype=torch.float32)
    T_O_val = torch.tensor(val_data['T_O'], dtype=torch.float32)

    g2_T_X_n = torch.tensor(g2_T_X_n, dtype=torch.float32)
    sigma2_ws = torch.tensor(sigma2_ws, dtype=torch.float32)

    d_X = X_train.size()[1]
    n = X_all.size()[0] 

    # Define the DNN model
    class DNNModel(torch.nn.Module):
        def __init__(self, in_features=d_X + 1, out_features=1, hidden_nodes=n_node, hidden_layers=n_layer, drop_rate=0):
            super(DNNModel, self).__init__()
            layers = []
            # Input layer
            layers.append(torch.nn.Linear(in_features, hidden_nodes))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(drop_rate))
            # Hidden layers
            for _ in range(hidden_layers):
                layers.append(torch.nn.Linear(hidden_nodes, hidden_nodes))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(drop_rate))
            # Output layer
            layers.append(torch.nn.Linear(hidden_nodes, out_features))
            self.linear_relu_stack = torch.nn.Sequential(*layers)
        
        def forward(self, x):
            return self.linear_relu_stack(x)

    # Initialize model and optimizer
    model = DNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)

    # Custom loss function
    def my_loss(De, g_TX, int_exp_g_TX):
        loss_fun = De * g_TX - int_exp_g_TX
        return -loss_fun.mean()


    def vectorized_gaussian_quadrature_integral(func, a, b, n_points):
        
        nodes, weights = np.polynomial.legendre.leggauss(n_points)
        nodes = torch.tensor(nodes, dtype=torch.float32) 
        weights = torch.tensor(weights, dtype=torch.float32) 


        nodes = 0.5 * (nodes + 1) * (b - a).unsqueeze(1) + a.unsqueeze(1)  
        weights = weights * 0.5 * (b - a).unsqueeze(1)  

        values = func(nodes)

        return torch.sum(weights * values, dim=1) 


    def batch_func_train(t):
        batch_size, n_points = t.shape
        t = t.reshape(-1, 1) 
        X_repeated = X_train.repeat_interleave(n_points, dim=0) 
        inputs = torch.cat((t, X_repeated), dim=1) 
        outputs = model(inputs).squeeze() 
        return torch.exp(outputs).reshape(batch_size, n_points) 
    

    def batch_func_val(t):
        batch_size, n_points = t.shape
        t = t.reshape(-1, 1) 
        X_repeated = X_val.repeat_interleave(n_points, dim=0) 
        inputs = torch.cat((t, X_repeated), dim=1) 
        outputs = model(inputs).squeeze() 
        return torch.exp(outputs).reshape(batch_size, n_points) 
    
    def batch_func_all(t):
        batch_size, n_points = t.shape
        t = t.reshape(-1, 1) 
        X_repeated = X_all.repeat_interleave(n_points, dim=0) 
        inputs = torch.cat((t, X_repeated), dim=1) 
        outputs = model(inputs).squeeze() 
        return torch.exp(outputs).reshape(batch_size, n_points) 
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(n_epoch):
        model.train()  # Set model to training mode
            
        g_TX = model(torch.cat((T_O_train.unsqueeze(1), X_train), dim=1)).squeeze()

        int_exp_g_TX = vectorized_gaussian_quadrature_integral(batch_func_train, torch.zeros_like(T_O_train), T_O_train, n_points=100)

        loss = my_loss(De_train, g_TX, int_exp_g_TX)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            g_TX_val = model(torch.cat((T_O_val.unsqueeze(1), X_val), dim=1)).squeeze()
            int_exp_g_TX_val = vectorized_gaussian_quadrature_integral(batch_func_val, torch.zeros_like(T_O_val), T_O_val, n_points=100) 
            
            val_loss = my_loss(De_val, g_TX_val, int_exp_g_TX_val)
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # Save the best model state
        else:
            patience_counter += 1
            
        if patience_counter >= patiences:
            break

    # Restore best model if needed
    model.load_state_dict(best_model_state)

    # Evaluation and other code remains unchanged...
    # Test
    model.eval()
    with torch.no_grad():
        int_exp_g_TX = vectorized_gaussian_quadrature_integral(batch_func_all, torch.zeros_like(T_O_all), T_O_all, n_points=100)
        S1_T_X_n = torch.exp(-int_exp_g_TX)
        # g(T_i,X_i), i = 1,..., n
        g1_T_X_n = model(torch.cat((T_O_all.unsqueeze(1), X_all), dim=1)).squeeze()

        I_T_T_n = Indicator_matrix(T_O_all, T_O_all).float()

        I_T_T_mean = torch.mean(I_T_T_n, dim=0) 
        
        # (0,0), (1,0), (0.5,0.5), (1,1)
        W_1_n = I_T_T_mean
        W_2_n = I_T_T_mean * S1_T_X_n
        W_3_n = I_T_T_mean * torch.sqrt(S1_T_X_n) * torch.sqrt(1 - S1_T_X_n)
        W_4_n = I_T_T_mean * S1_T_X_n * (1 - S1_T_X_n)

        g_diff = g1_T_X_n - g2_T_X_n
    

        T_w_1 = math.sqrt(n) * torch.mean(De_all * W_1_n * g_diff)
        T_w_2 = math.sqrt(n) * torch.mean(De_all * W_2_n * g_diff)
        T_w_3 = math.sqrt(n) * torch.mean(De_all * W_3_n * g_diff)
        T_w_4 = math.sqrt(n) * torch.mean(De_all * W_4_n * g_diff)



        nodes, weights = np.polynomial.legendre.leggauss(100)
        nodes = torch.tensor(nodes, dtype=torch.float32) 
        weights = torch.tensor(weights, dtype=torch.float32)

        s_k = 0.5 * (nodes + 1) * tau
        S_s_k_X = torch.zeros(n, 100, dtype=torch.float32) 
        for k in range(len(s_k)):
            s_k_k_repeat = torch.full((n,), s_k[ k], dtype=torch.float32) 
            int_exp_g_TX = vectorized_gaussian_quadrature_integral(batch_func_all, torch.zeros_like(s_k_k_repeat), s_k_k_repeat, n_points=100)
            S_s_k_X[:,k] = torch.exp(-int_exp_g_TX)

        I_T_t_nodes = Indicator_matrix(T_O_all, 0.5 * (nodes + 1) * tau).float() 
        I_T_s_k_mean = torch.mean(I_T_t_nodes, dim=0) 

        W_s_k_1 = I_T_s_k_mean.tile((n, 1))
        W_s_k_2 = S_s_k_X * I_T_s_k_mean
        W_s_k_3 = torch.sqrt(S_s_k_X * (1-S_s_k_X)) * I_T_s_k_mean 
        W_s_k_4 = S_s_k_X * (1 - S_s_k_X) * I_T_s_k_mean

        b = tau * torch.ones(n)
        a = torch.zeros(n)
        t_nodes = 0.5 * (nodes + 1) * (b - a).unsqueeze(1) + a.unsqueeze(1) 
        weights_new = weights * 0.5 * (b - a).unsqueeze(1)  
        Exp_g = batch_func_all(t_nodes)
        int_Y_exp_1 = torch.sum(weights_new * I_T_t_nodes * Exp_g * W_s_k_1, dim=1) 
        int_Y_exp_2 = torch.sum(weights_new * I_T_t_nodes * Exp_g * W_s_k_2, dim=1) 
        int_Y_exp_3 = torch.sum(weights_new * I_T_t_nodes * Exp_g * W_s_k_3, dim=1) 
        int_Y_exp_4 = torch.sum(weights_new * I_T_t_nodes * Exp_g * W_s_k_4, dim=1) 

        sigma_1 = torch.sqrt(torch.mean((De_all * W_1_n - int_Y_exp_1) ** 2)) 
        sigma_2 = torch.sqrt(torch.mean((De_all * W_2_n - int_Y_exp_2) ** 2))
        sigma_3 = torch.sqrt(torch.mean((De_all * W_3_n - int_Y_exp_3) ** 2)) 
        sigma_4 = torch.sqrt(torch.mean((De_all * W_4_n - int_Y_exp_4) ** 2))
        sigma1_ws =  torch.tensor([sigma_1, sigma_2, sigma_3, sigma_4])

        tau_ws = torch.sqrt((len(T_O_all) / len(T_O_train)) * sigma1_ws ** 2 + (len(T_O_all) / (len(T_O_all) - len(T_O_train))) * sigma2_ws ** 2)

        U_tau_w1 = T_w_1/tau_ws[0]
        U_tau_w2 = T_w_2/tau_ws[1] 
        U_tau_w3 = T_w_3/tau_ws[2]
        U_tau_w4 = T_w_4/tau_ws[3]

    
    return {
        'U_tau_w1': U_tau_w1.detach().numpy(),
        'U_tau_w2': U_tau_w2.detach().numpy(),
        'U_tau_w3': U_tau_w3.detach().numpy(),
        'U_tau_w4': U_tau_w4.detach().numpy(),
    }

