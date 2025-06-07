import torch
import numpy as np


def Indicator_matrix(a, b):
    
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    I_M = (a.unsqueeze(1) >= b.unsqueeze(0)).int()

    return I_M



#%% ------------------------
def g1_dnn(polled_data, train_data, val_data, tau, n_layer, n_node, n_lr, n_epoch, patiences):

    X_all = torch.tensor(polled_data['X'], dtype=torch.float32)
    De_all = torch.tensor(polled_data['De'], dtype=torch.float32)
    T_O_all = torch.tensor(polled_data['T_O'], dtype=torch.float32)

    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    De_train = torch.tensor(train_data['De'], dtype=torch.float32)
    T_O_train = torch.tensor(train_data['T_O'], dtype=torch.float32)

    X_val = torch.tensor(val_data['X'], dtype=torch.float32)
    De_val = torch.tensor(val_data['De'], dtype=torch.float32)
    T_O_val = torch.tensor(val_data['T_O'], dtype=torch.float32)

    d_X = X_train.size()[1]
    n = X_all.size()[0] # 1280

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


        nodes = 0.5 * (nodes + 1) * (b - a).unsqueeze(1) + a.unsqueeze(1)  # [batch_size, n_points]， len(a) = len(b) = batch_size
        weights = weights * 0.5 * (b - a).unsqueeze(1)  # [batch_size, n_points], len(a) = len(b) = batch_size

    
        values = func(nodes)  # [batch_size, n_points]

        return torch.sum(weights * values, dim=1)  # [batch_size]

    def batch_func_train(t):
        # t: [batch_size, n_points]
        batch_size, n_points = t.shape
        t = t.reshape(-1, 1)  # [batch_size * n_points, 1]
        X_repeated = X_train.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, d_X]
        inputs = torch.cat((t, X_repeated), dim=1)  # [batch_size * n_points, d_X + 1]
        outputs = model(inputs).squeeze()  # [batch_size * n_points]
        return torch.exp(outputs).reshape(batch_size, n_points)  # 恢复为 [batch_size, n_points]
    
    def batch_func_val(t):
        # t: [batch_size, n_points]
        batch_size, n_points = t.shape
        t = t.reshape(-1, 1)  # [batch_size * n_points, 1]
        X_repeated = X_val.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, d_X]
        inputs = torch.cat((t, X_repeated), dim=1)  # [batch_size * n_points, d_X + 1]
        outputs = model(inputs).squeeze()  # [batch_size * n_points]
        return torch.exp(outputs).reshape(batch_size, n_points)  # 恢复为 [batch_size, n_points]
    
    def batch_func_all(t):
        # t: [batch_size, n_points]
        batch_size, n_points = t.shape
        t = t.reshape(-1, 1)  # [batch_size * n_points, 1]
        X_repeated = X_all.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, d_X]
        inputs = torch.cat((t, X_repeated), dim=1)  # [batch_size * n_points, d_X + 1]
        outputs = model(inputs).squeeze()  # [batch_size * n_points]
        return torch.exp(outputs).reshape(batch_size, n_points)  # 恢复为 [batch_size, n_points]
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(n_epoch):
        model.train()  # Set model to training mode
            
        g_TX = model(torch.cat((T_O_train.unsqueeze(1), X_train), dim=1)).squeeze()

        
        int_exp_g_TX = vectorized_gaussian_quadrature_integral(batch_func_train, torch.zeros_like(T_O_train), T_O_train, n_points=100) # batch_size = 0.8n

        loss = my_loss(De_train, g_TX, int_exp_g_TX)
        # print('epoch=', epoch, 'loss=', loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            g_TX_val = model(torch.cat((T_O_val.unsqueeze(1), X_val), dim=1)).squeeze()
            
            int_exp_g_TX_val = vectorized_gaussian_quadrature_integral(batch_func_val, torch.zeros_like(T_O_val), T_O_val, n_points=100) # batch_size = 0.2n
            
            val_loss = my_loss(De_val, g_TX_val, int_exp_g_TX_val)
            # print('epoch=', epoch, 'val_loss=', val_loss.detach().numpy())
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # Save the best model state
        else:
            patience_counter += 1
            # print('patience_counter =', patience_counter)
            
        if patience_counter >= patiences:
            # print(f'Early stopping at epoch {epoch + 1}', 'validation—loss=', val_loss.detach().numpy())
            break

    # Restore best model if needed
    model.load_state_dict(best_model_state)

    # Evaluation and other code remains unchanged...
    # Test
    model.eval()
    with torch.no_grad():
        g_T_X_n = model(torch.cat((T_O_all.unsqueeze(1), X_all), dim=1)).squeeze() # n

        
        I_T_T_n = Indicator_matrix(T_O_train, T_O_train).float() # n * n

        I_T_T_mean = torch.mean(I_T_T_n, dim=0) # n
        
        W_1_n = I_T_T_mean


        nodes, weights = np.polynomial.legendre.leggauss(100)
        nodes = torch.tensor(nodes, dtype=torch.float32)  
        weights = torch.tensor(weights, dtype=torch.float32)  

        I_T_t_nodes = Indicator_matrix(T_O_train, 0.5 * (nodes + 1) * tau).float() # n * 100
        I_T_s_k_mean = torch.mean(I_T_t_nodes, dim=0) # 100

        W_s_k_1 = I_T_s_k_mean.tile((int(n/2), 1)) # n * 100

        b = tau * torch.ones(int(n/2))
        a = torch.zeros(int(n/2))
        t_nodes = 0.5 * (nodes + 1) * (b - a).unsqueeze(1) + a.unsqueeze(1)  # [n, 100]， len(a) = len(b) = n
        weights_new = weights * 0.5 * (b - a).unsqueeze(1)  # [n, 100], len(a) = len(b) = n
        Exp_g = batch_func_train(t_nodes) # [n, 100]
        int_Y_exp_1 = torch.sum(weights_new * I_T_t_nodes * Exp_g * W_s_k_1, dim=1) # n

        sigma_1_n1 = torch.sqrt(torch.mean((De_train * W_1_n - int_Y_exp_1) ** 2)) 


    
    return {
        'g1_T_X_n': g_T_X_n.detach().numpy(), # n
        'sigma_1_n1': sigma_1_n1.detach().numpy(),
    }

