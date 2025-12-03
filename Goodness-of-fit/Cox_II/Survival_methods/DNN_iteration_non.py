import numpy as np
import torch
import scipy.optimize as spo
from B_spline import B_S


def Indicator_matrix(a, b):
    a = np.array(a)
    b = np.array(b)
    I_M = (a[:, np.newaxis] >= b).astype(int)
    return I_M



#%% estimate lambda_0
def Est_lambda(train_data, t_nodes, m, nodevec, tau, f_x_train):
    Y_train = train_data['T_O']
    De_train = train_data['De']
    def COF(*args):
        c = args[0]
        # Beta_X = X_train @ c[:d] # n
        lambda_t_nodes = B_S(m, t_nodes, nodevec) @ c  # len(t_nodes)
        lambda_Y = B_S(m, Y_train, nodevec) @ c # n
        Loss = - np.mean(De_train * (np.log(lambda_Y + 1e-5) + f_x_train)- 
                         Indicator_matrix(Y_train, t_nodes) @ lambda_t_nodes * (tau / len(t_nodes)) * np.exp(f_x_train))
        return Loss


    initial_c = 0.1 * np.ones(m+4)
    bounds = [(0, None)] * (m+4)
    result = spo.minimize(COF, initial_c, method='SLSQP', bounds=bounds)
    return result['x']


#%% estimate the function of covariates
def f_dnn(X_all, train_data, val_data, n_layer, n_node, n_lr, n_epoch, patiences, lambda_Y_train, Inte_lambda_Y_train, lambda_Y_val, Inte_lambda_Y_val):

    X_all = torch.tensor(X_all, dtype=torch.float32)

    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    De_train = torch.tensor(train_data['De'], dtype=torch.float32)

    X_val = torch.tensor(val_data['X'], dtype=torch.float32)
    De_val = torch.tensor(val_data['De'], dtype=torch.float32)

    lambda_Y_train = torch.tensor(lambda_Y_train, dtype=torch.float32)
    Inte_lambda_Y_train = torch.tensor(Inte_lambda_Y_train, dtype=torch.float32)
    lambda_Y_val = torch.tensor(lambda_Y_val, dtype=torch.float32)
    Inte_lambda_Y_val = torch.tensor(Inte_lambda_Y_val, dtype=torch.float32)

    d_X = X_train.size()[1]

    # Define the DNN model
    class DNNModel(torch.nn.Module):
        def __init__(self, in_features=d_X, out_features=1, hidden_nodes=n_node, hidden_layers=n_layer, drop_rate=0):
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
    def my_loss(De, lambda_Y, Inte_lambda_Y, f_TX):
        loss_fun = De * (torch.log(lambda_Y + 1e-5) + f_TX) - Inte_lambda_Y * torch.exp(f_TX)
        return -loss_fun.mean()
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(n_epoch):
        model.train()  # Set model to training mode    
        f_TX = model(X_train)
        loss = my_loss(De_train, lambda_Y_train, Inte_lambda_Y_train, f_TX[:, 0])
        # print('epoch=', epoch, 'loss=', loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            f_TX_val = model(X_val)
            val_loss = my_loss(De_val, lambda_Y_val, Inte_lambda_Y_val, f_TX_val[:, 0])
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
    model.eval()
    with torch.no_grad():
        f_T_X_all = model(X_all) # n
        f_T_X_train = model(X_train) # n2
    
    return {
        'f_T_X_all': f_T_X_all[:, 0].detach().numpy(), # n
        'f_T_X_train': f_T_X_train[:, 0].detach().numpy(), # n2
    }




#%% iteration
def Estimates_non_DNN(polled_data, train_data, val_data, t_nodes, m, nodevec, tau, n_layer, n_node, n_lr, n_epoch, patiences):
    T_O_train = train_data['T_O']
    De_train = train_data['De']

    T_O_val = val_data['T_O']

    T_O_all = polled_data['T_O']
    X_all = polled_data['X']
    De_all = polled_data['De']

    coefs0 = np.zeros(m+4)
    f_x_train_0 = np.zeros(len(T_O_train))

    for loop in range(5):
        # print('iteration =', loop)

        coefs = Est_lambda(train_data, t_nodes, m, nodevec, tau, f_x_train_0)
        
        lambda_Y_train = B_S(m, T_O_train, nodevec) @ coefs
        lambda_Y_val = B_S(m, T_O_val, nodevec) @ coefs


        
        Inte_lambda_Y_train = (tau / len(t_nodes)) * Indicator_matrix(T_O_train, t_nodes) @ (B_S(m, t_nodes, nodevec) @ coefs)
        Inte_lambda_Y_val = (tau / len(t_nodes)) * Indicator_matrix(T_O_val, t_nodes) @ (B_S(m, t_nodes, nodevec) @ coefs)

        f_x = f_dnn(X_all, train_data, val_data, n_layer, n_node, n_lr, n_epoch, patiences, lambda_Y_train, Inte_lambda_Y_train, lambda_Y_val, Inte_lambda_Y_val)
        # print(coefs)

        if (np.max(abs(coefs - coefs0)) < 0.05) and (loop >= 2):
            break
        else:
            coefs0 = coefs
            f_x_train_0 = f_x['f_T_X_train']




    # After estimation
    lambda_Y_all = B_S(m, T_O_all, nodevec) @ coefs

    g0_T_X_n_non = np.log(lambda_Y_all + 1e-4) + f_x['f_T_X_all']

    lambda_t_nodes = B_S(m, t_nodes, nodevec) @ coefs



    # ----------------------------------------
    I_T_T_n_train = Indicator_matrix(T_O_train, T_O_train) # n * n

    I_T_T_mean_train = np.mean(I_T_T_n_train, axis=0) # n
    W_1_n_train = I_T_T_mean_train

    I_T_t_nodes_train = Indicator_matrix(T_O_train, t_nodes) # n * 101

    int_Y_exp_train = (tau / len(t_nodes)) * (I_T_t_nodes_train @ (lambda_t_nodes * np.mean(I_T_t_nodes_train, axis = 0))) * np.exp(f_x['f_T_X_train'])

    sigma_1_n2 = np.sqrt(np.mean((De_train * W_1_n_train - int_Y_exp_train) ** 2)) 
    

    return{
        "g0_T_X_n_non": g0_T_X_n_non,
        'sigma_1_n2_noncoxph': sigma_1_n2,
    }

