import torch
import numpy as np

#%% ------------------------
def g_dnn(train_data, val_data, test_data, t_nodes, t_fig, s_k, n_layer, n_node, n_lr, n_epoch, patiences,
         agq_n_low=50, agq_n_high=100, agq_tol=1e-5, agq_max_subdiv=20, clamp_exp_max=None):
    print('DNN_iteration')

    # ---------- Data tensors ----------
    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    De_train = torch.tensor(train_data['De'], dtype=torch.float32)
    T_O_train = torch.tensor(train_data['T_O'], dtype=torch.float32)

    X_val = torch.tensor(val_data['X'], dtype=torch.float32)
    De_val = torch.tensor(val_data['De'], dtype=torch.float32)
    T_O_val = torch.tensor(val_data['T_O'], dtype=torch.float32)

    X_test = torch.tensor(test_data['X'], dtype=torch.float32)
    T_O_test = torch.tensor(test_data['T_O'], dtype=torch.float32)

    t_nodes = torch.tensor(t_nodes, dtype=torch.float32)
    t_fig = torch.tensor(t_fig, dtype=torch.float32)

    d_X = X_train.size()[1]

    # ---------- Model ----------
    class DNNModel(torch.nn.Module):
        def __init__(self, in_features=d_X + 1, out_features=1, hidden_nodes=n_node, hidden_layers=n_layer, drop_rate=0.0):
            super(DNNModel, self).__init__()
            layers = []
            layers.append(torch.nn.Linear(in_features, hidden_nodes))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(drop_rate))
            for _ in range(hidden_layers):
                layers.append(torch.nn.Linear(hidden_nodes, hidden_nodes))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(drop_rate))
            layers.append(torch.nn.Linear(hidden_nodes, out_features))
            self.linear_relu_stack = torch.nn.Sequential(*layers)
        def forward(self, x):
            return self.linear_relu_stack(x)

    model = DNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)

    # ---------- Loss ----------
    def my_loss(De, g_TX, int_exp_g_TX):
        loss_fun = De * g_TX - int_exp_g_TX
        return -loss_fun.mean()

    # ---------- Adaptive Gaussian Quadrature ----------
    def adaptive_gaussian_quadrature_integral(func, a, b, X_source, n_low=50, n_high=100, tol=1e-5, max_subdiv=20):
        assert a.shape == b.shape
        device = a.device
        dtype = a.dtype
        N = a.shape[0]

        nodes_low_np, weights_low_np = np.polynomial.legendre.leggauss(n_low)
        nodes_high_np, weights_high_np = np.polynomial.legendre.leggauss(n_high)
        nodes_low = torch.tensor(nodes_low_np, dtype=dtype, device=device)     # (n_low,)
        weights_low = torch.tensor(weights_low_np, dtype=dtype, device=device) # (n_low,)
        nodes_high = torch.tensor(nodes_high_np, dtype=dtype, device=device)   # (n_high,)
        weights_high = torch.tensor(weights_high_np, dtype=dtype, device=device)

        stacks = [[(a[i], b[i], 0)] for i in range(N)]
        results = torch.zeros(N, dtype=dtype, device=device)

        while True:
            batch_left, batch_right, batch_idx, batch_depth = [], [], [], []
            for i in range(N):
                if stacks[i]:
                    l, r, d = stacks[i].pop()
                    batch_left.append(l)
                    batch_right.append(r)
                    batch_idx.append(i)
                    batch_depth.append(d)
            if not batch_left:
                break

            lefts = torch.stack(batch_left)   # (M,)
            rights = torch.stack(batch_right) # (M,)
            idxs = torch.tensor(batch_idx, dtype=torch.long, device=device)  # (M,)
            depths = batch_depth  # python list of len M

            mids = 0.5 * (lefts + rights)
            half_len = 0.5 * (rights - lefts)

            t_low = mids.unsqueeze(1) + half_len.unsqueeze(1) * nodes_low.unsqueeze(0)    # (M, n_low)
            t_high = mids.unsqueeze(1) + half_len.unsqueeze(1) * nodes_high.unsqueeze(0)  # (M, n_high)

            X_batch = X_source[idxs, :]  # (M, d)

            vals_low = func(t_low, X_batch)     # (M, n_low)
            vals_high = func(t_high, X_batch)   # (M, n_high)

            est_low = (half_len.unsqueeze(1) * weights_low.unsqueeze(0) * vals_low).sum(dim=1)    # (M,)
            est_high = (half_len.unsqueeze(1) * weights_high.unsqueeze(0) * vals_high).sum(dim=1) # (M,)
            err = (est_high - est_low).abs()  # (M,)

            for j in range(idxs.shape[0]):
                i = batch_idx[j]
                d = depths[j]
                if (err[j] <= tol) or (d >= max_subdiv):
                    results[i] = results[i] + est_high[j]
                else:
                    mid = 0.5 * (lefts[j] + rights[j])
                    stacks[i].append((mid, rights[j], d + 1))
                    stacks[i].append((lefts[j], mid, d + 1))

        return results

    # ---------- Batch integrands: func(t, X_batch) ----------
    def batch_func_train(t, X_batch):
        # t: (M, n_points), X_batch: (M, d_X)
        M, n_points = t.shape
        t_flat = t.reshape(-1, 1)  # (M*n_points, 1)
        X_rep = X_batch.repeat_interleave(n_points, dim=0)  # (M*n_points, d_X)
        inputs = torch.cat((t_flat, X_rep), dim=1)  # (M*n_points, d_X+1)
        outputs = model(inputs).squeeze()  # (M*n_points,)
        if clamp_exp_max is not None:
            outputs = torch.clamp(outputs, max=clamp_exp_max)
        vals = torch.exp(outputs).reshape(M, n_points)
        return vals

    def batch_func_val(t, X_batch):
        M, n_points = t.shape
        t_flat = t.reshape(-1, 1)
        X_rep = X_batch.repeat_interleave(n_points, dim=0)
        inputs = torch.cat((t_flat, X_rep), dim=1)
        outputs = model(inputs).squeeze()
        if clamp_exp_max is not None:
            outputs = torch.clamp(outputs, max=clamp_exp_max)
        vals = torch.exp(outputs).reshape(M, n_points)
        return vals

    def batch_func_test(t, X_batch):
        M, n_points = t.shape
        t_flat = t.reshape(-1, 1)
        X_rep = X_batch.repeat_interleave(n_points, dim=0)
        inputs = torch.cat((t_flat, X_rep), dim=1)
        outputs = model(inputs).squeeze()
        if clamp_exp_max is not None:
            outputs = torch.clamp(outputs, max=clamp_exp_max)
        vals = torch.exp(outputs).reshape(M, n_points)
        return vals

    # ---------- Training loop with early stopping ----------
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(n_epoch):
        model.train()

        g_TX = model(torch.cat((T_O_train.unsqueeze(1), X_train), dim=1)).squeeze()

        # ∫ exp(g(s|X)) ds using AGQ
        int_exp_g_TX = adaptive_gaussian_quadrature_integral(
            batch_func_train, torch.zeros_like(T_O_train), T_O_train, X_train,
            n_low=agq_n_low, n_high=agq_n_high, tol=agq_tol, max_subdiv=agq_max_subdiv
        )

        loss = my_loss(De_train, g_TX, int_exp_g_TX)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            g_TX_val = model(torch.cat((T_O_val.unsqueeze(1), X_val), dim=1)).squeeze()
            int_exp_g_TX_val = adaptive_gaussian_quadrature_integral(
                batch_func_val, torch.zeros_like(T_O_val), T_O_val, X_val,
                n_low=agq_n_low, n_high=agq_n_high, tol=agq_tol, max_subdiv=agq_max_subdiv
            )
            val_loss = my_loss(De_val, g_TX_val, int_exp_g_TX_val)
            print('epoch=', epoch, 'val_loss=', val_loss.detach().cpu().numpy())

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            print('patience_counter =', patience_counter)
        if patience_counter >= patiences:
            print(f'Early stopping at epoch {epoch + 1}', 'validation--loss=', val_loss.detach().cpu().numpy())
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ---------- Test / Evaluation ----------
    model.eval()
    with torch.no_grad():
        S_T_X_fig = torch.ones(len(t_fig), len(T_O_test), dtype=torch.float32)
        for k in range(len(t_fig)):
            t_fig_k_repeat = torch.full((len(T_O_test),), t_fig[k], dtype=torch.float32)
            int_exp_g_TX = adaptive_gaussian_quadrature_integral(
                batch_func_test, torch.zeros_like(t_fig_k_repeat), t_fig_k_repeat, X_test,
                n_low=agq_n_low, n_high=agq_n_high, tol=agq_tol, max_subdiv=agq_max_subdiv
            )
            S_T_X_fig[k] = torch.exp(-int_exp_g_TX)

        int_exp_g_TX = adaptive_gaussian_quadrature_integral(
            batch_func_test, torch.zeros_like(T_O_test), T_O_test, X_test,
            n_low=agq_n_low, n_high=agq_n_high, tol=agq_tol, max_subdiv=agq_max_subdiv
        )
        S_T_X_DMS = torch.exp(-int_exp_g_TX)

        S_T_X_ibs = torch.ones(len(s_k), len(T_O_test), dtype=torch.float32)
        for k in range(len(s_k)):
            s_k_k_repeat = torch.full((len(T_O_test),), s_k[k], dtype=torch.float32)
            int_exp_g_TX = adaptive_gaussian_quadrature_integral(
                batch_func_test, torch.zeros_like(s_k_k_repeat), s_k_k_repeat, X_test,
                n_low=agq_n_low, n_high=agq_n_high, tol=agq_tol, max_subdiv=agq_max_subdiv
            )
            S_T_X_ibs[k] = torch.exp(-int_exp_g_TX)

    return {
        'S_T_X_fig': S_T_X_fig.T.detach().cpu().numpy(),  # (n_test, len(t_fig))
        'S_T_X_ibs': S_T_X_ibs.T.detach().cpu().numpy(),  # (n_test, len(s_k))
        'S_T_X_DMS': S_T_X_DMS.detach().cpu().numpy()     # (n_test,)
    }