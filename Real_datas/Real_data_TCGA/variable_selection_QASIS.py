import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass
from patsy import dmatrix
import cvxpy as cp
from sksurv.util import Surv
import torch
import random
from joblib import Parallel, delayed

try:
    from lifelines import KaplanMeierFitter
    HAS_LIFELINES = True
except Exception:
    HAS_LIFELINES = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(1)


features = np.load('histology_features_features.npy')
time_data = pd.read_csv('Survival_time_with_event.csv')

X = features

y = Surv.from_arrays(
    event=np.array(time_data['censored'], dtype='float32') == 1,
    time=np.array(time_data['Survival_time'], dtype='float32')
)

T = y['time']
delta = y['event']

n = len(T)


@dataclass
class QaSISConfig:
    quantile: float = 0.5
    df_spline: int = 4            
    degree: int = 3             
    include_intercept: bool = True
    top_k: Optional[int] = None
    threshold: Optional[float] = None
    standardize_X: bool = True
    random_state: int = 1

    is_survival: bool = False
    y_star_col: Optional[str] = None
    event_col: Optional[str] = None
    km_epsilon: float = 1e-3        
    weight_clip_q: float = 0.99     
 
    n_jobs: int = 4                 
    backend: str = "loky"          


@dataclass
class QaSISResult:
    selected_indices: List[int]
    utilities: np.ndarray
    ranked_indices: List[int]
    fhat_values: Optional[np.ndarray] = None
    spline_designs: Optional[List[np.ndarray]] = None


# ============================
def _scale_to_01(x: np.ndarray) -> np.ndarray:
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if xmax <= xmin:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def _build_bspline_design(x: np.ndarray, df: int, degree: int, include_intercept: bool) -> np.ndarray:
    df = max(df, degree + 1)  
    data = {"x": x}
    design = dmatrix(
        f"bs(x, df={df}, degree={degree}, include_intercept={str(include_intercept)}) - 1",
        data
    )
    return np.asarray(design)


def _quantile_regression_cvxpy(y, X, tau, sample_weight=None, l2_reg=1e-4):

    n, d = X.shape
    b = cp.Variable(d)
    r = y - X @ b
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)

    loss = cp.sum(cp.multiply(w, cp.maximum(tau * r, (tau - 1) * r))) + l2_reg * cp.sum_squares(b)
    prob = cp.Problem(cp.Minimize(loss))

    tried = []
    for solver, kwargs in [
        (cp.SCS,      dict(eps=1e-6, max_iters=300000, acceleration=True, verbose=False)),
        (cp.OSQP,     dict(eps_abs=1e-6, eps_rel=1e-6, max_iter=300000, verbose=False)),
        (cp.ECOS,     dict(abstol=1e-7, reltol=1e-7, feastol=1e-7, max_iters=10000, verbose=False)),
        (cp.CLARABEL, dict(verbose=False)),
        (cp.HIGHS,    dict(verbose=False)),
    ]:
        try:
            prob.solve(solver=solver, **kwargs)
            tried.append((getattr(solver, "__name__", str(solver)), prob.status))
            if b.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                break
        except Exception as e:
            tried.append((str(solver), f"error: {e}"))
            continue

    if b.value is None:
        raise RuntimeError(f"Quantile regression failed. Tried: {tried}. Final status: {prob.status}")

    return np.array(b.value).reshape(-1)


def _compute_ipcw_weights(y_star: np.ndarray, event: np.ndarray, km_epsilon: float = 1e-4) -> np.ndarray:
    if not HAS_LIFELINES:
        raise ImportError("lifelines is required for IPCW. pip install lifelines")

    km = KaplanMeierFitter()
    censor_event = 1 - event.astype(int) 
    km.fit(durations=y_star, event_observed=censor_event)
    G_vals = km.predict(y_star).to_numpy()
    G_vals = np.clip(G_vals, km_epsilon, 1.0)
    w = event.astype(float) / G_vals
    return w


# ============================
def _fit_single_feature(xj: np.ndarray, y_obs: np.ndarray, yq: float,
                        sample_weight: np.ndarray, cfg: QaSISConfig,
                        min_range: float = 1e-12) -> Tuple[float, np.ndarray]:

    if np.nanmax(xj) - np.nanmin(xj) < min_range:
        return 0.0, np.zeros_like(xj, dtype=float)

    B = _build_bspline_design(xj, df=cfg.df_spline, degree=cfg.degree, include_intercept=cfg.include_intercept)
    beta = _quantile_regression_cvxpy(y_obs, B, cfg.quantile, sample_weight=sample_weight, l2_reg=5e-4)
    fhat = B @ beta - yq
    utility = float(np.mean(fhat**2))
    return utility, fhat


# ===========================
def qasis_screen(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, pd.DataFrame],
    cfg: QaSISConfig
) -> QaSISResult:

    if isinstance(X, pd.DataFrame):
        X_mat = X.to_numpy()
    else:
        X_mat = np.asarray(X)

    n, p = X_mat.shape

    if cfg.is_survival:
        if not isinstance(y, (pd.DataFrame,)):
            raise ValueError("For survival mode, y must be a DataFrame with columns y_star_col and event_col.")
        if cfg.y_star_col is None or cfg.event_col is None:
            raise ValueError("Please set cfg.y_star_col and cfg.event_col for survival mode.")
        y_star = y[cfg.y_star_col].to_numpy().astype(float)
        event = y[cfg.event_col].to_numpy().astype(int)


        w = _compute_ipcw_weights(y_star, event, cfg.km_epsilon)

        if cfg.weight_clip_q is not None and 0 < cfg.weight_clip_q < 1:
            upper = np.quantile(w, cfg.weight_clip_q)
            w = np.clip(w, None, upper)

        def weighted_quantile(values, quantile, sample_weight):
            sorter = np.argsort(values)
            values = values[sorter]
            sw = sample_weight[sorter]
            cdf = np.cumsum(sw) / np.sum(sw)
            return np.interp(quantile, cdf, values)

        yq = weighted_quantile(y_star, cfg.quantile, w)
        y_obs = y_star.copy()
        sample_weight = w
    else:
        y_vec = np.asarray(y).astype(float).reshape(-1)
        yq = np.quantile(y_vec, cfg.quantile)
        y_obs = y_vec
        sample_weight = np.ones(n, dtype=float)

    if cfg.standardize_X:
        X_scaled = np.zeros_like(X_mat, dtype=float)
        for j in range(p):
            X_scaled[:, j] = _scale_to_01(X_mat[:, j])
    else:
        X_scaled = X_mat.astype(float)

    results = Parallel(n_jobs=cfg.n_jobs, backend=cfg.backend, verbose=0)(
        delayed(_fit_single_feature)(
            X_scaled[:, j], y_obs, yq, sample_weight, cfg
        ) for j in range(p)
    )

    utilities = np.array([res[0] for res in results], dtype=float)
    fhat_vals = np.column_stack([res[1] for res in results]) if results else np.zeros_like(X_scaled)

    ranked_idx = np.argsort(utilities)[::-1].tolist()

    if cfg.top_k is not None:
        selected = ranked_idx[:cfg.top_k]
    elif cfg.threshold is not None:
        selected = np.where(utilities >= cfg.threshold)[0].tolist()
    else:
        k = int(n / max(np.log(n), 1.0))
        selected = ranked_idx[:max(1, k)]

    return QaSISResult(
        selected_indices=selected,
        utilities=utilities,
        ranked_indices=ranked_idx,
        fhat_values=fhat_vals,
        spline_designs=None
    )






# ============================
if __name__ == "__main__":
    if HAS_LIFELINES:
        Y_star = T
        event = delta.astype(int)
        y_surv = pd.DataFrame({"Y_star": Y_star, "event": event})

        cfg_surv = QaSISConfig(
            quantile=0.5,
            df_spline=4,
            degree=3,
            include_intercept=True,
            top_k=20,       
            standardize_X=True,
            is_survival=True,
            y_star_col="Y_star",
            event_col="event",
            km_epsilon=1e-3,
            weight_clip_q=0.99,
            n_jobs=4,          
            backend="loky"     
        )

        res_surv = qasis_screen(X, y_surv, cfg_surv)


        np.savetxt("Variable_index_QASIS_20.csv", np.array(res_surv.selected_indices, dtype=int), fmt="%d")

        idx_int = np.loadtxt("Variable_index_QASIS_20.csv", delimiter=",").astype(np.int32)
        if idx_int.ndim == 0:
            idx_int = np.array([int(idx_int)], dtype=np.int32) 

        X_sub = X[:, idx_int]
        np.savetxt("X_selected_QASIS_20.csv", X_sub, delimiter=",")
