import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


RADIAL_BRANCHES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
    [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
    [15, 16], [16, 17], [1, 18], [18, 19], [19, 20], [20, 21],
    [2, 22], [22, 23], [23, 24], [5, 25], [25, 26], [26, 27],
    [27, 28], [28, 29], [29, 30], [30, 31], [31, 32]
]


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_tensor(x):
    return x if torch.is_tensor(x) else torch.tensor(x)


def sanitize_edge_list(edge_list):
    return [[int(e[0]), int(e[1])] for e in edge_list]


def get_edge_list(data):
    if "edge_list" in data:
        return sanitize_edge_list(data["edge_list"])
    return RADIAL_BRANCHES


def build_adj_norm(edge_list, num_nodes=33):
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for u, v in edge_list:
        A[u, v] = 1.0
        A[v, u] = 1.0

    A_hat = A + torch.eye(num_nodes, dtype=torch.float32)
    deg = A_hat.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)

    return torch.diag(deg_inv_sqrt) @ A_hat @ torch.diag(deg_inv_sqrt)


def build_adj_powers(A_norm, K, include_order0=True):
    powers = []
    current = torch.eye(A_norm.size(0), dtype=torch.float32)

    if include_order0:
        powers.append(current.clone())

    for _ in range(K):
        current = current @ A_norm
        powers.append(current.clone())

    return torch.stack(powers, dim=0)


def build_topology_matrices(edge_list, num_nodes=33):
    adj = [[] for _ in range(num_nodes)]

    for u, v in edge_list:
        adj[u].append(v)
        adj[v].append(u)

    parent = np.full(num_nodes, -2, dtype=int)
    children = [[] for _ in range(num_nodes)]

    parent[0] = -1
    queue = [0]

    while queue:
        u = queue.pop(0)

        for v in adj[u]:
            if parent[v] == -2:
                parent[v] = u
                children[u].append(v)
                queue.append(v)

    S_down = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    def dfs(u):
        S_down[u, u] = 1.0

        for v in children[u]:
            dfs(v)
            S_down[u] += S_down[v]

    dfs(0)

    S_path = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i in range(num_nodes):
        cur = i

        while cur != 0 and parent[cur] >= 0:
            S_path[i] += S_down[cur]
            cur = parent[cur]

    return S_down, S_path, parent


def augment_path_power_features(X, S_down, S_path):
    P = X[:, :, 0]
    Q = X[:, :, 1]

    P_down = P @ S_down.T
    Q_down = Q @ S_down.T

    P_path = P @ S_path.T
    Q_path = Q @ S_path.T

    return np.stack(
        [
            P,
            Q,
            P_down,
            Q_down,
            P_path,
            Q_path,
        ],
        axis=2,
    ).astype(np.float32)


def get_labels(data):
    if "Y_V" in data:
        YV = data["Y_V"]
    elif "Y_V_full" in data:
        YV = data["Y_V_full"]
    else:
        raise KeyError("数据集中找不到 Y_V 或 Y_V_full。")

    if "Y_I" in data:
        YI = data["Y_I"]
    elif "Y_I_full" in data:
        YI = data["Y_I_full"]
    else:
        raise KeyError("数据集中找不到 Y_I 或 Y_I_full。")

    return to_tensor(YV).float(), to_tensor(YI).float()


def normalize_data(X, YV, YI_worst, train_idx):
    X_mean = X[train_idx].mean(dim=(0, 1), keepdim=True)
    X_std = X[train_idx].std(dim=(0, 1), keepdim=True).clamp_min(1e-6)

    YV_mean = YV[train_idx, 1:].mean(dim=0, keepdim=True)
    YV_std = YV[train_idx, 1:].std(dim=0, keepdim=True).clamp_min(1e-6)

    YI_mean = YI_worst[train_idx].mean(dim=0, keepdim=True)
    YI_std = YI_worst[train_idx].std(dim=0, keepdim=True).clamp_min(1e-6)

    Xn = (X - X_mean) / X_std

    YVn = torch.zeros_like(YV)
    YVn[:, 1:] = (YV[:, 1:] - YV_mean) / YV_std

    YIn = (YI_worst - YI_mean) / YI_std

    norm = {
        "X_mean": X_mean.squeeze(0).squeeze(0),
        "X_std": X_std.squeeze(0).squeeze(0),
        "YV_mean_wo_slack": YV_mean.squeeze(0),
        "YV_std_wo_slack": YV_std.squeeze(0),
        "YI_worst_mean": YI_mean.squeeze(0),
        "YI_worst_std": YI_std.squeeze(0),
    }

    return Xn, YVn, YIn, norm


def fit_voltage_linear_prior(Xn, YVn, train_idx, alpha=1e-3):
    Xmat = Xn[train_idx].reshape(len(train_idx), -1).double().numpy()
    Ymat = YVn[train_idx, 1:].double().numpy()

    ones = np.ones((Xmat.shape[0], 1), dtype=np.float64)
    Xa = np.concatenate([Xmat, ones], axis=1)

    reg = float(alpha) * np.eye(Xa.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0

    coef = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ Ymat)

    W = coef[:-1].T.astype(np.float32)
    b = coef[-1].astype(np.float32)

    return torch.tensor(W), torch.tensor(b)


def eval_voltage_prior(Xn, YV, W, b, norm, idx):
    Xflat = Xn[idx].reshape(len(idx), -1)
    Vn = Xflat @ W.T + b

    V = torch.ones((len(idx), 33), dtype=torch.float32)
    V[:, 1:] = Vn * norm["YV_std_wo_slack"].view(1, -1) + norm["YV_mean_wo_slack"].view(1, -1)

    err = torch.abs(V[:, 1:] - YV[idx, 1:])

    return (
        float(err.mean()),
        float(torch.sqrt(torch.mean((V[:, 1:] - YV[idx, 1:]) ** 2))),
        float(err.max()),
    )

