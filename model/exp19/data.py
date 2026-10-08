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
    [27, 28], [28, 29], [29, 30], [30, 31], [31, 32],
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
    return sanitize_edge_list(data["edge_list"]) if "edge_list" in data else RADIAL_BRANCHES


def build_adj_norm(edge_list, num_nodes=33):
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for u, v in edge_list:
        A[u, v] = 1.0
        A[v, u] = 1.0

    A_hat = A + torch.eye(num_nodes, dtype=torch.float32)
    deg_inv_sqrt = A_hat.sum(dim=1).clamp_min(1e-8).pow(-0.5)
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

    return np.stack(
        [
            P,
            Q,
            P @ S_down.T,
            Q @ S_down.T,
            P @ S_path.T,
            Q @ S_path.T,
        ],
        axis=2,
    ).astype(np.float32)


def get_labels(data):
    if "Y_V" in data:
        YV = data["Y_V"]
    elif "Y_V_full" in data:
        YV = data["Y_V_full"]
    else:
        raise KeyError("Dataset must contain Y_V or Y_V_full.")

    if "Y_I" in data:
        YI = data["Y_I"]
    elif "Y_I_full" in data:
        YI = data["Y_I_full"]
    else:
        raise KeyError("Dataset must contain Y_I or Y_I_full.")

    return to_tensor(YV).float(), to_tensor(YI).float()


def compute_cumulative_voltage_deviation(YV):
    return torch.sum(torch.abs(YV[:, 1:] - 1.0), dim=1, keepdim=True)


def compute_worst_voltage_margin(YV, v_lower=0.95, v_upper=1.05):
    V = YV[:, 1:]
    upper_worst = torch.max(V - float(v_upper), dim=1, keepdim=True).values
    lower_worst = torch.max(float(v_lower) - V, dim=1, keepdim=True).values
    return torch.maximum(upper_worst, lower_worst)


def normalize_data(X, YV_dev, YV_worst, YI_worst, train_idx):
    # 保持与 Exp18 相同的全局逐特征标准化，避免引入新的比较变量。
    X_mean = X[train_idx].mean(dim=(0, 1), keepdim=True)
    X_std = X[train_idx].std(dim=(0, 1), keepdim=True).clamp_min(1e-6)

    def scalar_stats(y):
        mean = y[train_idx].mean(dim=0, keepdim=True)
        std = y[train_idx].std(dim=0, keepdim=True).clamp_min(1e-6)
        return mean, std, (y - mean) / std

    YV_dev_mean, YV_dev_std, YVDevn = scalar_stats(YV_dev)
    YV_worst_mean, YV_worst_std, YVWorstn = scalar_stats(YV_worst)
    YI_worst_mean, YI_worst_std, YIn = scalar_stats(YI_worst)

    norm = {
        "X_mean": X_mean.squeeze(0).squeeze(0),
        "X_std": X_std.squeeze(0).squeeze(0),
        "YV_dev_mean": YV_dev_mean.squeeze(0),
        "YV_dev_std": YV_dev_std.squeeze(0),
        "YV_worst_mean": YV_worst_mean.squeeze(0),
        "YV_worst_std": YV_worst_std.squeeze(0),
        "YI_worst_mean": YI_worst_mean.squeeze(0),
        "YI_worst_std": YI_worst_std.squeeze(0),
    }

    return (X - X_mean) / X_std, YVDevn, YVWorstn, YIn, norm

