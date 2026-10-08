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


RADIAL_BRANCH_R_OHM = [
    0.0922, 0.4930, 0.3660, 0.3811, 0.8190, 0.1872, 0.7114, 1.0300,
    1.0440, 0.1966, 0.3744, 1.4680, 0.5416, 0.5910, 0.7463, 1.2890,
    0.3720, 0.1640, 1.5042, 0.4095, 0.7089, 0.4512, 0.8980, 0.8960,
    0.2030, 0.2842, 1.0590, 0.8042, 0.5075, 0.9744, 0.3105, 0.3410,
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


def orient_radial_branches(edge_list, num_nodes=33, root=0):
    """Orient an undirected radial edge list away from ``root``.

    The returned parent/child arrays retain the input edge order.  This is
    important because branch targets and branch latent states must use the
    same indexing convention.
    """
    edges = sanitize_edge_list(edge_list)
    if len(edges) != num_nodes - 1:
        raise ValueError(
            f"A radial network with {num_nodes} buses must have "
            f"{num_nodes - 1} branches, got {len(edges)}."
        )

    adjacency = [[] for _ in range(num_nodes)]
    for edge_id, (u, v) in enumerate(edges):
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(f"Invalid branch ({u}, {v}) for {num_nodes} buses.")
        adjacency[u].append((v, edge_id))
        adjacency[v].append((u, edge_id))

    parent = np.full(num_nodes, -2, dtype=np.int64)
    depth = np.zeros(num_nodes, dtype=np.int64)
    oriented = [None] * len(edges)
    parent[root] = -1
    queue = [int(root)]
    cursor = 0
    while cursor < len(queue):
        u = queue[cursor]
        cursor += 1
        for v, edge_id in adjacency[u]:
            if parent[v] != -2:
                continue
            parent[v] = u
            depth[v] = depth[u] + 1
            oriented[edge_id] = (u, v)
            queue.append(v)

    if len(queue) != num_nodes or any(edge is None for edge in oriented):
        raise ValueError("edge_list is disconnected or is not a radial tree.")

    parent_nodes = np.asarray([edge[0] for edge in oriented], dtype=np.int64)
    child_nodes = np.asarray([edge[1] for edge in oriented], dtype=np.int64)

    subtree_size = np.ones(num_nodes, dtype=np.float32)
    for node in reversed(queue[1:]):
        subtree_size[parent[node]] += subtree_size[node]
    child_count = np.bincount(parent_nodes, minlength=num_nodes)

    max_depth = max(int(depth.max()), 1)
    branch_topology = np.stack(
        [
            depth[child_nodes].astype(np.float32) / float(max_depth),
            subtree_size[child_nodes] / float(num_nodes),
            (child_count[child_nodes] == 0).astype(np.float32),
        ],
        axis=1,
    )
    return parent_nodes, child_nodes, parent, depth, subtree_size, branch_topology


def build_directed_adj_powers(
    edge_list,
    K,
    num_nodes=33,
    root=0,
    include_order0=True,
):
    """Build fixed child-to-parent and parent-to-child propagation powers.

    Rows are receiving buses and columns are sending buses, matching
    ``torch.einsum('kij,bjf->bkif', powers, X)``.  The channels are ordered
    as I, A_up, A_down, A_up^2, A_down^2, ... when order zero is enabled.
    """
    parent_nodes, child_nodes, parent, depth, subtree_size, branch_topology = (
        orient_radial_branches(edge_list, num_nodes, root)
    )
    A_up = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    A_down = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for u, v in zip(parent_nodes.tolist(), child_nodes.tolist()):
        A_up[u, v] = 1.0
        A_down[v, u] = 1.0

    powers = []
    if include_order0:
        powers.append(torch.eye(num_nodes, dtype=torch.float32))
    up_power = torch.eye(num_nodes, dtype=torch.float32)
    down_power = torch.eye(num_nodes, dtype=torch.float32)
    for _ in range(int(K)):
        up_power = up_power @ A_up
        down_power = down_power @ A_down
        powers.extend([up_power.clone(), down_power.clone()])

    return {
        "A_up": A_up,
        "A_down": A_down,
        "A_powers": torch.stack(powers, dim=0),
        "parent_nodes": torch.from_numpy(parent_nodes),
        "child_nodes": torch.from_numpy(child_nodes),
        "parent": parent,
        "depth": depth,
        "subtree_size": subtree_size,
        "branch_topology": torch.from_numpy(branch_topology),
    }


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


def get_line_max_i_ka(data, cfg):
    base_config = data.get("base_config", {})
    return float(base_config.get("line_max_i_ka", cfg["default_line_max_i_ka"]))


def get_branch_resistance(data, edge_list):
    if "branch_full" in data:
        branch_full = data["branch_full"]
        return torch.tensor(
            [float(row[2]) for row in branch_full],
            dtype=torch.float32,
        )

    if len(edge_list) == len(RADIAL_BRANCH_R_OHM):
        return torch.tensor(RADIAL_BRANCH_R_OHM, dtype=torch.float32)

    raise KeyError("Dataset must contain branch_full to compute total network loss.")


def compute_total_network_loss(YI_margin, branch_r_ohm, line_max_i_ka):
    current_ka = (YI_margin + 1.0).clamp_min(0.0) * float(line_max_i_ka)
    branch_r_ohm = branch_r_ohm.to(YI_margin.device).view(1, -1)
    return torch.sum(3.0 * current_ka.square() * branch_r_ohm, dim=1, keepdim=True)


def normalize_data(X, YV_dev, YV_worst, YI_worst, YP_loss, train_idx):
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
    YP_loss_mean, YP_loss_std, YPLossn = scalar_stats(YP_loss)

    norm = {
        "X_mean": X_mean.squeeze(0).squeeze(0),
        "X_std": X_std.squeeze(0).squeeze(0),
        "YV_dev_mean": YV_dev_mean.squeeze(0),
        "YV_dev_std": YV_dev_std.squeeze(0),
        "YV_worst_mean": YV_worst_mean.squeeze(0),
        "YV_worst_std": YV_worst_std.squeeze(0),
        "YI_worst_mean": YI_worst_mean.squeeze(0),
        "YI_worst_std": YI_worst_std.squeeze(0),
        "YP_loss_mean": YP_loss_mean.squeeze(0),
        "YP_loss_std": YP_loss_std.squeeze(0),
    }

    return (X - X_mean) / X_std, YVDevn, YVWorstn, YIn, YPLossn, norm

