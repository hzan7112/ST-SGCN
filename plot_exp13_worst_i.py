import os
import random
import warnings
import inspect
from collections import deque

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


RADIAL_BRANCHES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
    [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
    [15, 16], [16, 17], [1, 18], [18, 19], [19, 20], [20, 21],
    [2, 22], [22, 23], [23, 24], [5, 25], [25, 26], [26, 27],
    [27, 28], [28, 29], [29, 30], [30, 31], [31, 32]
]


def get_eval_config():
    return {
        "data_path": os.path.join("data", "ieee33_static_vvo_24h_dataset.pt"),
        "out_root": os.path.join("results", "st_sgcn_exp12_worst_i_accuracy"),
        "seed": 42,
        "batch_size": 512,
        "train_ratio": 0.8,
        "v_lower": 0.95,
        "v_upper": 1.05,

        "experiments": [
            {
                "name": "Exp13-K4-H24-n1g16-WorstI",
                "engine_path": os.path.join(
                    "checkpoints",
                    "st_sgcn_k4_h24_n1_g16_worst_i_exp12_milp_engine.pt",
                ),
            }
        ],
    }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_plot_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["axes.linewidth"] = 0.6
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def save_figure(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    tiff_path = os.path.splitext(save_path)[0] + ".tiff"

    try:
        fig.savefig(tiff_path, dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    except Exception:
        fig.savefig(tiff_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def load_pt(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def sanitize_edge_list(edge_list):
    return [[int(e[0]), int(e[1])] for e in edge_list]


def get_engine_value(engine, cfg, key, default=None):
    if key in engine:
        return engine[key]
    if key in cfg:
        return cfg[key]
    return default


def get_binary_count_text(engine):
    bc = engine.get("binary_count", {})

    if not bc:
        return "N/A"

    if isinstance(bc, dict):
        if "total_binary" in bc:
            return str(int(bc["total_binary"]))
        if "total_binary_num" in bc:
            return str(int(bc["total_binary_num"]))
        return str(bc)

    try:
        return str(int(bc))
    except Exception:
        return str(bc)


def strip_base_prefix_if_needed(state_dict):
    if not state_dict:
        return state_dict

    keys = list(state_dict.keys())

    if all(k.startswith("base.") for k in keys):
        return {k.replace("base.", "", 1): v for k, v in state_dict.items()}

    return state_dict


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


def build_topology_matrices(edge_list, num_nodes=33, root=0):
    edge_list = sanitize_edge_list(edge_list)

    adj = [[] for _ in range(num_nodes)]

    for u, v in edge_list:
        adj[u].append(v)
        adj[v].append(u)

    parent = [-2] * num_nodes
    children = [[] for _ in range(num_nodes)]
    parent[root] = -1

    q = deque([root])

    while q:
        u = q.popleft()

        for v in adj[u]:
            if parent[v] == -2:
                parent[v] = u
                children[u].append(v)
                q.append(v)

    S_down = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

    def dfs(u):
        S_down[u, u] = 1.0

        for v in children[u]:
            dfs(v)
            S_down[u] += S_down[v]

    dfs(root)

    S_path = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

    for i in range(num_nodes):
        cur = i

        while cur != root and parent[cur] >= 0:
            S_path[i] += S_down[cur]
            cur = parent[cur]

    return S_down, S_path, parent


def augment_downstream_features(X_base, S_down):
    P = X_base[:, :, 0]
    Q = X_base[:, :, 1]

    S_down = S_down.to(X_base.device)

    P_down = P @ S_down.T
    Q_down = Q @ S_down.T

    return torch.cat(
        [
            X_base,
            P_down.unsqueeze(-1),
            Q_down.unsqueeze(-1),
        ],
        dim=-1,
    )


def augment_path_features(X_base, S_down, S_path):
    P = X_base[:, :, 0]
    Q = X_base[:, :, 1]

    S_down = S_down.to(X_base.device)
    S_path = S_path.to(X_base.device)

    P_down = P @ S_down.T
    Q_down = Q @ S_down.T
    P_path = P @ S_path.T
    Q_path = Q @ S_path.T

    return torch.cat(
        [
            X_base,
            P_down.unsqueeze(-1),
            Q_down.unsqueeze(-1),
            P_path.unsqueeze(-1),
            Q_path.unsqueeze(-1),
        ],
        dim=-1,
    )


def prepare_input_for_engine(X_all, engine, edge_list):
    cfg = engine.get("config", {})

    expected = int(get_engine_value(engine, cfg, "in_features", 2))
    raw_features = X_all.shape[-1]
    num_nodes = X_all.shape[1]

    if raw_features == expected:
        return X_all

    if raw_features != 2:
        raise ValueError(f"原始数据 X 维度应为 2，当前为 {raw_features}")

    if "downstream_matrix" in engine:
        S_down = torch.as_tensor(engine["downstream_matrix"], dtype=torch.float32)
    else:
        S_down, _, _ = build_topology_matrices(edge_list, num_nodes)

    if expected == 4:
        return augment_downstream_features(X_all, S_down)

    if expected == 6:
        if "path_power_matrix" in engine:
            S_path = torch.as_tensor(engine["path_power_matrix"], dtype=torch.float32)
        else:
            _, S_path, _ = build_topology_matrices(edge_list, num_nodes)

        return augment_path_features(X_all, S_down, S_path)

    raise ValueError(f"暂不支持 engine in_features={expected}")


class STSGCNWorstI(nn.Module):
    def __init__(
        self,
        in_features=6,
        hidden_dim=24,
        K=4,
        edge_list=None,
        num_nodes=33,
        node_relu_dim=1,
        global_relu_dim=32,
        include_order0=True,
        use_sgc_relu=False,
        use_linear_skip=True,
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.hidden_dim = int(hidden_dim)
        self.K = int(K)
        self.num_nodes = int(num_nodes)
        self.node_relu_dim = int(node_relu_dim)
        self.global_relu_dim = int(global_relu_dim)
        self.include_order0 = bool(include_order0)
        self.use_sgc_relu = bool(use_sgc_relu)
        self.use_linear_skip = bool(use_linear_skip)

        edge_list = sanitize_edge_list(edge_list if edge_list is not None else RADIAL_BRANCHES)
        self.edge_list = edge_list

        A_norm = build_adj_norm(edge_list, self.num_nodes)
        A_powers = build_adj_powers(A_norm, self.K, self.include_order0)

        self.register_buffer("A_norm", A_norm)
        self.register_buffer("A_powers", A_powers)

        self.order_count = A_powers.size(0)

        self.sgc_linear = nn.Linear(self.order_count * self.in_features, self.hidden_dim)

        self.node_emb_dim = 4
        self.node_emb = nn.Embedding(self.num_nodes, self.node_emb_dim)

        node_input_dim = self.hidden_dim + self.node_emb_dim

        if self.node_relu_dim > 0:
            self.node_hidden = nn.Linear(node_input_dim, self.node_relu_dim)
            self.node_out = nn.Linear(self.node_relu_dim, 1)
        else:
            self.node_hidden = None
            self.node_out = None

        self.node_skip = nn.Linear(node_input_dim, 1) if self.use_linear_skip else None

        if self.node_relu_dim == 0 and self.node_skip is None:
            self.node_linear = nn.Linear(node_input_dim, 1)
        else:
            self.node_linear = None

        global_input_dim = self.num_nodes * self.hidden_dim + self.num_nodes * self.in_features

        self.global_hidden = nn.Linear(global_input_dim, self.global_relu_dim)
        self.global_out = nn.Linear(self.global_relu_dim, 1)
        self.global_skip = nn.Linear(global_input_dim, 1) if self.use_linear_skip else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.sgc_linear.weight)
        nn.init.zeros_(self.sgc_linear.bias)

        nn.init.normal_(self.node_emb.weight, mean=0.0, std=0.02)

        if self.node_hidden is not None:
            nn.init.xavier_uniform_(self.node_hidden.weight)
            nn.init.zeros_(self.node_hidden.bias)

        if self.node_out is not None:
            nn.init.xavier_uniform_(self.node_out.weight)
            nn.init.zeros_(self.node_out.bias)

        if self.node_skip is not None:
            nn.init.xavier_uniform_(self.node_skip.weight)
            nn.init.zeros_(self.node_skip.bias)

        if self.node_linear is not None:
            nn.init.xavier_uniform_(self.node_linear.weight)
            nn.init.zeros_(self.node_linear.bias)

        nn.init.xavier_uniform_(self.global_hidden.weight)
        nn.init.zeros_(self.global_hidden.bias)

        nn.init.xavier_uniform_(self.global_out.weight)
        nn.init.zeros_(self.global_out.bias)

        if self.global_skip is not None:
            nn.init.xavier_uniform_(self.global_skip.weight)
            nn.init.zeros_(self.global_skip.bias)

    def forward(self, X):
        batch_size = X.size(0)
        device = X.device

        A_powers = self.A_powers.to(device)

        X_orders = torch.einsum("kij,bjf->bkif", A_powers, X)
        X_multi = X_orders.permute(0, 2, 1, 3).reshape(
            batch_size,
            self.num_nodes,
            self.order_count * self.in_features,
        )

        Z_sgc = self.sgc_linear(X_multi)

        if self.use_sgc_relu:
            H = F.relu(Z_sgc)
            gcn_Z_list = [Z_sgc]
        else:
            H = Z_sgc
            gcn_Z_list = []

        node_ids = torch.arange(self.num_nodes, device=device)
        node_emb = self.node_emb(node_ids).unsqueeze(0).expand(batch_size, -1, -1)

        H_node = torch.cat([H, node_emb], dim=-1)

        if self.node_relu_dim > 0:
            Z_node = self.node_hidden(H_node)
            V_pred = self.node_out(F.relu(Z_node)).squeeze(-1)

            if self.node_skip is not None:
                V_pred = V_pred + self.node_skip(H_node).squeeze(-1)
        else:
            Z_node = H_node.new_zeros(batch_size, self.num_nodes, 0)

            if self.node_skip is not None:
                V_pred = self.node_skip(H_node).squeeze(-1)
            elif self.node_linear is not None:
                V_pred = self.node_linear(H_node).squeeze(-1)
            else:
                V_pred = H_node.new_zeros(batch_size, self.num_nodes)

        G = torch.cat(
            [
                H.reshape(batch_size, -1),
                X.reshape(batch_size, -1),
            ],
            dim=-1,
        )

        Z_global = self.global_hidden(G)
        YI_worst_pred = self.global_out(F.relu(Z_global))

        if self.global_skip is not None:
            YI_worst_pred = YI_worst_pred + self.global_skip(G)

        return V_pred, YI_worst_pred, gcn_Z_list, Z_node, Z_global


def build_model_from_engine(engine, device):
    cfg = engine.get("config", {})
    edge_list = sanitize_edge_list(engine["edge_list"])

    in_features = int(get_engine_value(engine, cfg, "in_features", 6))
    num_nodes = int(get_engine_value(engine, cfg, "num_nodes", 33))

    K = int(get_engine_value(engine, cfg, "K", 4))

    model = STSGCNWorstI(
        in_features=in_features,
        hidden_dim=int(get_engine_value(engine, cfg, "hidden_dim", 24)),
        K=K,
        edge_list=edge_list,
        num_nodes=num_nodes,
        node_relu_dim=int(get_engine_value(engine, cfg, "node_relu_dim", 1)),
        global_relu_dim=int(get_engine_value(engine, cfg, "global_relu_dim", 32)),
        include_order0=bool(get_engine_value(engine, cfg, "include_order0", True)),
        use_sgc_relu=bool(get_engine_value(engine, cfg, "use_sgc_relu", False)),
        use_linear_skip=bool(get_engine_value(engine, cfg, "use_linear_skip", True)),
    ).to(device)

    return model


def load_engine_and_model(engine_path, device):
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"未找到 engine 文件: {engine_path}")

    engine = load_pt(engine_path, map_location=device)
    model = build_model_from_engine(engine, device)

    state_dict = engine.get("state_dict", None)

    if state_dict is None:
        state_dict = engine.get("base_state_dict", None)

    if state_dict is None and "wrapper_state_dict" in engine:
        state_dict = strip_base_prefix_if_needed(engine["wrapper_state_dict"])

    if state_dict is None:
        raise KeyError("engine 中未找到 state_dict、base_state_dict 或 wrapper_state_dict。")

    state_dict = strip_base_prefix_if_needed(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print("\n[警告] strict=True 加载失败，尝试 strict=False。")
        print(str(e))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {missing}")
        print(f"unexpected keys: {unexpected}")

    model.eval()

    return engine, model


def load_norm_stats(engine, device):
    ns = engine["norm_stats"]

    X_mean = torch.as_tensor(ns["X_mean"], dtype=torch.float32, device=device)
    X_std = torch.as_tensor(ns["X_std"], dtype=torch.float32, device=device)

    if "YV_mean_wo_slack" in ns:
        YV_mean = torch.as_tensor(ns["YV_mean_wo_slack"], dtype=torch.float32, device=device)
        YV_std = torch.as_tensor(ns["YV_std_wo_slack"], dtype=torch.float32, device=device)
    else:
        YV_mean = torch.as_tensor(ns["YV_mean"], dtype=torch.float32, device=device)
        YV_std = torch.as_tensor(ns["YV_std"], dtype=torch.float32, device=device)

    if "YI_worst_mean" in ns:
        YI_mean = torch.as_tensor(ns["YI_worst_mean"], dtype=torch.float32, device=device)
        YI_std = torch.as_tensor(ns["YI_worst_std"], dtype=torch.float32, device=device)
    else:
        YI_mean = torch.as_tensor(ns["YI_mean"], dtype=torch.float32, device=device)
        YI_std = torch.as_tensor(ns["YI_std"], dtype=torch.float32, device=device)

    return X_mean, X_std, YV_mean, YV_std, YI_mean, YI_std


def add_voltage_linear_prior_if_needed(engine, X_norm, V_pred_n):
    if not bool(engine.get("uses_voltage_linear_prior", False)):
        return V_pred_n

    W = torch.as_tensor(engine["voltage_linear_prior_W"], dtype=torch.float32, device=X_norm.device)
    b = torch.as_tensor(engine["voltage_linear_prior_b"], dtype=torch.float32, device=X_norm.device)

    V_lin_n = X_norm.reshape(X_norm.shape[0], -1) @ W.T + b

    if V_pred_n.shape[1] == 33:
        V_total = V_pred_n.clone()
        V_total[:, 1:] = V_lin_n + V_pred_n[:, 1:]
        return V_total

    return V_lin_n + V_pred_n


def add_current_linear_prior_if_needed(engine, X_norm, I_pred_n):
    if not bool(engine.get("uses_current_linear_prior", False)):
        return I_pred_n

    W = torch.as_tensor(engine["current_linear_prior_W"], dtype=torch.float32, device=X_norm.device)
    b = torch.as_tensor(engine["current_linear_prior_b"], dtype=torch.float32, device=X_norm.device)

    I_lin_n = X_norm.reshape(X_norm.shape[0], -1) @ W.T + b
    return I_lin_n + I_pred_n


def get_validation_indices(num_samples, train_ratio=0.8, seed=42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=g)

    return perm[int(train_ratio * num_samples):].tolist()


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    err = y_pred - y_true
    abs_err = np.abs(err)

    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12

    return {
        "MAE": float(np.mean(abs_err)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(abs_err / (np.abs(y_true) + 1e-8)) * 100.0),
        "R2": float(1.0 - ss_res / ss_tot),
        "P95": float(np.percentile(abs_err, 95)),
        "P99": float(np.percentile(abs_err, 99)),
        "MaxError": float(np.max(abs_err)),
    }


def compute_margin_safety_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    true_violate = y_true > 0.0
    pred_violate = y_pred > 0.0

    false_safe = true_violate & (~pred_violate)
    false_violate = (~true_violate) & pred_violate

    true_violate_count = int(np.sum(true_violate))
    true_safe_count = int(np.sum(~true_violate))

    return {
        "TrueViolateCount": true_violate_count,
        "FalseSafeCount": int(np.sum(false_safe)),
        "FalseSafeRate": float(np.sum(false_safe) / true_violate_count) if true_violate_count > 0 else 0.0,
        "FalseViolateCount": int(np.sum(false_violate)),
        "FalseViolateRate": float(np.sum(false_violate) / true_safe_count) if true_safe_count > 0 else 0.0,
    }


def compute_voltage_safety_metrics(v_true, v_pred, v_lower=0.95, v_upper=1.05):
    v_true = np.asarray(v_true)
    v_pred = np.asarray(v_pred)

    true_low = v_true < v_lower
    true_high = v_true > v_upper
    true_violate = true_low | true_high

    pred_low = v_pred < v_lower
    pred_high = v_pred > v_upper
    pred_violate = pred_low | pred_high

    false_safe = true_violate & (~pred_violate)
    false_low_safe = true_low & (~pred_low)
    false_high_safe = true_high & (~pred_high)
    false_violate = (~true_violate) & pred_violate

    true_violate_count = int(np.sum(true_violate))
    true_safe_count = int(np.sum(~true_violate))
    true_low_count = int(np.sum(true_low))
    true_high_count = int(np.sum(true_high))

    return {
        "TrueViolateCount": true_violate_count,
        "FalseSafeCount": int(np.sum(false_safe)),
        "FalseSafeRate": float(np.sum(false_safe) / true_violate_count) if true_violate_count > 0 else 0.0,
        "FalseViolateCount": int(np.sum(false_violate)),
        "FalseViolateRate": float(np.sum(false_violate) / true_safe_count) if true_safe_count > 0 else 0.0,
        "TrueLowCount": true_low_count,
        "FalseLowSafeCount": int(np.sum(false_low_safe)),
        "FalseLowSafeRate": float(np.sum(false_low_safe) / true_low_count) if true_low_count > 0 else 0.0,
        "TrueHighCount": true_high_count,
        "FalseHighSafeCount": int(np.sum(false_high_safe)),
        "FalseHighSafeRate": float(np.sum(false_high_safe) / true_high_count) if true_high_count > 0 else 0.0,
    }


def predict_on_validation_set(model, engine, X_raw, Y_V_raw, Y_I_branch_raw, val_indices, norm_stats, device, batch_size=512):
    X_mean, X_std, YV_mean, YV_std, YI_mean, YI_std = norm_stats

    X_val = X_raw[val_indices].to(device)
    YV_val = Y_V_raw[val_indices].to(device)
    YI_branch_val = Y_I_branch_raw[val_indices].to(device)

    YI_worst_val = YI_branch_val.max(dim=1, keepdim=True).values

    if X_val.shape[-1] != X_mean.shape[-1]:
        raise ValueError(f"输入维度不匹配：X_val={X_val.shape}, X_mean={X_mean.shape}")

    X_norm = (X_val - X_mean) / X_std

    V_list = []
    I_list = []

    with torch.no_grad():
        for s in range(0, X_norm.shape[0], batch_size):
            e = min(s + batch_size, X_norm.shape[0])

            out = model(X_norm[s:e])

            V_pred_n = out[0]
            I_pred_n = out[1]

            V_pred_n = add_voltage_linear_prior_if_needed(engine, X_norm[s:e], V_pred_n)
            I_pred_n = add_current_linear_prior_if_needed(engine, X_norm[s:e], I_pred_n)

            if V_pred_n.shape[1] == 33:
                V_phys = V_pred_n[:, 1:] * YV_std + YV_mean
            else:
                V_phys = V_pred_n * YV_std + YV_mean

            I_phys = I_pred_n * YI_std + YI_mean

            V_list.append(V_phys.detach().cpu())
            I_list.append(I_phys.detach().cpu())

    V_pred = torch.cat(V_list, dim=0).numpy()
    I_pred = torch.cat(I_list, dim=0).numpy()

    V_true = YV_val[:, 1:].detach().cpu().numpy()
    I_true = YI_worst_val.detach().cpu().numpy()

    return V_true, V_pred, I_true, I_pred


def plot_voltage_accuracy(V_true, V_pred, out_dir, v_lower=0.95, v_upper=1.05):
    err = V_pred - V_true
    metrics = compute_metrics(V_true, V_pred)
    safety = compute_voltage_safety_metrics(V_true, V_pred, v_lower=v_lower, v_upper=v_upper)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    ax = axes[0, 0]
    node_id = 30
    node_col = max(0, min(node_id - 2, V_true.shape[1] - 1))
    show_len = min(300, V_true.shape[0])

    ax.plot(V_true[:show_len, node_col], label="True", linewidth=1.2)
    ax.plot(V_pred[:show_len, node_col], label="Predicted", linewidth=1.2, alpha=0.85)
    ax.axhline(v_lower, linestyle="--", linewidth=1.0)
    ax.axhline(v_upper, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title(f"Voltage sequence at Bus {node_id}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.12, linewidth=0.4)

    ax = axes[0, 1]
    ax.scatter(V_true.reshape(-1), V_pred.reshape(-1), s=6, alpha=0.30, edgecolors="none")

    v_min = min(V_true.min(), V_pred.min())
    v_max = max(V_true.max(), V_pred.max())

    ax.plot([v_min, v_max], [v_min, v_max], linestyle="--", linewidth=1.0)
    ax.axvline(v_lower, linestyle=":", linewidth=0.8)
    ax.axhline(v_lower, linestyle=":", linewidth=0.8)
    ax.axvline(v_upper, linestyle=":", linewidth=0.8)
    ax.axhline(v_upper, linestyle=":", linewidth=0.8)
    ax.set_xlabel("True voltage (p.u.)")
    ax.set_ylabel("Predicted voltage (p.u.)")
    ax.set_title("Voltage mapping")
    ax.grid(alpha=0.12, linewidth=0.4)

    ax = axes[1, 0]
    ax.hist(err.reshape(-1), bins=70, alpha=0.85)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Prediction error (p.u.)")
    ax.set_ylabel("Frequency")
    ax.set_title("Voltage error distribution")
    ax.grid(alpha=0.12, linewidth=0.4)

    ax = axes[1, 1]
    text = (
        "Voltage prediction metrics\n"
        f"MAE = {metrics['MAE']:.6f} p.u.\n"
        f"RMSE = {metrics['RMSE']:.6f} p.u.\n"
        f"P95 = {metrics['P95']:.6f} p.u.\n"
        f"P99 = {metrics['P99']:.6f} p.u.\n"
        f"MaxError = {metrics['MaxError']:.6f} p.u.\n"
        f"R2 = {metrics['R2']:.6f}\n\n"
        "Voltage safety metrics\n"
        f"False Safe = {safety['FalseSafeRate'] * 100:.2f}%\n"
        f"False Violate = {safety['FalseViolateRate'] * 100:.2f}%\n"
        f"Low-V False Safe = {safety['FalseLowSafeRate'] * 100:.2f}%\n"
        f"High-V False Safe = {safety['FalseHighSafeRate'] * 100:.2f}%\n"
        f"True violated points = {safety['TrueViolateCount']}"
    )

    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=10)
    ax.axis("off")

    save_figure(fig, os.path.join(out_dir, "fig1_voltage_accuracy.png"))


def plot_worst_i_accuracy(I_true, I_pred, out_dir):
    err = I_pred - I_true
    metrics = compute_metrics(I_true, I_pred)
    safety = compute_margin_safety_metrics(I_true, I_pred)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    ax = axes[0, 0]
    show_len = min(500, I_true.reshape(-1).shape[0])

    ax.plot(I_true.reshape(-1)[:show_len], label="True", linewidth=1.2)
    ax.plot(I_pred.reshape(-1)[:show_len], label="Predicted", linewidth=1.2, alpha=0.85)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Sample")
    ax.set_ylabel(r"$Y_I^{worst}$")
    ax.set_title(r"System-level worst current margin")
    ax.legend(frameon=False)
    ax.grid(alpha=0.12, linewidth=0.4)

    ax = axes[0, 1]
    ax.scatter(I_true.reshape(-1), I_pred.reshape(-1), s=8, alpha=0.35, edgecolors="none")

    i_min = min(I_true.min(), I_pred.min())
    i_max = max(I_true.max(), I_pred.max())

    ax.plot([i_min, i_max], [i_min, i_max], linestyle="--", linewidth=1.0)
    ax.axvline(0.0, linestyle=":", linewidth=0.8)
    ax.axhline(0.0, linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"True $Y_I^{worst}$")
    ax.set_ylabel(r"Predicted $Y_I^{worst}$")
    ax.set_title(r"Worst current-margin mapping")
    ax.grid(alpha=0.12, linewidth=0.4)

    ax = axes[1, 0]
    ax.hist(err.reshape(-1), bins=70, alpha=0.85)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Frequency")
    ax.set_title(r"$Y_I^{worst}$ error distribution")
    ax.grid(alpha=0.12, linewidth=0.4)

    ax = axes[1, 1]
    text = (
        "Worst current-margin prediction\n"
        f"MAE = {metrics['MAE']:.6f}\n"
        f"RMSE = {metrics['RMSE']:.6f}\n"
        f"P95 = {metrics['P95']:.6f}\n"
        f"P99 = {metrics['P99']:.6f}\n"
        f"MaxError = {metrics['MaxError']:.6f}\n"
        f"R2 = {metrics['R2']:.6f}\n\n"
        "Safety classification metrics\n"
        f"False Safe = {safety['FalseSafeRate'] * 100:.2f}%\n"
        f"False Violate = {safety['FalseViolateRate'] * 100:.2f}%\n"
        f"True violated samples = {safety['TrueViolateCount']}\n"
        f"False-safe samples = {safety['FalseSafeCount']}"
    )

    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=10)
    ax.axis("off")

    save_figure(fig, os.path.join(out_dir, "fig2_worst_current_margin_accuracy.png"))


def plot_nodewise_voltage_error(V_true, V_pred, out_dir):
    abs_err = np.abs(V_pred - V_true)

    mae = abs_err.mean(axis=0)
    p95 = np.percentile(abs_err, 95, axis=0)
    p99 = np.percentile(abs_err, 99, axis=0)
    mx = abs_err.max(axis=0)

    bus_ids = np.arange(2, 34)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(bus_ids, mae, marker="o", markersize=3, linewidth=1.2, label="MAE")
    ax.plot(bus_ids, p95, marker="s", markersize=3, linewidth=1.2, label="P95")
    ax.plot(bus_ids, p99, marker="^", markersize=3, linewidth=1.2, label="P99")
    ax.set_xlabel("Bus")
    ax.set_ylabel("Voltage absolute error (p.u.)")
    ax.set_title("Node-wise voltage prediction error")
    ax.legend(frameon=False)
    ax.grid(alpha=0.12, linewidth=0.4)

    save_figure(fig, os.path.join(out_dir, "fig3_nodewise_voltage_error.png"))

    with open(os.path.join(out_dir, "nodewise_voltage_error.csv"), "w", encoding="utf-8") as f:
        f.write("Bus,MAE,P95,P99,MaxError\n")

        for bus, a, b, c, d in zip(bus_ids, mae, p95, p99, mx):
            f.write(f"{bus},{a:.8f},{b:.8f},{c:.8f},{d:.8f}\n")


def save_summary_report(V_true, V_pred, I_true, I_pred, engine, out_dir, v_lower=0.95, v_upper=1.05):
    cfg = engine.get("config", {})

    v_metrics = compute_metrics(V_true, V_pred)
    i_metrics = compute_metrics(I_true, I_pred)

    v_safety = compute_voltage_safety_metrics(V_true, V_pred, v_lower=v_lower, v_upper=v_upper)
    i_safety = compute_margin_safety_metrics(I_true, I_pred)

    report_path = os.path.join(out_dir, "accuracy_summary.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("ST-SGCN Exp13 Worst-I accuracy evaluation summary\n")
        f.write("============================================================\n\n")

        f.write(f"Model type: {engine.get('model_type', 'N/A')}\n")
        f.write(f"Base model class: {engine.get('base_model_class', 'ST-SGCN-WorstI')}\n")
        f.write(f"K: {get_engine_value(engine, cfg, 'K', 'N/A')}\n")
        f.write(f"Hidden dim: {get_engine_value(engine, cfg, 'hidden_dim', 'N/A')}\n")
        f.write(f"Node head dim: {get_engine_value(engine, cfg, 'node_relu_dim', 'N/A')}\n")
        f.write(f"Global head dim: {get_engine_value(engine, cfg, 'global_relu_dim', 'N/A')}\n")
        f.write(f"Include order0: {get_engine_value(engine, cfg, 'include_order0', 'N/A')}\n")
        f.write(f"Use SGC ReLU: {get_engine_value(engine, cfg, 'use_sgc_relu', 'N/A')}\n")
        f.write(f"Input features: {engine.get('feature_names', engine.get('input_feature_names', 'N/A'))}\n")
        f.write(f"Current target: {engine.get('predict_current_target', 'YI_worst=max(Y_I_branch)')}\n")
        f.write(f"Uses voltage linear prior: {engine.get('uses_voltage_linear_prior', False)}\n")
        f.write(f"Uses current linear prior: {engine.get('uses_current_linear_prior', False)}\n")
        f.write(f"Total ReLU binary variables: {get_binary_count_text(engine)}\n\n")

        f.write("[Voltage prediction]\n")

        for k, v in v_metrics.items():
            f.write(f"{k}: {v:.8f}\n")

        f.write("\n[Voltage safety]\n")

        for k, v in v_safety.items():
            if "Rate" in k:
                f.write(f"{k}: {v * 100:.4f}%\n")
            else:
                f.write(f"{k}: {v}\n")

        f.write("\n[Worst current-margin prediction]\n")

        for k, v in i_metrics.items():
            f.write(f"{k}: {v:.8f}\n")

        f.write("\n[Worst current-margin safety]\n")

        for k, v in i_safety.items():
            if "Rate" in k:
                f.write(f"{k}: {v * 100:.4f}%\n")
            else:
                f.write(f"{k}: {v}\n")

    return report_path


def evaluate_one_experiment(exp, data, device, eval_cfg):
    base_dir = os.getcwd()
    engine_path = os.path.join(base_dir, exp["engine_path"])

    if not os.path.exists(engine_path):
        print(f"\n[跳过] 未找到 engine: {engine_path}")
        return None

    engine, model = load_engine_and_model(engine_path, device)
    cfg = engine.get("config", {})

    edge_list = sanitize_edge_list(engine["edge_list"])

    X_base = data["X"].float()
    Y_V = data["Y_V"].float() if "Y_V" in data else data["Y_V_full"].float()
    Y_I_branch = data["Y_I"].float() if "Y_I" in data else data["Y_I_full"].float()

    X_eval = prepare_input_for_engine(X_base, engine, edge_list)
    norm_stats = load_norm_stats(engine, device)

    train_ratio = get_engine_value(engine, cfg, "train_ratio", eval_cfg["train_ratio"])
    seed = get_engine_value(engine, cfg, "seed", eval_cfg["seed"])
    v_lower = get_engine_value(engine, cfg, "v_lower", eval_cfg["v_lower"])
    v_upper = get_engine_value(engine, cfg, "v_upper", eval_cfg["v_upper"])

    val_indices = get_validation_indices(len(X_eval), train_ratio=train_ratio, seed=seed)

    print("\n============================================================")
    print(f"开始评估: {exp['name']}")
    print("============================================================")
    print(f"engine_path = {engine_path}")
    print(f"X_base = {tuple(X_base.shape)}, X_eval = {tuple(X_eval.shape)}")
    print(f"model_type = {engine.get('model_type', 'N/A')}")
    print(f"base_model_class = {engine.get('base_model_class', 'ST-SGCN-WorstI')}")
    print(f"in_features = {get_engine_value(engine, cfg, 'in_features', 'N/A')}")
    print(f"K = {get_engine_value(engine, cfg, 'K', 'N/A')}")
    print(f"hidden_dim = {get_engine_value(engine, cfg, 'hidden_dim', 'N/A')}")
    print(f"node_relu_dim = {get_engine_value(engine, cfg, 'node_relu_dim', 'N/A')}")
    print(f"global_relu_dim = {get_engine_value(engine, cfg, 'global_relu_dim', 'N/A')}")
    print(f"include_order0 = {get_engine_value(engine, cfg, 'include_order0', 'N/A')}")
    print(f"use_sgc_relu = {get_engine_value(engine, cfg, 'use_sgc_relu', 'N/A')}")
    print(f"current target = {engine.get('predict_current_target', 'YI_worst=max(Y_I_branch)')}")
    print(f"uses_voltage_linear_prior = {engine.get('uses_voltage_linear_prior', False)}")
    print(f"uses_current_linear_prior = {engine.get('uses_current_linear_prior', False)}")
    print(f"binary_count = {get_binary_count_text(engine)}")
    print(f"验证集样本数 = {len(val_indices)}")

    V_true, V_pred, I_true, I_pred = predict_on_validation_set(
        model=model,
        engine=engine,
        X_raw=X_eval,
        Y_V_raw=Y_V,
        Y_I_branch_raw=Y_I_branch,
        val_indices=val_indices,
        norm_stats=norm_stats,
        device=device,
        batch_size=eval_cfg["batch_size"],
    )

    v_metrics = compute_metrics(V_true, V_pred)
    i_metrics = compute_metrics(I_true, I_pred)

    v_safety = compute_voltage_safety_metrics(V_true, V_pred, v_lower=v_lower, v_upper=v_upper)
    i_safety = compute_margin_safety_metrics(I_true, I_pred)

    out_dir = os.path.join(base_dir, eval_cfg["out_root"], exp["name"])
    ensure_dir(out_dir)

    plot_voltage_accuracy(V_true, V_pred, out_dir, v_lower, v_upper)
    plot_worst_i_accuracy(I_true, I_pred, out_dir)
    plot_nodewise_voltage_error(V_true, V_pred, out_dir)

    report_path = save_summary_report(V_true, V_pred, I_true, I_pred, engine, out_dir, v_lower, v_upper)

    np.savez(
        os.path.join(out_dir, "accuracy_arrays.npz"),
        V_true=V_true,
        V_pred=V_pred,
        YI_worst_true=I_true,
        YI_worst_pred=I_pred,
        val_indices=np.array(val_indices),
    )

    print("[节点电压预测]")
    print(
        f"MAE={v_metrics['MAE']:.6f}, "
        f"RMSE={v_metrics['RMSE']:.6f}, "
        f"P95={v_metrics['P95']:.6f}, "
        f"P99={v_metrics['P99']:.6f}, "
        f"Max={v_metrics['MaxError']:.6f}"
    )
    print(f"V-FS={v_safety['FalseSafeRate'] * 100:.2f}%, V-FV={v_safety['FalseViolateRate'] * 100:.2f}%")

    print("[全网最危险电流裕度预测]")
    print(
        f"MAE={i_metrics['MAE']:.6f}, "
        f"RMSE={i_metrics['RMSE']:.6f}, "
        f"P95={i_metrics['P95']:.6f}, "
        f"P99={i_metrics['P99']:.6f}, "
        f"Max={i_metrics['MaxError']:.6f}"
    )
    print(f"WorstI-FS={i_safety['FalseSafeRate'] * 100:.2f}%, WorstI-FV={i_safety['FalseViolateRate'] * 100:.2f}%")
    print(f"输出目录: {out_dir}")
    print(f"报告: {report_path}")

    return {
        "name": exp["name"],
        "engine_path": engine_path,
        "model_type": engine.get("model_type", "N/A"),
        "base_model_class": engine.get("base_model_class", "ST-SGCN-WorstI"),
        "in_features": get_engine_value(engine, cfg, "in_features", "N/A"),
        "K": get_engine_value(engine, cfg, "K", "N/A"),
        "hidden_dim": get_engine_value(engine, cfg, "hidden_dim", "N/A"),
        "node_relu_dim": get_engine_value(engine, cfg, "node_relu_dim", "N/A"),
        "global_relu_dim": get_engine_value(engine, cfg, "global_relu_dim", "N/A"),
        "binary": int(get_binary_count_text(engine)) if get_binary_count_text(engine).isdigit() else np.nan,
        "uses_vlin": bool(engine.get("uses_voltage_linear_prior", False)),
        "uses_ilin": bool(engine.get("uses_current_linear_prior", False)),

        "V_MAE": v_metrics["MAE"],
        "V_RMSE": v_metrics["RMSE"],
        "V_P95": v_metrics["P95"],
        "V_P99": v_metrics["P99"],
        "V_MaxErr": v_metrics["MaxError"],
        "V_R2": v_metrics["R2"],
        "V_FS": v_safety["FalseSafeRate"] * 100,
        "V_FV": v_safety["FalseViolateRate"] * 100,

        "WorstI_MAE": i_metrics["MAE"],
        "WorstI_RMSE": i_metrics["RMSE"],
        "WorstI_P95": i_metrics["P95"],
        "WorstI_P99": i_metrics["P99"],
        "WorstI_MaxErr": i_metrics["MaxError"],
        "WorstI_R2": i_metrics["R2"],
        "WorstI_FS": i_safety["FalseSafeRate"] * 100,
        "WorstI_FV": i_safety["FalseViolateRate"] * 100,
    }


def save_comparison_csv(rows, out_root):
    csv_path = os.path.join(out_root, "st_sgcn_exp13_worst_i_metrics.csv")

    keys = [
        "name",
        "binary",
        "K",
        "hidden_dim",
        "node_relu_dim",
        "global_relu_dim",
        "in_features",
        "uses_vlin",
        "uses_ilin",
        "V_MAE",
        "V_RMSE",
        "V_P95",
        "V_P99",
        "V_MaxErr",
        "V_FS",
        "V_FV",
        "WorstI_MAE",
        "WorstI_RMSE",
        "WorstI_P95",
        "WorstI_P99",
        "WorstI_MaxErr",
        "WorstI_FS",
        "WorstI_FV",
    ]

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")

        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    return csv_path


def plot_comparison(rows, out_root):
    if len(rows) == 0:
        return

    names = [r["name"] for r in rows]
    x = np.arange(len(names))

    metrics = [
        ("V_MAE", "Voltage MAE (p.u.)"),
        ("V_RMSE", "Voltage RMSE (p.u.)"),
        ("V_FS", "Voltage false-safe rate (%)"),
        ("WorstI_MAE", r"$Y_I^{worst}$ MAE"),
        ("WorstI_RMSE", r"$Y_I^{worst}$ RMSE"),
        ("WorstI_FS", r"$Y_I^{worst}$ false-safe rate (%)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    axes = axes.reshape(-1)

    for ax, (key, title) in zip(axes, metrics):
        vals = [r[key] for r in rows]

        ax.bar(x, vals, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.12, linewidth=0.4)

        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4g}", ha="center", va="bottom", fontsize=8)

    save_figure(fig, os.path.join(out_root, "fig0_st_sgcn_exp13_worst_i_comparison.png"))


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    eval_cfg = get_eval_config()
    set_seed(eval_cfg["seed"])
    set_plot_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.getcwd()

    data_path = os.path.join(base_dir, eval_cfg["data_path"])
    out_root = os.path.join(base_dir, eval_cfg["out_root"])

    ensure_dir(out_root)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到数据集文件: {data_path}")

    data = load_pt(data_path, map_location="cpu")

    print(f"当前验证设备: {device}")
    print(f"数据集: {data_path}")
    print(f"输出目录: {out_root}")

    rows = []

    for exp in eval_cfg["experiments"]:
        row = evaluate_one_experiment(exp, data, device, eval_cfg)

        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("没有任何 Exp13 engine 被成功评估，请检查 get_eval_config() 中的路径。")

    csv_path = save_comparison_csv(rows, out_root)
    plot_comparison(rows, out_root)

    print("\n============================================================")
    print("ST-SGCN Exp13 Worst-I 绘图与评估完成")
    print("============================================================")
    print(f"对比指标表: {csv_path}")
    print(f"对比图: {os.path.join(out_root, 'fig0_st_sgcn_exp13_worst_i_comparison.png')}")
    print("实验子目录中包含电压预测图、全网最危险电流裕度预测图、逐节点电压误差图和文本报告。")


if __name__ == "__main__":
    main()