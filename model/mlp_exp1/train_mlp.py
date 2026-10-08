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


class MLPExp17Outputs(nn.Module):
    """Pure raw-PQ MLP that predicts Exp17-style V_nodes and YI_worst."""

    def __init__(
        self,
        input_dim=66,
        hidden_dims=(32, 32, 32),
        output_dim=34,
        dropout=0.0,
        batchnorm=False,
    ):
        super().__init__()
        if output_dim != 34:
            raise ValueError(f"output_dim must be 34 = 33 voltages + 1 YI_worst, got {output_dim}")

        self.input_dim = int(input_dim)
        self.hidden_dims = [int(dim) for dim in hidden_dims]
        self.output_dim = int(output_dim)
        self.dropout = float(dropout)
        self.batchnorm = bool(batchnorm)

        dims = [self.input_dim] + self.hidden_dims
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(self.hidden_dims))]
        )
        self.batchnorm_layers = (
            nn.ModuleList([nn.BatchNorm1d(dim) for dim in self.hidden_dims])
            if self.batchnorm
            else None
        )
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, X):
        H = X.reshape(X.size(0), -1)
        if H.size(1) != self.input_dim:
            raise ValueError(f"expected flattened input dimension {self.input_dim}, got {H.size(1)}")

        hidden_Z_list = []
        for layer_idx, layer in enumerate(self.hidden_layers):
            Z = layer(H)
            hidden_Z_list.append(Z)
            if self.batchnorm_layers is not None:
                Z = self.batchnorm_layers[layer_idx](Z)
            H = F.relu(Z)
            if self.dropout > 0.0:
                H = F.dropout(H, p=self.dropout, training=self.training)

        out = self.output_layer(H)
        Vn = out[:, :33]
        YI_worst_n = out[:, 33:34]
        return Vn, YI_worst_n, hidden_Z_list

    def get_binary_count(self):
        return {
            "mlp_hidden_binary_per_layer": list(self.hidden_dims),
            "mlp_binary": sum(self.hidden_dims),
            "total_binary": sum(self.hidden_dims),
            "predict_target": "V_nodes_and_YI_worst",
        }


def get_config():
    return {
        "exp_name": "MLP_Exp1_Exp17Outputs_RawPQ",
        "data_path": r"data/ieee33_nodal_pq_correlated_raw_pool_50k.pt",
        "save_dir": r"checkpoints",
        "best_model_name": "mlp_exp1_exp17_outputs_rawpq_best.pt",
        "engine_name": "mlp_exp1_exp17_outputs_rawpq_milp_engine.pt",

        "seed": 42,
        "train_ratio": 0.8,
        "batch_size": 256,
        "epochs": 800,
        "patience": 160,

        "input_dim": 33 * 2,
        "hidden_dims": [32, 32, 32],
        "output_dim": 33 + 1,
        "activation": "ReLU",
        "dropout": 0.0,
        "batchnorm": False,

        "lr": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "scheduler_patience": 25,
        "min_lr": 2e-6,

        "voltage_loss_weight": 5.0,
        "worst_i_loss_weight": 2.0,
        "voltage_violate_mse_weight": 6.0,
        "worst_i_violate_mse_weight": 5.0,

        "warmup_epochs": 80,
        "penalty_ramp_epochs": 120,
        "worst_i_false_safe_lambda": 0.0,
        "worst_i_sign_loss_weight": 8.0,
        "worst_i_sign_margin": 0.003,
        "worst_i_false_safe_margin": 0.006,

        "v_false_safe_lambda": 0.0,
        "v_false_safe_guard": 0.003,
        "v_safety_loss_weight": 1.0,
        "v_safety_margin": 0.002,
        "v_safety_scale": 0.005,
        "v_safety_fs_weight": 1.0,
        "v_safety_fv_weight": 1.0,

        "v_lower": 0.95,
        "v_upper": 1.05,
        "big_m_beta": 1.10,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def build_base_model(cfg, edge_list=None):
    return MLPExp17Outputs(
        input_dim=cfg["input_dim"],
        hidden_dims=cfg["hidden_dims"],
        output_dim=cfg["output_dim"],
        dropout=cfg["dropout"],
        batchnorm=cfg["batchnorm"],
    )


def unpack_forward(model, X):
    out = model(X)
    if isinstance(out, tuple) and len(out) >= 3:
        return out
    raise RuntimeError("model.forward must return Vn, YI_worst_n, hidden_Z_list.")


def denorm_outputs(Vn, YI_worst_n, norm):
    device = Vn.device
    batch_size = Vn.shape[0]

    YV_mean = norm["YV_mean_wo_slack"].to(device).view(1, -1)
    YV_std = norm["YV_std_wo_slack"].to(device).view(1, -1)
    YI_mean = norm["YI_worst_mean"].to(device).view(1, -1)
    YI_std = norm["YI_worst_std"].to(device).view(1, -1)

    V = torch.ones((batch_size, 33), dtype=Vn.dtype, device=device)
    if Vn.shape[1] == 33:
        V[:, 1:] = Vn[:, 1:] * YV_std + YV_mean
    else:
        V[:, 1:] = Vn * YV_std + YV_mean

    YI_worst = YI_worst_n * YI_std + YI_mean
    return V, YI_worst


def masked_mean(x, mask):
    mask = mask.float()
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def voltage_safety_classification_loss(v_true, v_pred, cfg):
    v_lower = cfg["v_lower"]
    v_upper = cfg["v_upper"]
    margin = cfg["v_safety_margin"]
    scale = max(float(cfg["v_safety_scale"]), 1e-8)

    low_unsafe = v_true < v_lower
    high_unsafe = v_true > v_upper
    safe = (v_true >= v_lower) & (v_true <= v_upper)

    low_fs_loss = masked_mean((F.relu(v_pred - (v_lower - margin)) / scale).square(), low_unsafe)
    high_fs_loss = masked_mean((F.relu((v_upper + margin) - v_pred) / scale).square(), high_unsafe)
    fv_low_loss = masked_mean((F.relu(v_lower - v_pred) / scale).square(), safe)
    fv_high_loss = masked_mean((F.relu(v_pred - v_upper) / scale).square(), safe)

    v_fs_loss = low_fs_loss + high_fs_loss
    v_fv_loss = fv_low_loss + fv_high_loss
    v_safety_loss = cfg["v_safety_fs_weight"] * v_fs_loss + cfg["v_safety_fv_weight"] * v_fv_loss
    return v_safety_loss, v_fs_loss, v_fv_loss, low_fs_loss, high_fs_loss


def compute_loss(model, batch, norm, cfg, ramp=1.0):
    X, YVn, YIn, YV, YI_worst = batch
    Vn, YI_worst_n, *_ = unpack_forward(model, X)
    V, YI_worst_pred = denorm_outputs(Vn, YI_worst_n, norm)

    Vn_use = Vn[:, 1:] if Vn.shape[1] == 33 else Vn
    YVn_use = YVn[:, 1:]
    v_true = YV[:, 1:]
    v_pred = V[:, 1:]
    i_true = YI_worst
    i_pred = YI_worst_pred

    v_unsafe = (v_true < cfg["v_lower"]) | (v_true > cfg["v_upper"])
    i_unsafe = i_true > 0.0
    v_weight = 1.0 + cfg["voltage_violate_mse_weight"] * v_unsafe.float()
    i_weight = 1.0 + cfg["worst_i_violate_mse_weight"] * i_unsafe.float()

    node_mse = torch.mean(v_weight * (Vn_use - YVn_use).square())
    worst_i_mse = torch.mean(i_weight * (YI_worst_n - YIn).square())
    base = cfg["voltage_loss_weight"] * node_mse + cfg["worst_i_loss_weight"] * worst_i_mse

    i_pen = masked_mean(F.relu(cfg["worst_i_false_safe_margin"] - i_pred).square(), i_unsafe)
    i_safe = ~i_unsafe
    i_margin = cfg["worst_i_sign_margin"]
    i_sign_fs_loss = masked_mean(F.relu(i_margin - i_pred).square(), i_unsafe)
    i_sign_fv_loss = masked_mean(F.relu(i_pred + i_margin).square(), i_safe)
    i_sign_loss = i_sign_fs_loss + i_sign_fv_loss

    low_mask = v_true < cfg["v_lower"]
    high_mask = v_true > cfg["v_upper"]
    low_pen = masked_mean(F.relu(v_pred - (cfg["v_lower"] - cfg["v_false_safe_guard"])).square(), low_mask)
    high_pen = masked_mean(F.relu((cfg["v_upper"] + cfg["v_false_safe_guard"]) - v_pred).square(), high_mask)
    v_pen = low_pen + high_pen

    v_safety_loss, v_safety_fs_loss, v_safety_fv_loss, v_low_fs_loss, v_high_fs_loss = (
        voltage_safety_classification_loss(v_true, v_pred, cfg)
    )
    total = base + ramp * (
        cfg["worst_i_false_safe_lambda"] * i_pen
        + cfg["worst_i_sign_loss_weight"] * i_sign_loss
        + cfg["v_false_safe_lambda"] * v_pen
        + cfg["v_safety_loss_weight"] * v_safety_loss
    )

    return total, {
        "base": base.detach(),
        "node_mse": node_mse.detach(),
        "worst_i_mse": worst_i_mse.detach(),
        "i_pen": i_pen.detach(),
        "i_sign_loss": i_sign_loss.detach(),
        "i_sign_fs_loss": i_sign_fs_loss.detach(),
        "i_sign_fv_loss": i_sign_fv_loss.detach(),
        "v_pen": v_pen.detach(),
        "v_safety_loss": v_safety_loss.detach(),
        "v_safety_fs_loss": v_safety_fs_loss.detach(),
        "v_safety_fv_loss": v_safety_fv_loss.detach(),
        "v_low_fs_loss": v_low_fs_loss.detach(),
        "v_high_fs_loss": v_high_fs_loss.detach(),
    }


@torch.no_grad()
def evaluate(model, loader, norm, cfg, device):
    model.eval()
    V_pred_all, I_pred_all, V_true_all, I_true_all = [], [], [], []
    sums = {
        "base": 0.0,
        "total": 0.0,
        "node_mse": 0.0,
        "worst_i_mse": 0.0,
        "i_sign_loss": 0.0,
        "v_safety_loss": 0.0,
    }
    n_batch = 0

    for batch in loader:
        batch = [x.to(device) for x in batch]
        loss, info = compute_loss(model, batch, norm, cfg, ramp=1.0)
        X, _, _, YV, YI_worst = batch
        Vn, YI_worst_n, *_ = unpack_forward(model, X)
        V, YI_pred = denorm_outputs(Vn, YI_worst_n, norm)

        V_pred_all.append(V.cpu())
        I_pred_all.append(YI_pred.cpu())
        V_true_all.append(YV.cpu())
        I_true_all.append(YI_worst.cpu())

        sums["total"] += float(loss.item())
        for key in ["base", "node_mse", "worst_i_mse", "i_sign_loss", "v_safety_loss"]:
            sums[key] += float(info[key].item())
        n_batch += 1

    Vp = torch.cat(V_pred_all)
    Ip = torch.cat(I_pred_all)
    Vt = torch.cat(V_true_all)
    It = torch.cat(I_true_all)

    v_err = torch.abs(Vp[:, 1:] - Vt[:, 1:])
    i_err = torch.abs(Ip - It)
    v_true_unsafe = (Vt[:, 1:] < cfg["v_lower"]) | (Vt[:, 1:] > cfg["v_upper"])
    v_pred_unsafe = (Vp[:, 1:] < cfg["v_lower"]) | (Vp[:, 1:] > cfg["v_upper"])
    i_true_unsafe = It > 0.0
    i_pred_unsafe = Ip > 0.0

    def rates(true_unsafe, pred_unsafe):
        tp = (true_unsafe & pred_unsafe).sum().item()
        fn = (true_unsafe & (~pred_unsafe)).sum().item()
        fp = ((~true_unsafe) & pred_unsafe).sum().item()
        tn = ((~true_unsafe) & (~pred_unsafe)).sum().item()
        total = max(tp + tn + fp + fn, 1)
        unsafe_total = max(tp + fn, 1)
        safe_total = max(tn + fp, 1)
        unsafe_recall = tp / unsafe_total * 100.0
        safe_recall = tn / safe_total * 100.0
        return {
            "FalseSafe": fn / unsafe_total * 100.0,
            "FalseViolate": fp / safe_total * 100.0,
            "SafetyAcc": (tp + tn) / total * 100.0,
            "BalancedAcc": 0.5 * (unsafe_recall + safe_recall),
            "UnsafeRecall": unsafe_recall,
            "SafeRecall": safe_recall,
            "TrueUnsafeCount": int(true_unsafe.sum().item()),
            "PredUnsafeCount": int(pred_unsafe.sum().item()),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
        }

    metrics = {
        "ValBaseLoss": sums["base"] / n_batch,
        "ValTotalLoss": sums["total"] / n_batch,
        "ValNodeMSE": sums["node_mse"] / n_batch,
        "ValWorstIMSE": sums["worst_i_mse"] / n_batch,
        "ValISignLoss": sums["i_sign_loss"] / n_batch,
        "ValVSafetyLoss": sums["v_safety_loss"] / n_batch,
        "V_MAE": float(v_err.mean()),
        "V_RMSE": float(torch.sqrt(torch.mean((Vp[:, 1:] - Vt[:, 1:]).square()))),
        "V_MaxErr": float(v_err.max()),
        "WorstI_MAE": float(i_err.mean()),
        "WorstI_RMSE": float(torch.sqrt(torch.mean((Ip - It).square()))),
        "WorstI_MaxErr": float(i_err.max()),
    }
    for prefix, values in [("V", rates(v_true_unsafe, v_pred_unsafe)), ("WorstI", rates(i_true_unsafe, i_pred_unsafe))]:
        for key, value in values.items():
            metrics[f"{prefix}_{key}"] = value
    metrics["WorstI_SignAcc"] = metrics["WorstI_SafetyAcc"]
    return metrics


def ramp_lambda(epoch, cfg):
    if epoch <= cfg["warmup_epochs"]:
        return 0.0
    x = (epoch - cfg["warmup_epochs"]) / max(cfg["penalty_ramp_epochs"], 1)
    return float(min(max(x, 0.0), 1.0))


def selection_metric(m):
    return (
        m["V_MAE"]
        + 0.2 * m["WorstI_MAE"]
        + 0.0030 * m["V_FalseSafe"]
        + 0.0015 * m["V_FalseViolate"]
        + 0.0020 * m["WorstI_FalseSafe"]
        + 0.0015 * m["WorstI_FalseViolate"]
        + 0.0010 * (100.0 - m["V_BalancedAcc"])
        + 0.0008 * (100.0 - m["WorstI_BalancedAcc"])
    )


@torch.no_grad()
def extract_big_m(model, Xn, cfg, device):
    model.eval()
    hidden_z_all = [[] for _ in cfg["hidden_dims"]]
    for i in range(0, len(Xn), cfg["batch_size"]):
        xb = Xn[i:i + cfg["batch_size"]].to(device)
        *_, hidden_Z_list = unpack_forward(model, xb)
        if len(hidden_Z_list) != len(cfg["hidden_dims"]):
            raise RuntimeError("hidden_Z_list length must equal config.hidden_dims.")
        for layer_idx, Z in enumerate(hidden_Z_list):
            hidden_z_all[layer_idx].append(Z.detach().cpu())

    beta = cfg["big_m_beta"]
    M_plus_mlp, M_minus_mlp = [], []
    for layer_values in hidden_z_all:
        Z = torch.cat(layer_values, dim=0)
        M_plus_mlp.append(torch.clamp(Z.max(dim=0).values, min=0.0) * beta)
        M_minus_mlp.append(torch.clamp((-Z).max(dim=0).values, min=0.0) * beta)
    return {"M_plus_mlp_layers": M_plus_mlp, "M_minus_mlp_layers": M_minus_mlp}


def move_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_cpu(v) for v in obj)
    return obj


def main():
    cfg = get_config()
    set_seed(cfg["seed"])
    os.makedirs(cfg["save_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    data = load_pt(cfg["data_path"])
    edge_list = get_edge_list(data)
    X_raw = to_tensor(data["X"]).float()[:, :, :2].contiguous()
    YV, YI_branch = get_labels(data)
    YI_worst = YI_branch.max(dim=1, keepdim=True).values

    unsafe_count = int((YI_worst > 0.0).sum().item())
    total_count = int(YI_worst.numel())

    n = len(X_raw)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"]))
    n_train = int(n * cfg["train_ratio"])
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    Xn, YVn, YIn, norm = normalize_data(X_raw, YV, YI_worst, train_idx)

    train_ds = TensorDataset(Xn[train_idx], YVn[train_idx], YIn[train_idx], YV[train_idx], YI_worst[train_idx])
    val_ds = TensorDataset(Xn[val_idx], YVn[val_idx], YIn[val_idx], YV[val_idx], YI_worst[val_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, drop_last=False)

    model = build_base_model(cfg, edge_list).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=cfg["scheduler_patience"],
        min_lr=cfg["min_lr"],
    )

    best_metric = float("inf")
    best_state = None
    best_metrics = None
    wait = 0
    t0 = time.time()
    best_model_path = os.path.join(cfg["save_dir"], cfg["best_model_name"])

    print("\n================ MLP Exp1 Exp17-Output Config ================")
    print(f"Experiment: {cfg['exp_name']}")
    print("Input features: [P_net,Q_net], flattened 33*2=66")
    print("No aggregated power features; no voltage linear prior")
    print("Outputs: V_nodes(33) and YI_worst=max(Y_I_branch)")
    print(f"input_dim={cfg['input_dim']}, hidden_dims={cfg['hidden_dims']}, output_dim={cfg['output_dim']}")
    print(f"Binary count estimate: {model.get_binary_count()}")
    print(f"Samples: {n}, train={len(train_idx)}, val={len(val_idx)}")
    print(f"YI_worst unsafe samples: {unsafe_count}, ratio={unsafe_count / max(total_count, 1) * 100.0:.2f}%")
    print("==============================================================\n")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_train = 0.0
        lam = ramp_lambda(epoch, cfg)
        for batch in train_loader:
            batch = [x.to(device) for x in batch]
            optimizer.zero_grad()
            loss, _ = compute_loss(model, batch, norm, cfg, ramp=lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            total_train += float(loss.item())

        metrics = evaluate(model, val_loader, norm, cfg, device)
        sel = selection_metric(metrics)
        scheduler.step(sel)
        if sel < best_metric:
            best_metric = sel
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics.copy()
            torch.save(
                {
                    "model_state_dict": best_state,
                    "config": cfg,
                    "norm_stats": move_to_cpu(norm),
                    "edge_list": edge_list,
                    "best_metrics": best_metrics,
                    "binary_count": model.get_binary_count(),
                },
                best_model_path,
            )
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch:04d}/{cfg['epochs']}] | LR={lr:.2e} | "
                f"Train={total_train / len(train_loader):.4f} | "
                f"NodeMSE={metrics['ValNodeMSE']:.4f} | WorstIMSE={metrics['ValWorstIMSE']:.4f} | "
                f"V_MAE={metrics['V_MAE']:.6f} | WorstI_MAE={metrics['WorstI_MAE']:.6f} | "
                f"V-FS/FV={metrics['V_FalseSafe']:.2f}/{metrics['V_FalseViolate']:.2f}% | "
                f"WorstI-FS/FV={metrics['WorstI_FalseSafe']:.2f}/{metrics['WorstI_FalseViolate']:.2f}% | "
                f"V-BAcc={metrics['V_BalancedAcc']:.2f}% | "
                f"WorstI-BAcc={metrics['WorstI_BalancedAcc']:.2f}% | lambda={lam:.2f}"
            )

        if wait >= cfg["patience"]:
            print(f"\nEarly stopping: epoch={epoch}, best_metric={best_metric:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate(model, val_loader, norm, cfg, device)
    M = extract_big_m(model, Xn, cfg, device)
    engine_path = os.path.join(cfg["save_dir"], cfg["engine_name"])
    engine = {
        "model_type": cfg["exp_name"],
        "base_model_class": "MLP-Exp17-Outputs-RawPQ",
        "state_dict": move_to_cpu(model.state_dict()),
        "output_mode": "node_voltage_and_worst_current",
        "output_names": ["V_nodes", "YI_worst"],
        "predict_voltage_target": "Y_V node voltages including slack node",
        "predict_current_target": "YI_worst=max(Y_I_branch)",
        "current_constraint_meaning": "YI_worst_pred <= 0 implies predicted system-level current safety",
        "predicts_node_voltage": True,
        "predicts_branch_current": False,
        "uses_aggregated_power_features": False,
        "uses_voltage_linear_prior": False,
        "uses_current_linear_prior": False,
        "config": cfg,
        "input_shape": [33, 2],
        "input_dim": cfg["input_dim"],
        "feature_names": ["P_net", "Q_net"],
        "hidden_dims": list(cfg["hidden_dims"]),
        "output_dim": cfg["output_dim"],
        "binary_count": model.get_binary_count(),
        "norm_stats": move_to_cpu(norm),
        "edge_list": edge_list,
        "safety_settings": {
            "v_lower": cfg["v_lower"],
            "v_upper": cfg["v_upper"],
            "current_target_limit": 0.0,
            "worst_i_false_safe_margin": cfg["worst_i_false_safe_margin"],
            "worst_i_sign_margin": cfg["worst_i_sign_margin"],
            "worst_i_sign_loss_weight": cfg["worst_i_sign_loss_weight"],
            "v_false_safe_guard": cfg["v_false_safe_guard"],
            "v_safety_margin": cfg["v_safety_margin"],
            "v_safety_scale": cfg["v_safety_scale"],
            "v_safety_loss_weight": cfg["v_safety_loss_weight"],
        },
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "elapsed_sec": float(time.time() - t0),
        **M,
    }
    torch.save(engine, engine_path)

    print("\n================ Final Validation ================")
    print(f"SelectionMetric: {best_metric:.6f}")
    for key, value in final_metrics.items():
        if "False" in key or "Acc" in key or "Recall" in key:
            print(f"{key}: {value:.2f}%")
        elif "Count" in key or key.endswith(("_TP", "_TN", "_FP", "_FN")):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")
    print(f"\nBinary count estimate: {model.get_binary_count()}")
    print(f"best model: {best_model_path}")
    print(f"MILP engine: {engine_path}")
    print("==================================================\n")


if __name__ == "__main__":
    main()
