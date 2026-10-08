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


class MLPFourScalars(nn.Module):
    """Raw-PQ MLP that predicts Vdev_total, Vworst, WorstI, and Ploss_total."""

    def __init__(
        self,
        input_dim=66,
        hidden_dims=(32, 32, 32),
        output_dim=4,
        dropout=0.0,
        batchnorm=False,
    ):
        super().__init__()
        if output_dim != 4:
            raise ValueError(f"output_dim must be 4, got {output_dim}")
        if sum(hidden_dims) != 96:
            raise ValueError(f"sum(hidden_dims) must be 96, got {sum(hidden_dims)}")

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
        return out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4], hidden_Z_list

    def get_binary_count(self):
        return {
            "mlp_hidden_binary_per_layer": list(self.hidden_dims),
            "mlp_binary": sum(self.hidden_dims),
            "total_binary": sum(self.hidden_dims),
        }

def get_config():
    return {
        "exp_name": "MLP_Exp20_FourScalars_96Binary_RawPQ",
        "data_path": r"data/ieee33_nodal_pq_correlated_raw_pool_50k.pt",
        "save_dir": r"checkpoints",
        "best_model_name": "mlp_exp20_four_scalars_96bin_best.pt",
        "engine_name": "mlp_exp20_four_scalars_96bin_milp_engine.pt",

        "seed": 42,
        "train_ratio": 0.8,
        "batch_size": 256,
        "epochs": 800,
        "patience": 160,

        "input_dim": 33 * 2,
        "hidden_dims": [32, 32, 32],
        "output_dim": 4,
        "activation": "ReLU",
        "dropout": 0.0,
        "batchnorm": False,

        "lr": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "scheduler_patience": 25,
        "min_lr": 2e-6,

        "vdev_loss_weight": 1.0,
        "vworst_loss_weight": 1.0,
        "iworst_loss_weight": 1.0,
        "ploss_loss_weight": 1.0,
        "huber_beta": 0.35,
        "mse_mix": 0.10,

        "vworst_unsafe_extra": 1.0,
        "iworst_unsafe_extra": 1.0,
        "boundary_extra": 1.0,
        "boundary_tau": 0.010,

        "warmup_epochs": 80,
        "penalty_ramp_epochs": 120,
        "vworst_sign_loss_weight": 0.25,
        "iworst_sign_loss_weight": 0.15,
        "vworst_sign_margin": 0.0015,
        "iworst_sign_margin": 0.0020,
        "sign_scale": 0.005,

        "v_lower": 0.95,
        "v_upper": 1.05,
        "default_line_max_i_ka": 0.20,
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


def compute_cumulative_voltage_deviation(YV):
    return torch.sum(torch.abs(YV[:, 1:] - 1.0), dim=1, keepdim=True)


def compute_worst_voltage_margin(YV, v_lower=0.95, v_upper=1.05):
    V = YV[:, 1:]
    upper_worst = torch.max(V - float(v_upper), dim=1, keepdim=True).values
    lower_worst = torch.max(float(v_lower) - V, dim=1, keepdim=True).values
    return torch.maximum(upper_worst, lower_worst)


def get_line_max_i_ka(data, cfg):
    return float(data.get("base_config", {}).get("line_max_i_ka", cfg["default_line_max_i_ka"]))


def get_branch_resistance(data, edge_list):
    if "branch_full" in data:
        return torch.tensor([float(row[2]) for row in data["branch_full"]], dtype=torch.float32)
    if len(edge_list) == len(RADIAL_BRANCH_R_OHM):
        return torch.tensor(RADIAL_BRANCH_R_OHM, dtype=torch.float32)
    raise KeyError("Dataset must contain branch_full to compute total network loss.")


def compute_total_network_loss(YI_margin, branch_r_ohm, line_max_i_ka):
    current_ka = (YI_margin + 1.0).clamp_min(0.0) * float(line_max_i_ka)
    branch_r_ohm = branch_r_ohm.to(YI_margin.device).view(1, -1)
    return torch.sum(3.0 * current_ka.square() * branch_r_ohm, dim=1, keepdim=True)


def normalize_data(X, YV_dev, YV_worst, YI_worst, YP_loss, train_idx):
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


def build_base_model(cfg, edge_list=None):
    return MLPFourScalars(
        input_dim=cfg["input_dim"],
        hidden_dims=cfg["hidden_dims"],
        output_dim=cfg["output_dim"],
        dropout=cfg["dropout"],
        batchnorm=cfg["batchnorm"],
    )


def unpack_forward(model, X):
    out = model(X)
    if isinstance(out, tuple) and len(out) >= 5:
        return out
    raise RuntimeError("model.forward must return Vdev, Vworst, WorstI, Ploss, hidden_Z_list.")


def denorm_outputs(Vdev_n, Vworst_n, Iworst_n, Ploss_n, norm):
    device = Vdev_n.device

    def denorm(y, mean_key, std_key):
        mean = norm[mean_key].to(device).view(1, -1)
        std = norm[std_key].to(device).view(1, -1)
        return y * std + mean

    return (
        denorm(Vdev_n, "YV_dev_mean", "YV_dev_std"),
        denorm(Vworst_n, "YV_worst_mean", "YV_worst_std"),
        denorm(Iworst_n, "YI_worst_mean", "YI_worst_std"),
        denorm(Ploss_n, "YP_loss_mean", "YP_loss_std"),
    )


def masked_mean(x, mask):
    mask = mask.float()
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def weighted_regression_loss(pred, true, weight, cfg):
    huber = F.smooth_l1_loss(pred, true, beta=cfg["huber_beta"], reduction="none")
    mse = (pred - true).square()
    loss = huber + cfg["mse_mix"] * mse
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def sign_margin_loss(true, pred, margin, scale):
    unsafe = true > 0.0
    safe = ~unsafe
    fs = masked_mean((F.relu(float(margin) - pred) / float(scale)).square(), unsafe)
    fv = masked_mean((F.relu(pred + float(margin)) / float(scale)).square(), safe)
    return fs + fv, fs, fv


def compute_loss(model, batch, norm, cfg, ramp=1.0):
    X, YVDevn, YVWorstn, YIn, YPLossn, YV_dev, YV_worst, YI_worst, YP_loss = batch
    Vdev_n, Vworst_n, Iworst_n, Ploss_n, *_ = unpack_forward(model, X)
    _, Vworst_pred, Iworst_pred, _ = denorm_outputs(
        Vdev_n,
        Vworst_n,
        Iworst_n,
        Ploss_n,
        norm,
    )

    v_unsafe = YV_worst > 0.0
    i_unsafe = YI_worst > 0.0
    boundary = torch.exp(-torch.abs(YV_worst) / cfg["boundary_tau"])
    i_boundary = torch.exp(-torch.abs(YI_worst) / cfg["boundary_tau"])

    vdev_reg = weighted_regression_loss(Vdev_n, YVDevn, torch.ones_like(YV_dev), cfg)
    vworst_reg = weighted_regression_loss(
        Vworst_n,
        YVWorstn,
        1.0 + cfg["vworst_unsafe_extra"] * v_unsafe.float() + cfg["boundary_extra"] * boundary,
        cfg,
    )
    iworst_reg = weighted_regression_loss(
        Iworst_n,
        YIn,
        1.0 + cfg["iworst_unsafe_extra"] * i_unsafe.float() + cfg["boundary_extra"] * i_boundary,
        cfg,
    )
    ploss_reg = weighted_regression_loss(Ploss_n, YPLossn, torch.ones_like(YP_loss), cfg)

    base = (
        cfg["vdev_loss_weight"] * vdev_reg
        + cfg["vworst_loss_weight"] * vworst_reg
        + cfg["iworst_loss_weight"] * iworst_reg
        + cfg["ploss_loss_weight"] * ploss_reg
    )
    v_sign, v_sign_fs, v_sign_fv = sign_margin_loss(
        YV_worst,
        Vworst_pred,
        cfg["vworst_sign_margin"],
        cfg["sign_scale"],
    )
    i_sign, i_sign_fs, i_sign_fv = sign_margin_loss(
        YI_worst,
        Iworst_pred,
        cfg["iworst_sign_margin"],
        cfg["sign_scale"],
    )
    total = base + ramp * (
        cfg["vworst_sign_loss_weight"] * v_sign
        + cfg["iworst_sign_loss_weight"] * i_sign
    )
    return total, {
        "base": base.detach(),
        "vdev_reg": vdev_reg.detach(),
        "vworst_reg": vworst_reg.detach(),
        "iworst_reg": iworst_reg.detach(),
        "ploss_reg": ploss_reg.detach(),
        "v_sign": v_sign.detach(),
        "v_sign_fs": v_sign_fs.detach(),
        "v_sign_fv": v_sign_fv.detach(),
        "i_sign": i_sign.detach(),
        "i_sign_fs": i_sign_fs.detach(),
        "i_sign_fv": i_sign_fv.detach(),
    }


def binary_classification_metrics(true, pred, prefix):
    true_unsafe = true > 0.0
    pred_unsafe = pred > 0.0
    tp = int((true_unsafe & pred_unsafe).sum().item())
    fn = int((true_unsafe & (~pred_unsafe)).sum().item())
    fp = int(((~true_unsafe) & pred_unsafe).sum().item())
    tn = int(((~true_unsafe) & (~pred_unsafe)).sum().item())
    unsafe_total = max(tp + fn, 1)
    safe_total = max(tn + fp, 1)
    total = max(tp + tn + fp + fn, 1)
    unsafe_recall = tp / unsafe_total * 100.0
    safe_recall = tn / safe_total * 100.0
    return {
        f"{prefix}_FalseSafe": fn / unsafe_total * 100.0,
        f"{prefix}_FalseViolate": fp / safe_total * 100.0,
        f"{prefix}_SignAcc": (tp + tn) / total * 100.0,
        f"{prefix}_BalancedAcc": 0.5 * (unsafe_recall + safe_recall),
        f"{prefix}_UnsafeRecall": unsafe_recall,
        f"{prefix}_SafeRecall": safe_recall,
        f"{prefix}_TrueUnsafeCount": int(true_unsafe.sum().item()),
        f"{prefix}_PredUnsafeCount": int(pred_unsafe.sum().item()),
        f"{prefix}_TP": tp,
        f"{prefix}_TN": tn,
        f"{prefix}_FP": fp,
        f"{prefix}_FN": fn,
    }


@torch.no_grad()
def evaluate(model, loader, norm, cfg, device):
    model.eval()
    pred_store = [[], [], [], []]
    true_store = [[], [], [], []]
    sums = {
        k: 0.0
        for k in [
            "base",
            "total",
            "vdev_reg",
            "vworst_reg",
            "iworst_reg",
            "ploss_reg",
            "v_sign",
            "i_sign",
        ]
    }
    n_batch = 0
    for batch in loader:
        batch = [x.to(device) for x in batch]
        loss, info = compute_loss(model, batch, norm, cfg, ramp=1.0)
        X, _, _, _, _, YV_dev, YV_worst, YI_worst, YP_loss = batch
        Vdev_n, Vworst_n, Iworst_n, Ploss_n, *_ = unpack_forward(model, X)
        preds = denorm_outputs(Vdev_n, Vworst_n, Iworst_n, Ploss_n, norm)
        for k, value in enumerate(preds):
            pred_store[k].append(value.cpu())
        for k, value in enumerate([YV_dev, YV_worst, YI_worst, YP_loss]):
            true_store[k].append(value.cpu())
        sums["total"] += float(loss.item())
        for key in ["base", "vdev_reg", "vworst_reg", "iworst_reg", "ploss_reg", "v_sign", "i_sign"]:
            sums[key] += float(info[key].item())
        n_batch += 1

    Vp, Vwp, Ip, Pp = [torch.cat(x) for x in pred_store]
    Vt, Vwt, It, Pt = [torch.cat(x) for x in true_store]
    metrics = {
        "ValBaseLoss": sums["base"] / n_batch,
        "ValTotalLoss": sums["total"] / n_batch,
        "ValVDevReg": sums["vdev_reg"] / n_batch,
        "ValVWorstReg": sums["vworst_reg"] / n_batch,
        "ValWorstIReg": sums["iworst_reg"] / n_batch,
        "ValPLossReg": sums["ploss_reg"] / n_batch,
        "ValVSignLoss": sums["v_sign"] / n_batch,
        "ValISignLoss": sums["i_sign"] / n_batch,
    }
    for name, pred, true in [
        ("Vdev", Vp, Vt),
        ("Vworst", Vwp, Vwt),
        ("WorstI", Ip, It),
        ("Ploss", Pp, Pt),
    ]:
        err = torch.abs(pred - true)
        metrics[f"{name}_MAE"] = float(err.mean())
        metrics[f"{name}_RMSE"] = float(torch.sqrt(torch.mean((pred - true).square())))
        metrics[f"{name}_MaxErr"] = float(err.max())
    metrics.update(binary_classification_metrics(Vwt, Vwp, "Vworst"))
    metrics.update(binary_classification_metrics(It, Ip, "WorstI"))
    return metrics


def ramp_lambda(epoch, cfg):
    if epoch <= cfg["warmup_epochs"]:
        return 0.0
    x = (epoch - cfg["warmup_epochs"]) / max(cfg["penalty_ramp_epochs"], 1)
    return float(min(max(x, 0.0), 1.0))


def selection_metric(metrics):
    classification_penalty = 0.01 * (
        2.0 * metrics["Vworst_FalseSafe"]
        + 1.5 * metrics["Vworst_FalseViolate"]
        + 2.0 * metrics["WorstI_FalseSafe"]
        + 1.5 * metrics["WorstI_FalseViolate"]
    )
    return (
        metrics["ValVDevReg"]
        + metrics["ValVWorstReg"]
        + metrics["ValWorstIReg"]
        + metrics["ValPLossReg"]
        + classification_penalty
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
    YV_dev = compute_cumulative_voltage_deviation(YV)
    YV_worst = compute_worst_voltage_margin(YV, cfg["v_lower"], cfg["v_upper"])
    YI_worst = YI_branch.max(dim=1, keepdim=True).values
    branch_r_ohm = get_branch_resistance(data, edge_list)
    line_max_i_ka = get_line_max_i_ka(data, cfg)
    YP_loss = compute_total_network_loss(YI_branch, branch_r_ohm, line_max_i_ka)

    n = len(X_raw)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"]))
    n_train = int(n * cfg["train_ratio"])
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    Xn, YVDevn, YVWorstn, YIn, YPLossn, norm = normalize_data(
        X_raw,
        YV_dev,
        YV_worst,
        YI_worst,
        YP_loss,
        train_idx,
    )

    train_ds = TensorDataset(
        Xn[train_idx],
        YVDevn[train_idx],
        YVWorstn[train_idx],
        YIn[train_idx],
        YPLossn[train_idx],
        YV_dev[train_idx],
        YV_worst[train_idx],
        YI_worst[train_idx],
        YP_loss[train_idx],
    )
    val_ds = TensorDataset(
        Xn[val_idx],
        YVDevn[val_idx],
        YVWorstn[val_idx],
        YIn[val_idx],
        YPLossn[val_idx],
        YV_dev[val_idx],
        YV_worst[val_idx],
        YI_worst[val_idx],
        YP_loss[val_idx],
    )
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

    print("\n================ MLP Four-Scalar Config ================")
    print(f"Experiment: {cfg['exp_name']}")
    print("Input features: [P_net,Q_net], flattened 33*2=66")
    print("Outputs: Vdev_total, Vworst, WorstI, Ploss_total")
    print(f"input_dim={cfg['input_dim']}, hidden_dims={cfg['hidden_dims']}, output_dim={cfg['output_dim']}")
    print(f"Binary count estimate: {model.get_binary_count()}")
    print(f"Samples: {n}, train={len(train_idx)}, val={len(val_idx)}")
    print("========================================================\n")

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
                    "branch_r_ohm": branch_r_ohm,
                    "line_max_i_ka": line_max_i_ka,
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
                f"Vdev={metrics['Vdev_MAE']:.6f} | "
                f"Vworst={metrics['Vworst_MAE']:.6f} | "
                f"Iworst={metrics['WorstI_MAE']:.6f} | "
                f"V-FS/FV={metrics['Vworst_FalseSafe']:.2f}/{metrics['Vworst_FalseViolate']:.2f}% | "
                f"I-FS/FV={metrics['WorstI_FalseSafe']:.2f}/{metrics['WorstI_FalseViolate']:.2f}% | "
                f"Ploss={metrics['Ploss_MAE']:.6f} | lambda={lam:.2f}"
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
        "base_model_class": "MLP-Four-Direct-Scalars",
        "state_dict": move_to_cpu(model.state_dict()),
        "output_mode": "four_direct_system_level_scalars",
        "output_names": ["Vdev_total", "Vworst", "WorstI", "Ploss_total"],
        "predict_voltage_objective_target": "Vdev_total=sum(abs(Y_V_without_slack-1.0))",
        "predict_voltage_safety_target": "Vworst=max(max(Y_V_without_slack-v_upper),max(v_lower-Y_V_without_slack))",
        "predict_current_target": "WorstI=max(Y_I_branch)",
        "predict_network_loss_target": "Ploss_total=sum(3*I_ka^2*R_ohm)",
        "voltage_constraint_meaning": "Vworst_pred <= 0",
        "current_constraint_meaning": "WorstI_pred <= 0",
        "predicts_node_voltage": False,
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
        "branch_r_ohm": branch_r_ohm,
        "line_max_i_ka": line_max_i_ka,
        "safety_settings": {
            "v_lower": cfg["v_lower"],
            "v_upper": cfg["v_upper"],
            "voltage_safety_limit": 0.0,
            "current_safety_limit": 0.0,
            "vworst_sign_margin": cfg["vworst_sign_margin"],
            "iworst_sign_margin": cfg["iworst_sign_margin"],
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
