import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

if __package__ is None or __package__ == "":
    import sys

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir in sys.path:
        sys.path.remove(script_dir)
    sys.path.insert(0, os.path.dirname(os.path.dirname(script_dir)))
    __package__ = "model.exp23"

from .data import (
    load_pt,
    to_tensor,
    sanitize_edge_list,
    get_edge_list,
    build_adj_norm,
    build_adj_powers,
    build_topology_matrices,
    augment_path_power_features,
    get_labels,
    compute_cumulative_voltage_deviation,
    compute_worst_voltage_margin,
    get_line_max_i_ka,
    get_branch_resistance,
    compute_total_network_loss,
    normalize_data,
)
from .loss import compute_loss
from .model import (
    ScalarReadout,
    STSGCNThreeDirectScalars,
    denorm_outputs,
    unpack_forward,
)


def get_config():
    return {
        "exp_name": "ST_SGCN_Exp23_K4_H24_ThreeDirectScalars_Hsafe",
        "data_path": r"data/ieee33_static_vvo_balanced_20k.pt",
        "save_dir": r"checkpoints",
        "best_model_name": "st_sgcn_exp23_three_direct_scalar_heads_best.pt",
        "engine_name": "st_sgcn_exp23_three_direct_scalar_heads_milp_engine.pt",
        "report_name": "exp23_final_validation.txt",

        "seed": 42,
        "train_ratio": 0.8,
        "batch_size": 256,
        "epochs": 1000,
        "patience": 140,

        "K": 4,
        "hidden_dim": 24,
        "vdev_head_dim": 32,
        "hsafe_head_dim": 32,
        "ploss_head_dim": 32,
        "include_order0": True,
        "use_sgc_relu": False,
        "use_linear_skip": True,

        "lr": 8e-4,
        "weight_decay": 5e-5,
        "grad_clip": 2.0,
        "scheduler_patience": 30,
        "min_lr": 2e-6,

        # 三个标签均已标准化，因此基础回归权重保持一致。
        "vdev_loss_weight": 1.0,
        "hsafe_loss_weight": 1.0,
        "ploss_loss_weight": 1.0,

        # Blend normalized absolute regression with a physical-scale relative
        # loss. The floor prevents tiny network-loss targets from dominating.

        # SmoothL1 更贴近 MAE，少量 MSE 抑制大误差。
        "huber_beta": 0.35,
        "mse_mix": 0.10,

        # 对安全边界附近样本和越限样本适度加权。

        "warmup_epochs": 150,
        "penalty_ramp_epochs": 150,

        # 符号损失先按物理量尺度归一化，避免原始平方项过小。
        "hsafe_sign_loss_weight": 0.05,
        "hsafe_sign_scale": 1.0,

        "v_safe_scale": 0.05,
        "i_safe_scale": 0.50,


        # Checkpoint selection is lexicographic: every safety-feasible
        # checkpoint beats an infeasible one; regression decides among
        # feasible checkpoints. Values are percentages for classification.
        "selection_hsafe_false_safe_max": 1.0,
        "selection_hsafe_false_violate_max": 2.0,
        "selection_vdev_mae_scale": 0.008,
        "selection_hsafe_mae_scale": 0.05,
        "selection_ploss_mae_scale": 0.007,
        "selection_infeasible_offset": 100.0,
        "selection_infeasible_reg_weight": 0.10,

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

    pred_store = [[], [], []]
    true_store = [[], [], []]
    sums = {
        "base": 0.0,
        "total": 0.0,
        "vdev_reg": 0.0,
        "hsafe_reg": 0.0,
        "ploss_reg": 0.0,
        "h_sign": 0.0,
    }
    n_batch = 0

    for batch in loader:
        batch = [x.to(device) for x in batch]
        loss, info = compute_loss(
            model,
            batch,
            norm,
            cfg,
            ramp=1.0,
        )

        X, _, _, _, YV_dev, YH_safe, YP_loss = batch
        Vdev_n, Hsafe_n, Ploss_n, *_ = unpack_forward(model, X)

        preds = denorm_outputs(
            Vdev_n,
            Hsafe_n,
            Ploss_n,
            norm,
        )

        for k, value in enumerate(preds):
            pred_store[k].append(value.cpu())

        for k, value in enumerate(
            [YV_dev, YH_safe, YP_loss]
        ):
            true_store[k].append(value.cpu())

        sums["total"] += float(loss.item())
        for key in [
            "base",
            "vdev_reg",
            "hsafe_reg",
            "ploss_reg",
            "h_sign",
        ]:
            sums[key] += float(info[key].item())

        n_batch += 1

    Vp, Hp, Pp = [torch.cat(x) for x in pred_store]
    Vt, Ht, Pt = [torch.cat(x) for x in true_store]

    metrics = {
        "ValBaseLoss": sums["base"] / n_batch,
        "ValTotalLoss": sums["total"] / n_batch,
        "ValVDevReg": sums["vdev_reg"] / n_batch,
        "ValHsafeReg": sums["hsafe_reg"] / n_batch,
        "ValPLossReg": sums["ploss_reg"] / n_batch,
        "ValHSignLoss": sums["h_sign"] / n_batch,
    }

    for name, pred, true in [
        ("Vdev", Vp, Vt),
        ("Hsafe", Hp, Ht),
        ("Ploss", Pp, Pt),
    ]:
        err = torch.abs(pred - true)
        metrics[f"{name}_MAE"] = float(err.mean())
        metrics[f"{name}_RMSE"] = float(
            torch.sqrt(torch.mean((pred - true).square()))
        )
        metrics[f"{name}_MaxErr"] = float(err.max())

    metrics.update(
        binary_classification_metrics(Ht, Hp, "Hsafe")
    )

    return metrics


def ramp_lambda(epoch, cfg):
    if epoch <= cfg["warmup_epochs"]:
        return 0.0

    x = (
        epoch - cfg["warmup_epochs"]
    ) / max(cfg["penalty_ramp_epochs"], 1)

    return float(min(max(x, 0.0), 1.0))


def regression_selection_score(metrics, cfg):
    """Dimensionless mean regression score used for model selection."""
    return (
        metrics["Vdev_MAE"] / cfg["selection_vdev_mae_scale"]
        + metrics["Hsafe_MAE"] / cfg["selection_hsafe_mae_scale"]
        + metrics["Ploss_MAE"] / cfg["selection_ploss_mae_scale"]
    ) / 3.0


def safety_selection_violation(metrics, cfg):
    """Normalized amount by which the Hsafe safety-rate goals are missed."""
    specs = [
        ("Hsafe_FalseSafe", "selection_hsafe_false_safe_max"),
        ("Hsafe_FalseViolate", "selection_hsafe_false_violate_max"),
    ]
    return sum(
        max(0.0, metrics[metric_key] - cfg[limit_key]) / cfg[limit_key]
        for metric_key, limit_key in specs
    )


def selection_metric(metrics, cfg):
    """Prefer safety-feasible checkpoints, then minimize regression error."""
    regression_score = regression_selection_score(metrics, cfg)
    safety_violation = safety_selection_violation(metrics, cfg)
    metrics["SelectionRegressionScore"] = float(regression_score)
    metrics["SelectionSafetyViolation"] = float(safety_violation)
    metrics["SelectionSafetyFeasible"] = bool(safety_violation <= 1e-12)
    if safety_violation <= 1e-12:
        return regression_score
    return (
        cfg["selection_infeasible_offset"]
        + safety_violation
        + cfg["selection_infeasible_reg_weight"] * regression_score
    )

def extract_big_m(model, Xn, cfg, device):
    model.eval()

    gcn_z_all = None
    global_z_all = []

    for i in range(0, len(Xn), cfg["batch_size"]):
        xb = Xn[i:i + cfg["batch_size"]].to(device)
        out = unpack_forward(model, xb)

        gcn_Z_list = out[3]
        Z_global = out[5]

        if gcn_z_all is None:
            gcn_z_all = [[] for _ in range(len(gcn_Z_list))]

        for k, z in enumerate(gcn_Z_list):
            gcn_z_all[k].append(z.detach().cpu())

        global_z_all.append(Z_global.detach().cpu())

    beta = cfg["big_m_beta"]
    M_plus_gcn = []
    M_minus_gcn = []

    if gcn_z_all:
        for z_list in gcn_z_all:
            Z = torch.cat(z_list, dim=0)
            M_plus_gcn.append(
                torch.clamp(Z.max(dim=0).values, min=0.0) * beta
            )
            M_minus_gcn.append(
                torch.clamp((-Z).max(dim=0).values, min=0.0) * beta
            )

    Z_global = torch.cat(global_z_all, dim=0)

    return {
        "M_plus_gcn_layers": M_plus_gcn,
        "M_minus_gcn_layers": M_minus_gcn,
        "M_plus_node": None,
        "M_minus_node": None,
        "M_plus_global": (
            torch.clamp(Z_global.max(dim=0).values, min=0.0) * beta
        ),
        "M_minus_global": (
            torch.clamp((-Z_global).max(dim=0).values, min=0.0) * beta
        ),
    }


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


def format_metric(metrics, key, fmt=".6f", suffix=""):
    value = metrics.get(key)

    if value is None:
        return f"{key}: <missing>"

    if torch.is_tensor(value):
        value = value.detach().cpu().item()

    if fmt == "int":
        return f"{key}: {int(round(float(value)))}"

    return f"{key}: {float(value):{fmt}}{suffix}"


def classification_lines(metrics, prefix):
    specs = [
        (f"{prefix}_FalseSafe", ".2f", "%"),
        (f"{prefix}_FalseViolate", ".2f", "%"),
        (f"{prefix}_SignAcc", ".6f", ""),
        (f"{prefix}_BalancedAcc", ".6f", ""),
        (f"{prefix}_UnsafeRecall", ".6f", ""),
        (f"{prefix}_SafeRecall", ".6f", ""),
        (f"{prefix}_TrueUnsafeCount", "int", ""),
        (f"{prefix}_PredUnsafeCount", "int", ""),
        (f"{prefix}_TP", "int", ""),
        (f"{prefix}_TN", "int", ""),
        (f"{prefix}_FP", "int", ""),
        (f"{prefix}_FN", "int", ""),
    ]

    return [
        format_metric(metrics, key, fmt, suffix)
        for key, fmt, suffix in specs
    ]


def build_final_report(
    best_metric,
    final_metrics,
    binary_count,
    best_model_path,
    engine_path,
):
    lines = [
        "================ ST-SGCN Exp23 Final Validation ================",
        f"SelectionMetric: {best_metric:.6f}",
        "Output mode: three directly predicted system-level scalars",
        format_metric(final_metrics, "SelectionRegressionScore"),
        format_metric(final_metrics, "SelectionSafetyViolation"),
        format_metric(final_metrics, "SelectionSafetyFeasible", fmt="int"),
        "",
        "[Cumulative voltage deviation regression]",
        format_metric(final_metrics, "Vdev_MAE"),
        format_metric(final_metrics, "Vdev_RMSE"),
        format_metric(final_metrics, "Vdev_MaxErr"),
        "",
        "[Unified safety-margin regression]",
        format_metric(final_metrics, "Hsafe_MAE"),
        format_metric(final_metrics, "Hsafe_RMSE"),
        format_metric(final_metrics, "Hsafe_MaxErr"),
        "",
        "[Unified safety-margin classification]",
        *classification_lines(final_metrics, "Hsafe"),
        "",
        "[Total active network-loss regression]",
        format_metric(final_metrics, "Ploss_MAE"),
        format_metric(final_metrics, "Ploss_RMSE"),
        format_metric(final_metrics, "Ploss_MaxErr"),
        "",
        "[Loss]",
    ]

    for key in [
        "ValBaseLoss",
        "ValTotalLoss",
        "ValVDevReg",
        "ValHsafeReg",
        "ValHSignLoss",
        "ValPLossReg",
    ]:
        lines.append(format_metric(final_metrics, key))

    lines.extend(
        [
            "",
            f"Binary count estimate: {binary_count}",
            f"best model: {best_model_path}",
            f"MILP engine: {engine_path}",
            "===============================================================",
        ]
    )

    return "\n".join(lines)


def main():
    cfg = get_config()
    set_seed(cfg["seed"])
    os.makedirs(cfg["save_dir"], exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Current device: {device}")

    data = load_pt(cfg["data_path"])
    edge_list = get_edge_list(data)

    X_raw = to_tensor(data["X"]).float().numpy()
    YV, YI_branch = get_labels(data)

    # 仅利用节点和支路标签构造三个系统级标量监督值。
    # 模型前向传播不预测节点电压或支路电流。
    YV_dev = compute_cumulative_voltage_deviation(YV)
    YV_worst = compute_worst_voltage_margin(
        YV,
        cfg["v_lower"],
        cfg["v_upper"],
    )
    YI_worst = YI_branch.max(dim=1, keepdim=True).values

    v_safe_scale = cfg["v_safe_scale"]
    i_safe_scale = cfg["i_safe_scale"]

    YH_safe = torch.maximum(
        YV_worst / v_safe_scale,
        YI_worst / i_safe_scale,
    )

    branch_r_ohm = get_branch_resistance(data, edge_list)
    line_max_i_ka = get_line_max_i_ka(data, cfg)
    YP_loss = compute_total_network_loss(
        YI_branch,
        branch_r_ohm,
        line_max_i_ka,
    )

    S_down, S_path, parent = build_topology_matrices(
        edge_list,
        33,
    )

    X_aug = torch.tensor(
        augment_path_power_features(
            X_raw,
            S_down,
            S_path,
        ),
        dtype=torch.float32,
    )

    n = len(X_aug)
    idx = torch.randperm(
        n,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    n_train = int(n * cfg["train_ratio"])

    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    Xn, YVDevn, YHsafen, YPLossn, norm = normalize_data(
        X_aug,
        YV_dev,
        YH_safe,
        YP_loss,
        train_idx,
    )

    train_ds = TensorDataset(
        Xn[train_idx],
        YVDevn[train_idx],
        YHsafen[train_idx],
        YPLossn[train_idx],
        YV_dev[train_idx],
        YH_safe[train_idx],
        YP_loss[train_idx],
    )

    val_ds = TensorDataset(
        Xn[val_idx],
        YVDevn[val_idx],
        YHsafen[val_idx],
        YPLossn[val_idx],
        YV_dev[val_idx],
        YH_safe[val_idx],
        YP_loss[val_idx],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    model = STSGCNThreeDirectScalars(
        in_features=6,
        hidden_dim=cfg["hidden_dim"],
        K=cfg["K"],
        edge_list=edge_list,
        num_nodes=33,
        vdev_head_dim=cfg["vdev_head_dim"],
        hsafe_head_dim=cfg["hsafe_head_dim"],
        ploss_head_dim=cfg["ploss_head_dim"],
        include_order0=cfg["include_order0"],
        use_sgc_relu=cfg["use_sgc_relu"],
        use_linear_skip=cfg["use_linear_skip"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

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

    v_unsafe_count = int((YV_worst > 0.0).sum().item())
    i_unsafe_count = int((YI_worst > 0.0).sum().item())
    h_unsafe_count = int((YH_safe > 0.0).sum().item())

    print("\n================ ST-SGCN Exp23 Config ================")
    print(f"Experiment: {cfg['exp_name']}")
    print("Encoder: one shared K-order SGC encoder")
    print("Readout: three independent direct scalar heads")
    print("No node-voltage prediction")
    print("No branch-current prediction")
    print("Outputs: Vdev_total, Hsafe, Ploss_total")
    print(
        "Input features: "
        "[P_net,Q_net,P_down,Q_down,P_path,Q_path]"
    )
    print(
        f"Samples: {n}, train={len(train_idx)}, "
        f"val={len(val_idx)}"
    )
    print(
        f"Vworst unsafe: {v_unsafe_count}, "
        f"ratio={100.0 * v_unsafe_count / n:.2f}%"
    )
    print(
        f"WorstI unsafe: {i_unsafe_count}, "
        f"ratio={100.0 * i_unsafe_count / n:.2f}%"
    )
    print(
        f"Hsafe unsafe: {h_unsafe_count}, "
        f"ratio={100.0 * h_unsafe_count / n:.2f}%"
    )
    print(
        f"K={cfg['K']}, shared_hidden={cfg['hidden_dim']}, "
        f"heads="
        f"{cfg['vdev_head_dim']}/"
        f"{cfg['hsafe_head_dim']}/"
        f"{cfg['ploss_head_dim']}"
    )
    print(
        f"Total active network loss: "
        f"mean={float(YP_loss.mean()):.6f} MW, "
        f"max={float(YP_loss.max()):.6f} MW"
    )
    print(
        f"Binary count estimate: "
        f"{model.get_binary_count()}"
    )
    print("=========================================================\n")

    best_model_path = os.path.join(
        cfg["save_dir"],
        cfg["best_model_name"],
    )

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()

        train_total = 0.0
        train_base = 0.0
        lam = ramp_lambda(epoch, cfg)

        for batch in train_loader:
            batch = [x.to(device) for x in batch]

            optimizer.zero_grad(set_to_none=True)

            loss, info = compute_loss(
                model,
                batch,
                norm,
                cfg,
                ramp=lam,
            )

            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["grad_clip"],
            )
            optimizer.step()

            train_total += float(loss.item())
            train_base += float(info["base"].item())

        metrics = evaluate(
            model,
            val_loader,
            norm,
            cfg,
            device,
        )

        sel = selection_metric(metrics, cfg)
        scheduler.step(sel)

        if sel < best_metric:
            best_metric = sel
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_metrics = metrics.copy()

            torch.save(
                {
                    "model_state_dict": best_state,
                    "config": cfg,
                    "norm_stats": move_to_cpu(norm),
                    "edge_list": edge_list,
                    "downstream_matrix": torch.tensor(
                        S_down,
                        dtype=torch.float32,
                    ),
                    "path_power_matrix": torch.tensor(
                        S_path,
                        dtype=torch.float32,
                    ),
                    "parent_array": torch.tensor(
                        parent,
                        dtype=torch.long,
                    ),
                    "branch_r_ohm": branch_r_ohm,
                    "line_max_i_ka": line_max_i_ka,
                    "best_metrics": best_metrics,
                    "selection_metric": float(sel),
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
                f"Epoch [{epoch:04d}/{cfg['epochs']}] | "
                f"LR={lr:.2e} | "
                f"Train={train_total / len(train_loader):.4f} | "
                f"Vdev={metrics['Vdev_MAE']:.6f} | "
                f"Hsafe={metrics['Hsafe_MAE']:.6f} | "
                f"H-FS/FV="
                f"{metrics['Hsafe_FalseSafe']:.2f}/"
                f"{metrics['Hsafe_FalseViolate']:.2f}% | "
                f"H-Acc/BAcc="
                f"{metrics['Hsafe_SignAcc']:.2f}/"
                f"{metrics['Hsafe_BalancedAcc']:.2f}% | "
                f"Ploss={metrics['Ploss_MAE']:.6f} | "
                f"SafeOK={int(metrics['SelectionSafetyFeasible'])} | "
                f"Sel={sel:.4f} | "
                f"lambda={lam:.2f}"
            )

        if wait >= cfg["patience"]:
            print(
                f"\nEarly stopping: epoch={epoch}, "
                f"best_metric={best_metric:.6f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate(
        model,
        val_loader,
        norm,
        cfg,
        device,
    )
    final_selection_metric = selection_metric(final_metrics, cfg)

    M = extract_big_m(
        model,
        Xn,
        cfg,
        device,
    )

    frozen_adj = torch.tensor(
        model.get_frozen_adj_norm(),
        dtype=torch.float32,
    )
    frozen_adj_powers = torch.tensor(
        model.get_frozen_adj_powers(),
        dtype=torch.float32,
    )

    vdev_end = cfg["vdev_head_dim"]
    hsafe_end = vdev_end + cfg["hsafe_head_dim"]
    ploss_end = hsafe_end + cfg["ploss_head_dim"]

    engine = {
        "model_type": cfg["exp_name"],
        "base_model_class": "ST-SGCN-Three-Direct-Scalars",
        "state_dict": move_to_cpu(model.state_dict()),

        "output_mode": "three_direct_system_level_scalars",
        "output_names": [
            "Vdev_total",
            "Hsafe",
            "Ploss_total",
        ],
        "predict_voltage_objective_target": (
            "Vdev_total=sum(abs(Y_V_without_slack-1.0))"
        ),
        "predict_safety_target": (
            "Hsafe=max(Vworst/v_safe_scale,WorstI/i_safe_scale)"
        ),
        "predict_network_loss_target": (
            "Ploss_total=sum(3*I_ka^2*R_ohm)"
        ),
        "safety_constraint_meaning": (
            "Hsafe_pred <= 0"
        ),
        "predicts_node_voltage": False,
        "predicts_branch_current": False,
        "uses_voltage_linear_prior": False,
        "uses_current_linear_prior": False,

        "config": cfg,
        "in_features": 6,
        "feature_names": [
            "P_net",
            "Q_net",
            "P_down",
            "Q_down",
            "P_path",
            "Q_path",
        ],
        "K": cfg["K"],
        "hidden_dim": cfg["hidden_dim"],
        "vdev_head_dim": cfg["vdev_head_dim"],
        "hsafe_head_dim": cfg["hsafe_head_dim"],
        "ploss_head_dim": cfg["ploss_head_dim"],
        "global_head_slices": {
            "Vdev": [0, vdev_end],
            "Hsafe": [vdev_end, hsafe_end],
            "Ploss": [hsafe_end, ploss_end],
        },
        "include_order0": cfg["include_order0"],
        "use_sgc_relu": cfg["use_sgc_relu"],
        "use_linear_skip": cfg["use_linear_skip"],

        "edge_list": edge_list,
        "downstream_matrix": torch.tensor(
            S_down,
            dtype=torch.float32,
        ),
        "path_power_matrix": torch.tensor(
            S_path,
            dtype=torch.float32,
        ),
        "parent_array": torch.tensor(
            parent,
            dtype=torch.long,
        ),
        "branch_r_ohm": branch_r_ohm,
        "line_max_i_ka": line_max_i_ka,
        "frozen_adj_norm": frozen_adj,
        "frozen_adj_powers": frozen_adj_powers,

        "norm_stats": move_to_cpu(norm),
        "binary_count": model.get_binary_count(),

        "safety_settings": {
            "v_lower": cfg["v_lower"],
            "v_upper": cfg["v_upper"],
            "v_safe_scale": cfg["v_safe_scale"],
            "i_safe_scale": cfg["i_safe_scale"],
            "hsafe_safety_limit": 0.0,
            "hsafe_sign_margin": 0.0,
        },

        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "selection_metric": float(final_selection_metric),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "elapsed_sec": float(time.time() - t0),
        **M,
    }

    engine_path = os.path.join(
        cfg["save_dir"],
        cfg["engine_name"],
    )
    torch.save(engine, engine_path)

    report = build_final_report(
        best_metric,
        final_metrics,
        model.get_binary_count(),
        best_model_path,
        engine_path,
    )

    print("\n" + report + "\n", flush=True)

    report_path = os.path.join(
        cfg["save_dir"],
        cfg["report_name"],
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(
        f"Final validation report: {report_path}",
        flush=True,
    )



if __name__ == "__main__":
    main()
