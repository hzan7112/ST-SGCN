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
    __package__ = "model.exp18"

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
    get_branch_resistance,
    get_line_max_i_ka,
    compute_total_network_loss,
    normalize_data,
    fit_voltage_linear_prior,
    eval_voltage_prior,
)
from .loss import compute_loss
from .model import (
    STSGCNWorstI,
    VoltageLinearResidualWrapper,
    denorm_outputs,
    unpack_forward,
    zero_init_voltage_head,
)


def get_config():
    return {
        "exp_name": "ST_SGCN_Exp18_K4_H24_n2_g32_WorstI_PlossShared_CurrentSign_VoltageSafety",
        "data_path": r"data/ieee33_nodal_pq_correlated_raw_pool_50k.pt",
        "save_dir": r"checkpoints",
        "best_model_name": "st_sgcn_k4_h24_n2_g32_exp18_worsti_ploss_shared_best.pt",
        "engine_name": "st_sgcn_k4_h24_n2_g32_exp18_worsti_ploss_shared_milp_engine.pt",

        "seed": 42,
        "train_ratio": 0.8,
        "batch_size": 256,
        "epochs": 800,
        "patience": 180,

        # 鎺ㄨ崘涓荤粨鏋勶細Exp15 楠ㄦ灦
        "K": 4,
        "hidden_dim": 24,
        "node_relu_dim": 2,
        "global_relu_dim": 32,

        "include_order0": True,
        "use_sgc_relu": False,
        "use_linear_skip": True,

        "ridge_alpha": 1e-3,
        "zero_init_voltage_residual_head": True,

        "lr": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,

        # 鍥炲綊椤?
        "voltage_loss_weight": 5.0,
        "worst_i_loss_weight": 2.0,
        "ploss_loss_weight": 1.0,
        "voltage_violate_mse_weight": 6.0,
        "worst_i_violate_mse_weight": 5.0,
        "default_line_max_i_ka": 0.20,

        "warmup_epochs": 80,
        "penalty_ramp_epochs": 120,

        # 鐢垫祦绗﹀彿鍒嗙被澧炲己锛?
        # YI_worst > 0 琛ㄧず鐢垫祦瓒婇檺锛沋I_worst <= 0 琛ㄧず鐢垫祦瀹夊叏銆?
        # 杩欓噷涓嶇敤鍗曚晶 FS 鎯╃綒锛岃€岀敤瀵圭О sign loss 鍚屾椂绾︽潫 FS/FV銆?
        "worst_i_false_safe_lambda": 0.0,
        "worst_i_sign_loss_weight": 8.0,
        "worst_i_sign_margin": 0.003,
        "worst_i_false_safe_margin": 0.006,

        # 鐢靛帇涓夊尯鍩熷畨鍏ㄥ垎绫诲寮猴細
        # 浣庡帇瓒婇檺锛歏 < v_lower锛涘畨鍏細v_lower <= V <= v_upper锛涢珮鍘嬭秺闄愶細V > v_upper銆?
        # 涓轰簡璁?p.u. 閲忕骇鐨勮竟鐣屾崯澶辩湡姝ｅ弬涓庤缁冿紝杩欓噷灏嗚竟鐣岃窛绂婚櫎浠?v_safety_scale 鍚庡啀骞虫柟銆?
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


def evaluate(model, loader, norm, cfg, device):
    model.eval()

    V_pred_all = []
    I_pred_all = []
    P_pred_all = []
    V_true_all = []
    I_true_all = []
    P_true_all = []

    base_sum = 0.0
    total_sum = 0.0
    node_sum = 0.0
    i_sum = 0.0
    p_sum = 0.0
    i_sign_sum = 0.0
    v_safety_sum = 0.0
    n_batch = 0

    for batch in loader:
        batch = [x.to(device) for x in batch]

        loss, info = compute_loss(model, batch, norm, cfg, ramp=1.0)

        X, _, _, _, YV, YI_worst, YP_loss = batch

        Vn, YI_worst_n, Ploss_n, *_ = unpack_forward(model, X)
        V, YI_pred, Ploss_pred = denorm_outputs(Vn, YI_worst_n, Ploss_n, norm)

        V_pred_all.append(V.cpu())
        I_pred_all.append(YI_pred.cpu())
        P_pred_all.append(Ploss_pred.cpu())

        V_true_all.append(YV.cpu())
        I_true_all.append(YI_worst.cpu())
        P_true_all.append(YP_loss.cpu())

        total_sum += float(loss.item())
        base_sum += float(info["base"].item())
        node_sum += float(info["node_mse"].item())
        i_sum += float(info["worst_i_mse"].item())
        p_sum += float(info["ploss_mse"].item())
        i_sign_sum += float(info["i_sign_loss"].item())
        v_safety_sum += float(info["v_safety_loss"].item())
        n_batch += 1

    Vp = torch.cat(V_pred_all)
    Ip = torch.cat(I_pred_all)
    Pp = torch.cat(P_pred_all)

    Vt = torch.cat(V_true_all)
    It = torch.cat(I_true_all)
    Pt = torch.cat(P_true_all)

    v_err = torch.abs(Vp[:, 1:] - Vt[:, 1:])
    i_err = torch.abs(Ip - It)
    p_err = torch.abs(Pp - Pt)

    v_true_unsafe = (Vt[:, 1:] < cfg["v_lower"]) | (Vt[:, 1:] > cfg["v_upper"])
    v_pred_unsafe = (Vp[:, 1:] < cfg["v_lower"]) | (Vp[:, 1:] > cfg["v_upper"])

    i_true_unsafe = It > 0.0
    i_pred_unsafe = Ip > 0.0

    v_fs = (
        (v_true_unsafe & (~v_pred_unsafe)).sum().item()
        / max(v_true_unsafe.sum().item(), 1)
        * 100.0
    )

    v_fv = (
        ((~v_true_unsafe) & v_pred_unsafe).sum().item()
        / max((~v_true_unsafe).sum().item(), 1)
        * 100.0
    )

    i_fs = (
        (i_true_unsafe & (~i_pred_unsafe)).sum().item()
        / max(i_true_unsafe.sum().item(), 1)
        * 100.0
    )

    i_fv = (
        ((~i_true_unsafe) & i_pred_unsafe).sum().item()
        / max((~i_true_unsafe).sum().item(), 1)
        * 100.0
    )

    # 鐢垫祦绗﹀彿鍒嗙被鎸囨爣銆?
    i_tp = (i_true_unsafe & i_pred_unsafe).sum().item()
    i_fn = (i_true_unsafe & (~i_pred_unsafe)).sum().item()
    i_fp = ((~i_true_unsafe) & i_pred_unsafe).sum().item()
    i_tn = ((~i_true_unsafe) & (~i_pred_unsafe)).sum().item()

    i_total = max(i_tp + i_tn + i_fp + i_fn, 1)
    i_unsafe_total = max(i_tp + i_fn, 1)
    i_safe_total = max(i_tn + i_fp, 1)

    i_sign_acc = (i_tp + i_tn) / i_total * 100.0
    i_unsafe_recall = i_tp / i_unsafe_total * 100.0
    i_safe_recall = i_tn / i_safe_total * 100.0
    i_balanced_acc = 0.5 * (i_unsafe_recall + i_safe_recall)

    # 鐢靛帇瀹夊叏鍒嗙被鎸囨爣锛屾寜鏍锋湰-鑺傜偣瀵圭粺璁°€?
    v_tp = (v_true_unsafe & v_pred_unsafe).sum().item()
    v_fn = (v_true_unsafe & (~v_pred_unsafe)).sum().item()
    v_fp = ((~v_true_unsafe) & v_pred_unsafe).sum().item()
    v_tn = ((~v_true_unsafe) & (~v_pred_unsafe)).sum().item()

    v_total = max(v_tp + v_tn + v_fp + v_fn, 1)
    v_unsafe_total = max(v_tp + v_fn, 1)
    v_safe_total = max(v_tn + v_fp, 1)

    v_safety_acc = (v_tp + v_tn) / v_total * 100.0
    v_unsafe_recall = v_tp / v_unsafe_total * 100.0
    v_safe_recall = v_tn / v_safe_total * 100.0
    v_balanced_acc = 0.5 * (v_unsafe_recall + v_safe_recall)

    return {
        "ValBaseLoss": base_sum / n_batch,
        "ValTotalLoss": total_sum / n_batch,
        "ValNodeMSE": node_sum / n_batch,
        "ValWorstIMSE": i_sum / n_batch,
        "ValPlossMSE": p_sum / n_batch,
        "ValISignLoss": i_sign_sum / n_batch,
        "ValVSafetyLoss": v_safety_sum / n_batch,

        "V_MAE": float(v_err.mean()),
        "V_RMSE": float(torch.sqrt(torch.mean((Vp[:, 1:] - Vt[:, 1:]) ** 2))),
        "V_MaxErr": float(v_err.max()),

        "WorstI_MAE": float(i_err.mean()),
        "WorstI_RMSE": float(torch.sqrt(torch.mean((Ip - It) ** 2))),
        "WorstI_MaxErr": float(i_err.max()),

        "Ploss_MAE": float(p_err.mean()),
        "Ploss_RMSE": float(torch.sqrt(torch.mean((Pp - Pt) ** 2))),
        "Ploss_MaxErr": float(p_err.max()),

        "V_FalseSafe": v_fs,
        "V_FalseViolate": v_fv,
        "V_SafetyAcc": v_safety_acc,
        "V_BalancedAcc": v_balanced_acc,
        "V_UnsafeRecall": v_unsafe_recall,
        "V_SafeRecall": v_safe_recall,
        "V_TrueUnsafeCount": int(v_true_unsafe.sum().item()),
        "V_PredUnsafeCount": int(v_pred_unsafe.sum().item()),
        "V_TP": int(v_tp),
        "V_TN": int(v_tn),
        "V_FP": int(v_fp),
        "V_FN": int(v_fn),

        "WorstI_FalseSafe": i_fs,
        "WorstI_FalseViolate": i_fv,
        "WorstI_SignAcc": i_sign_acc,
        "WorstI_BalancedAcc": i_balanced_acc,
        "WorstI_UnsafeRecall": i_unsafe_recall,
        "WorstI_SafeRecall": i_safe_recall,

        "WorstI_TrueUnsafeCount": int(i_true_unsafe.sum().item()),
        "WorstI_PredUnsafeCount": int(i_pred_unsafe.sum().item()),
        "WorstI_TP": int(i_tp),
        "WorstI_TN": int(i_tn),
        "WorstI_FP": int(i_fp),
        "WorstI_FN": int(i_fn),
    }


def ramp_lambda(epoch, cfg):
    if epoch <= cfg["warmup_epochs"]:
        return 0.0

    x = (epoch - cfg["warmup_epochs"]) / max(cfg["penalty_ramp_epochs"], 1)

    return float(min(max(x, 0.0), 1.0))


def selection_metric(m):
    # 瀹夊叏鍒嗙被鎰熺煡閫夋嫨鎸囨爣锛?
    # 1) 淇濈暀 V_MAE 涓?WorstI_MAE锛?
    # 2) 鍚屾椂绾︽潫 FS/FV锛?
    # 3) 褰撳墠閲嶇偣鏄檷浣庣數鍘嬪亣瀹夊叏锛屽洜姝?V_FalseSafe 鏉冮噸鐣ラ珮锛?
    # 4) 鍔犲叆 V/WorstI balanced accuracy锛岄伩鍏嶅彧鍋忓悜鍗曚晶淇濆畧銆?
    return (
        m["V_MAE"]
        + 0.2 * m["WorstI_MAE"]
        + 0.5 * m["Ploss_MAE"]
        + 0.0030 * m["V_FalseSafe"]
        + 0.0015 * m["V_FalseViolate"]
        + 0.0020 * m["WorstI_FalseSafe"]
        + 0.0015 * m["WorstI_FalseViolate"]
        + 0.0010 * (100.0 - m["V_BalancedAcc"])
        + 0.0008 * (100.0 - m["WorstI_BalancedAcc"])
    )


def extract_big_m(model, Xn, cfg, device):
    model.eval()

    batch_size = cfg["batch_size"]

    gcn_z_all = None
    node_z_all = []
    global_z_all = []

    for i in range(0, len(Xn), batch_size):
        xb = Xn[i:i + batch_size].to(device)

        out = unpack_forward(model, xb)

        if len(out) < 6:
            raise RuntimeError("model forward must return gcn_Z_list, Z_node, Z_global for Big-M extraction.")

        gcn_Z_list = out[3]
        Z_node = out[4]
        Z_global = out[5]

        if gcn_z_all is None:
            gcn_z_all = [[] for _ in range(len(gcn_Z_list))]

        for k, z in enumerate(gcn_Z_list):
            gcn_z_all[k].append(z.detach().cpu())

        node_z_all.append(Z_node.detach().cpu())
        global_z_all.append(Z_global.detach().cpu())

    beta = cfg["big_m_beta"]

    M_plus_gcn = []
    M_minus_gcn = []

    if gcn_z_all is not None:
        for z_list in gcn_z_all:
            Z = torch.cat(z_list, dim=0)
            M_plus_gcn.append(torch.clamp(Z.max(dim=0).values, min=0.0) * beta)
            M_minus_gcn.append(torch.clamp((-Z).max(dim=0).values, min=0.0) * beta)

    Z_node = torch.cat(node_z_all, dim=0)
    Z_global = torch.cat(global_z_all, dim=0)

    return {
        "M_plus_gcn_layers": M_plus_gcn,
        "M_minus_gcn_layers": M_minus_gcn,
        "M_plus_node": torch.clamp(Z_node.max(dim=0).values, min=0.0) * beta,
        "M_minus_node": torch.clamp((-Z_node).max(dim=0).values, min=0.0) * beta,
        "M_plus_global": torch.clamp(Z_global.max(dim=0).values, min=0.0) * beta,
        "M_minus_global": torch.clamp((-Z_global).max(dim=0).values, min=0.0) * beta,
    }


def move_norm_to_cpu(norm):
    return {k: v.cpu() for k, v in norm.items()}


def main():
    cfg = get_config()

    set_seed(cfg["seed"])
    os.makedirs(cfg["save_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"褰撳墠璁惧: {device}")

    data = load_pt(cfg["data_path"])
    edge_list = get_edge_list(data)

    X_raw = to_tensor(data["X"]).float().numpy()
    YV, YI_branch = get_labels(data)

    YI_worst = YI_branch.max(dim=1, keepdim=True).values
    branch_r_ohm = get_branch_resistance(data, edge_list)
    line_max_i_ka = get_line_max_i_ka(data, cfg["default_line_max_i_ka"])
    YP_loss = compute_total_network_loss(YI_branch, branch_r_ohm, line_max_i_ka)

    unsafe_count = int((YI_worst > 0.0).sum().item())
    total_count = int(YI_worst.numel())

    S_down, S_path, parent = build_topology_matrices(edge_list, 33)

    X_aug = torch.tensor(
        augment_path_power_features(X_raw, S_down, S_path),
        dtype=torch.float32,
    )

    n = len(X_aug)

    idx = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"]))
    n_train = int(n * cfg["train_ratio"])

    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    Xn, YVn, YIn, YPn, norm = normalize_data(X_aug, YV, YI_worst, YP_loss, train_idx)

    W_vlin, b_vlin = fit_voltage_linear_prior(
        Xn=Xn,
        YVn=YVn,
        train_idx=train_idx,
        alpha=cfg["ridge_alpha"],
    )

    vlin_mae, vlin_rmse, vlin_max = eval_voltage_prior(
        Xn=Xn,
        YV=YV,
        W=W_vlin,
        b=b_vlin,
        norm=norm,
        idx=val_idx,
    )

    train_ds = TensorDataset(
        Xn[train_idx],
        YVn[train_idx],
        YIn[train_idx],
        YPn[train_idx],
        YV[train_idx],
        YI_worst[train_idx],
        YP_loss[train_idx],
    )

    val_ds = TensorDataset(
        Xn[val_idx],
        YVn[val_idx],
        YIn[val_idx],
        YPn[val_idx],
        YV[val_idx],
        YI_worst[val_idx],
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

    base_model = STSGCNWorstI(
        in_features=6,
        hidden_dim=cfg["hidden_dim"],
        K=cfg["K"],
        edge_list=edge_list,
        num_nodes=33,
        node_relu_dim=cfg["node_relu_dim"],
        global_relu_dim=cfg["global_relu_dim"],
        include_order0=cfg["include_order0"],
        use_sgc_relu=cfg["use_sgc_relu"],
        use_linear_skip=cfg["use_linear_skip"],
    )

    if cfg["zero_init_voltage_residual_head"]:
        zero_init_voltage_head(base_model)

    model = VoltageLinearResidualWrapper(
        base_model=base_model,
        W_vlin=W_vlin,
        b_vlin=b_vlin,
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
        patience=25,
    )

    best_metric = float("inf")
    best_state = None
    best_metrics = None
    wait = 0
    t0 = time.time()

    print("\n================ ST-SGCN Exp18 瀹為獙閰嶇疆 ================")
    print(f"瀹為獙: {cfg['exp_name']}")
    print("妯″瀷: ST-SGCN-WorstI-PlossShared")
    print("杈撳叆鐗瑰緛: [P_net,Q_net,P_down,Q_down,P_path,Q_path]")
    print("鐢靛帇杈撳嚭: 閫愯妭鐐圭數鍘嬶紝V_pred = V_linear_prior + ST-SGCN_voltage_residual")
    print("鐢垫祦杈撳嚭: 鍏ㄧ綉鏈€鍗遍櫓鐢垫祦瑁曞害锛孻I_worst=max(YI_branch)")
    print("浼樺寲绾︽潫鍚箟: YI_worst_pred <= 0 琛ㄧず棰勬祴鍏ㄧ綉鐢垫祦瀹夊叏")
    print("璁粌鐩爣澧炲己: 瀵圭О鐢垫祦绗﹀彿鍒嗙被鎹熷け锛屽悓鏃舵儵缃?WorstI-FS 鍜?WorstI-FV")
    print(f"鏁版嵁闆嗘牱鏈暟: {total_count}")
    print(f"YI_worst 瓒婇檺鏍锋湰鏁? {unsafe_count}, 鍗犳瘮: {unsafe_count / max(total_count, 1) * 100:.2f}%")
    print(f"绾挎€х數鍘嬪厛楠?Val: MAE={vlin_mae:.6f}, RMSE={vlin_rmse:.6f}, MaxErr={vlin_max:.6f}")
    print(f"K={cfg['K']}, hidden={cfg['hidden_dim']}, node_head={cfg['node_relu_dim']}, global_head={cfg['global_relu_dim']}")
    print(f"use_sgc_relu={cfg['use_sgc_relu']}, include_order0={cfg['include_order0']}")
    print(f"浜屽厓鍙橀噺浼拌: {model.get_binary_count()}")
    print("=========================================================\n")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()

        total_train = 0.0
        base_train = 0.0
        lam = ramp_lambda(epoch, cfg)

        for batch in train_loader:
            batch = [x.to(device) for x in batch]

            optimizer.zero_grad()

            loss, info = compute_loss(
                model=model,
                batch=batch,
                norm=norm,
                cfg=cfg,
                ramp=lam,
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["grad_clip"],
            )

            optimizer.step()

            total_train += float(loss.item())
            base_train += float(info["base"].item())

        metrics = evaluate(
            model=model,
            loader=val_loader,
            norm=norm,
            cfg=cfg,
            device=device,
        )

        sel = selection_metric(metrics)
        scheduler.step(sel)

        train_base = base_train / len(train_loader)
        train_total = total_train / len(train_loader)

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
                    "base_state_dict": model.base.state_dict(),
                    "voltage_linear_prior_W": model.W_vlin.detach().cpu(),
                    "voltage_linear_prior_b": model.b_vlin.detach().cpu(),
                    "config": cfg,
                    "norm_stats": move_norm_to_cpu(norm),
                    "edge_list": edge_list,
                    "branch_r_ohm": branch_r_ohm,
                    "line_max_i_ka": float(line_max_i_ka),
                    "downstream_matrix": torch.tensor(S_down, dtype=torch.float32),
                    "path_power_matrix": torch.tensor(S_path, dtype=torch.float32),
                    "parent_array": torch.tensor(parent, dtype=torch.long),
                    "best_metrics": best_metrics,
                    "binary_count": model.get_binary_count(),
                },
                os.path.join(cfg["save_dir"], cfg["best_model_name"]),
            )

            wait = 0

        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch:03d}/{cfg['epochs']}] | LR={lr:.2e} | "
                f"NodeMSE={metrics['ValNodeMSE']:.4f} | WorstIMSE={metrics['ValWorstIMSE']:.4f} | "
                f"PlossMSE={metrics['ValPlossMSE']:.4f} | "
                f"ISign={metrics['ValISignLoss']:.4f} | VSafety={metrics['ValVSafetyLoss']:.4f} | "
                f"V_MAE={metrics['V_MAE']:.6f} | WorstI_MAE={metrics['WorstI_MAE']:.6f} | "
                f"Ploss_MAE={metrics['Ploss_MAE']:.6f} | "
                f"V-FS={metrics['V_FalseSafe']:.2f}% | "
                f"V-FV={metrics['V_FalseViolate']:.2f}% | "
                f"V-BAcc={metrics['V_BalancedAcc']:.2f}% | "
                f"WorstI-FS={metrics['WorstI_FalseSafe']:.2f}% | "
                f"WorstI-FV={metrics['WorstI_FalseViolate']:.2f}% | "
                f"WorstI-BAcc={metrics['WorstI_BalancedAcc']:.2f}% | "
                f"位I-sign={cfg['worst_i_sign_loss_weight'] * lam:.3f} | "
                f"位V-safe={cfg['v_safety_loss_weight'] * lam:.3f}"
            )

        if wait >= cfg["patience"]:
            print(f"\n鏃╁仠瑙﹀彂: epoch={epoch}, best_metric={best_metric:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate(
        model=model,
        loader=val_loader,
        norm=norm,
        cfg=cfg,
        device=device,
    )

    M = extract_big_m(
        model=model,
        Xn=Xn,
        cfg=cfg,
        device=device,
    )

    frozen_adj = torch.tensor(model.get_frozen_adj_norm(), dtype=torch.float32)
    frozen_adj_powers = torch.tensor(model.get_frozen_adj_powers(), dtype=torch.float32)

    engine = {
        "model_type": cfg["exp_name"],
        "base_model_class": "ST-SGCN-WorstI-PlossShared",
        "state_dict": model.base.state_dict(),
        "wrapper_state_dict": model.state_dict(),

        "predict_current_target": "YI_worst=max(Y_I_branch)",
        "predict_loss_target": "Ploss_total=sum(3*I_branch^2*r_branch)",
        "ploss_head": "Ploss shares global_hidden/ReLU with WorstI and uses its own linear output head",
        "current_constraint_meaning": "YI_worst_pred <= 0 implies predicted system-level current safety",

        "uses_voltage_linear_prior": True,
        "voltage_linear_prior_input": "normalized_flattened_X6",
        "voltage_linear_prior_target": "normalized_voltage_without_slack",
        "voltage_linear_prior_W": model.W_vlin.detach().cpu(),
        "voltage_linear_prior_b": model.b_vlin.detach().cpu(),

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
        "node_relu_dim": cfg["node_relu_dim"],
        "global_relu_dim": cfg["global_relu_dim"],
        "include_order0": cfg["include_order0"],
        "use_sgc_relu": cfg["use_sgc_relu"],
        "use_linear_skip": cfg["use_linear_skip"],

        "edge_list": edge_list,
        "branch_r_ohm": branch_r_ohm,
        "line_max_i_ka": float(line_max_i_ka),
        "downstream_matrix": torch.tensor(S_down, dtype=torch.float32),
        "path_power_matrix": torch.tensor(S_path, dtype=torch.float32),
        "parent_array": torch.tensor(parent, dtype=torch.long),
        "frozen_adj_norm": frozen_adj,
        "frozen_adj_powers": frozen_adj_powers,

        "norm_stats": move_norm_to_cpu(norm),
        "binary_count": model.get_binary_count(),

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
            "worst_i_sign_margin": cfg["worst_i_sign_margin"],
            "worst_i_sign_loss_weight": cfg["worst_i_sign_loss_weight"],
        },

        "linear_prior_metrics": {
            "V_MAE": vlin_mae,
            "V_RMSE": vlin_rmse,
            "V_MaxErr": vlin_max,
        },

        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "elapsed_sec": float(time.time() - t0),

        **M,
    }

    torch.save(
        engine,
        os.path.join(cfg["save_dir"], cfg["engine_name"]),
    )

    print("\n================ ST-SGCN 鏈€缁堥獙璇佺粨鏋?================")
    print(f"SelectionMetric: {best_metric:.6f}")
    print(f"LinearPrior_V_MAE: {vlin_mae:.6f}")
    print(f"LinearPrior_V_RMSE: {vlin_rmse:.6f}")
    print(f"LinearPrior_V_MaxErr: {vlin_max:.6f}")

    for k, v in final_metrics.items():
        if "False" in k:
            print(f"{k}: {v:.2f}%")
        elif "Count" in k:
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.6f}")

    print(f"浜屽厓鍙橀噺浼拌: {model.get_binary_count()}")
    print(f"best model: {os.path.join(cfg['save_dir'], cfg['best_model_name'])}")
    print(f"MILP engine: {os.path.join(cfg['save_dir'], cfg['engine_name'])}")
    print("============================================================\n")



if __name__ == "__main__":
    main()

