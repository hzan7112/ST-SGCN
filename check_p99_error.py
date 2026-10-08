import os
import random
import numpy as np
import torch
from torch.utils.data import random_split, Subset

from model.src.model import ST_MGCN


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_config():
    return {
        "data_path": r"data/st_mgcn_ieee33.pt",
        "engine_path": r"checkpoints/st_mgcn_milp_engine.pt",
        "seed": 42,
        "train_ratio": 0.8,
        "batch_size": 512,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def load_norm_stats(engine):
    norm_stats = engine["norm_stats"]

    X_mean = torch.tensor(norm_stats["X_mean"], dtype=torch.float32)
    X_std = torch.tensor(norm_stats["X_std"], dtype=torch.float32)

    # 注意：这里只对应非平衡节点 1~32
    YV_mean = torch.tensor(norm_stats["YV_mean_wo_slack"], dtype=torch.float32)
    YV_std = torch.tensor(norm_stats["YV_std_wo_slack"], dtype=torch.float32)

    YI_mean = torch.tensor(norm_stats["YI_mean"], dtype=torch.float32)
    YI_std = torch.tensor(norm_stats["YI_std"], dtype=torch.float32)

    return X_mean, X_std, YV_mean, YV_std, YI_mean, YI_std


def build_model(engine, in_features, num_nodes, device):
    cfg = engine["config"]
    edge_list = engine["edge_list"]

    model = ST_MGCN(
        in_features=in_features,
        hidden_dim=cfg["hidden_dim"],
        K=cfg["K"],
        edge_list=edge_list,
        num_nodes=num_nodes,
        node_relu_dim=cfg["node_relu_dim"],
        edge_relu_dim=cfg["edge_relu_dim"],
        edge_emb_dim=cfg["edge_emb_dim"],
        learn_edge_weight=cfg["learn_edge_weight"],
    ).to(device)

    model.load_state_dict(engine["state_dict"])
    model.eval()

    return model


def get_validation_indices(num_samples, train_ratio, seed):
    dummy_dataset = list(range(num_samples))

    train_size = int(train_ratio * num_samples)
    val_size = num_samples - train_size

    _, val_subset = random_split(
        dummy_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    return list(val_subset.indices)


def percentile(x, q):
    return float(torch.quantile(x.reshape(-1), q).cpu().item())


def main():
    cfg = get_config()
    set_seed(cfg["seed"])

    device = torch.device(cfg["device"])

    print("==============================================")
    print("ST-MGCN P99 误差评估")
    print(f"Device: {device}")
    print("==============================================")

    # ==========================================
    # 1. 加载数据与模型引擎
    # ==========================================
    data = torch.load(cfg["data_path"], map_location="cpu", weights_only=False)
    engine = torch.load(cfg["engine_path"], map_location=device, weights_only=False)

    X_raw = data["X"].float()
    Y_V_raw = data["Y_V"].float()
    Y_I_raw = data["Y_I"].float()

    num_samples, num_nodes, in_features = X_raw.shape
    num_edges = Y_I_raw.shape[1]

    print(f"数据规模: X={tuple(X_raw.shape)}, Y_V={tuple(Y_V_raw.shape)}, Y_I={tuple(Y_I_raw.shape)}")

    X_mean, X_std, YV_mean, YV_std, YI_mean, YI_std = load_norm_stats(engine)

    # ==========================================
    # 2. 使用与训练一致的验证集划分
    # ==========================================
    val_indices = get_validation_indices(
        num_samples=num_samples,
        train_ratio=cfg["train_ratio"],
        seed=cfg["seed"],
    )

    X_val_raw = X_raw[val_indices]
    YV_val_raw = Y_V_raw[val_indices]
    YI_val_raw = Y_I_raw[val_indices]

    X_val_norm = (X_val_raw - X_mean) / X_std

    print(f"验证集样本数: {len(val_indices)}")

    # ==========================================
    # 3. 加载模型
    # ==========================================
    model = build_model(
        engine=engine,
        in_features=in_features,
        num_nodes=num_nodes,
        device=device,
    )

    # ==========================================
    # 4. 分批预测
    # ==========================================
    V_pred_list = []
    I_pred_list = []

    batch_size = cfg["batch_size"]

    with torch.no_grad():
        for start in range(0, len(X_val_norm), batch_size):
            end = min(start + batch_size, len(X_val_norm))

            X_b = X_val_norm[start:end].to(device)

            V_pred_n, I_pred_n, _, _ = model(X_b)

            # 反标准化电压，只评估非平衡节点 1~32
            V_pred_phys = V_pred_n[:, 1:] * YV_std.to(device) + YV_mean.to(device)

            # 反标准化线路裕度
            I_pred_phys = I_pred_n * YI_std.to(device) + YI_mean.to(device)

            V_pred_list.append(V_pred_phys.cpu())
            I_pred_list.append(I_pred_phys.cpu())

    V_pred = torch.cat(V_pred_list, dim=0)   # [N_val, 32]
    I_pred = torch.cat(I_pred_list, dim=0)   # [N_val, 32]

    V_true = YV_val_raw[:, 1:]               # [N_val, 32]
    I_true = YI_val_raw                      # [N_val, 32]

    # ==========================================
    # 5. 误差统计
    # ==========================================
    V_abs_err = torch.abs(V_pred - V_true)
    I_abs_err = torch.abs(I_pred - I_true)

    V_p95 = percentile(V_abs_err, 0.95)
    V_p99 = percentile(V_abs_err, 0.99)
    V_max = float(V_abs_err.max().item())
    V_mae = float(V_abs_err.mean().item())
    V_rmse = float(torch.sqrt(torch.mean((V_pred - V_true) ** 2)).item())

    I_p95 = percentile(I_abs_err, 0.95)
    I_p99 = percentile(I_abs_err, 0.99)
    I_max = float(I_abs_err.max().item())
    I_mae = float(I_abs_err.mean().item())
    I_rmse = float(torch.sqrt(torch.mean((I_pred - I_true) ** 2)).item())

    print("\n================ 全局误差统计 ================")
    print(f"电压 MAE        : {V_mae:.6f} p.u.")
    print(f"电压 RMSE       : {V_rmse:.6f} p.u.")
    print(f"电压 P95误差    : {V_p95:.6f} p.u.")
    print(f"电压 P99误差    : {V_p99:.6f} p.u.")
    print(f"电压 Max误差    : {V_max:.6f} p.u.")

    print(f"\n裕度 MAE        : {I_mae:.6f}")
    print(f"裕度 RMSE       : {I_rmse:.6f}")
    print(f"裕度 P95误差    : {I_p95:.6f}")
    print(f"裕度 P99误差    : {I_p99:.6f}")
    print(f"裕度 Max误差    : {I_max:.6f}")
    print("==============================================")

    # ==========================================
    # 6. 定位最大误差
    # ==========================================
    v_flat_idx = int(torch.argmax(V_abs_err).item())
    v_sample_idx = v_flat_idx // V_abs_err.shape[1]
    v_node_idx_wo_slack = v_flat_idx % V_abs_err.shape[1]
    v_node_idx = v_node_idx_wo_slack + 1

    i_flat_idx = int(torch.argmax(I_abs_err).item())
    i_sample_idx = i_flat_idx // I_abs_err.shape[1]
    i_branch_idx = i_flat_idx % I_abs_err.shape[1]

    print("\n================ 最大误差定位 ================")
    print(
        f"电压最大误差: sample={v_sample_idx}, "
        f"原始样本编号={val_indices[v_sample_idx]}, "
        f"节点={v_node_idx + 1}, "
        f"真实值={V_true[v_sample_idx, v_node_idx_wo_slack]:.6f}, "
        f"预测值={V_pred[v_sample_idx, v_node_idx_wo_slack]:.6f}, "
        f"误差={V_abs_err[v_sample_idx, v_node_idx_wo_slack]:.6f}"
    )

    print(
        f"裕度最大误差: sample={i_sample_idx}, "
        f"原始样本编号={val_indices[i_sample_idx]}, "
        f"支路索引={i_branch_idx}, "
        f"真实值={I_true[i_sample_idx, i_branch_idx]:.6f}, "
        f"预测值={I_pred[i_sample_idx, i_branch_idx]:.6f}, "
        f"误差={I_abs_err[i_sample_idx, i_branch_idx]:.6f}"
    )
    print("==============================================")

    # ==========================================
    # 7. 按节点 / 支路统计 P99
    # ==========================================
    V_p99_by_node = torch.quantile(V_abs_err, 0.99, dim=0)
    I_p99_by_branch = torch.quantile(I_abs_err, 0.99, dim=0)

    print("\n================ 按节点电压 P99 误差 ================")
    for i, err in enumerate(V_p99_by_node):
        # i 对应非平衡节点索引 1~32，即实际 Python 节点 i+1，论文节点编号 i+2
        print(f"节点 {i + 2:02d}: P99电压误差 = {err.item():.6f} p.u.")

    print("\n================ 按支路裕度 P99 误差 ================")
    edge_list = engine["edge_list"]
    for e_idx, err in enumerate(I_p99_by_branch):
        f_bus, t_bus = edge_list[e_idx]
        print(
            f"支路 {e_idx:02d} ({f_bus + 1:02d}->{t_bus + 1:02d}): "
            f"P99裕度误差 = {err.item():.6f}"
        )

    # ==========================================
    # 8. 保存结果
    # ==========================================
    save_path = os.path.join("checkpoints", "p99_error_report.npz")

    np.savez(
        save_path,
        V_abs_err=V_abs_err.numpy(),
        I_abs_err=I_abs_err.numpy(),
        V_p99=V_p99,
        I_p99=I_p99,
        V_p99_by_node=V_p99_by_node.numpy(),
        I_p99_by_branch=I_p99_by_branch.numpy(),
        val_indices=np.array(val_indices),
    )

    print(f"\n误差报告已保存至: {save_path}")


if __name__ == "__main__":
    main()