import argparse
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from train_mlp import (
    build_base_model,
    compute_cumulative_voltage_deviation,
    compute_total_network_loss,
    compute_worst_voltage_margin,
    denorm_outputs,
    get_branch_resistance,
    get_config,
    get_edge_list,
    get_labels,
    get_line_max_i_ka,
    load_pt,
    normalize_data,
    to_tensor,
    unpack_forward,
)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def regression_row(name, pred, true, eps=1e-8):
    pred = pred.detach().float().reshape(-1).cpu()
    true = true.detach().float().reshape(-1).cpu()
    err = pred - true
    mae = torch.mean(torch.abs(err))
    mape = torch.mean(torch.abs(err) / true.abs().clamp_min(eps)) * 100.0
    rmse = torch.sqrt(torch.mean(err.square()))
    nrmse = rmse / true.std(unbiased=False).clamp_min(eps) * 100.0
    return {
        "Target": name,
        "MAE": float(mae),
        "MAPE(%)": float(mape),
        "NRMSE(%)": float(nrmse),
    }


def classification_row(name, pred, true):
    pred = pred.detach().float().reshape(-1).cpu()
    true = true.detach().float().reshape(-1).cpu()
    true_unsafe = true > 0.0
    pred_unsafe = pred > 0.0

    tp = torch.sum(true_unsafe & pred_unsafe).item()
    tn = torch.sum((~true_unsafe) & (~pred_unsafe)).item()
    fp = torch.sum((~true_unsafe) & pred_unsafe).item()
    fn = torch.sum(true_unsafe & (~pred_unsafe)).item()

    unsafe_total = max(tp + fn, 1)
    safe_total = max(tn + fp, 1)
    total = max(tp + tn + fp + fn, 1)
    unsafe_recall = tp / unsafe_total * 100.0
    safe_recall = tn / safe_total * 100.0

    return {
        "Target": name,
        "MAE": float(torch.mean(torch.abs(pred - true))),
        "FalseSafe(%)": fn / unsafe_total * 100.0,
        "FalseViolate(%)": fp / safe_total * 100.0,
        "SignAcc(%)": (tp + tn) / total * 100.0,
        "BalancedAcc(%)": 0.5 * (unsafe_recall + safe_recall),
    }


def print_table(title, rows, columns):
    print(f"\n{title}")
    if not rows:
        print("(no metrics)")
        return

    def fmt(value):
        if isinstance(value, str):
            return value
        return f"{value:.6f}"

    widths = []
    for col in columns:
        values = [fmt(row[col]) for row in rows]
        widths.append(max(len(col), *(len(v) for v in values)))

    sep = "+".join("-" * (width + 2) for width in widths)
    header = " | ".join(col.ljust(widths[i]) for i, col in enumerate(columns))
    print(f"+{sep}+")
    print(f"| {header} |")
    print(f"+{sep}+")

    for row in rows:
        line = " | ".join(fmt(row[col]).rjust(widths[i]) for i, col in enumerate(columns))
        print(f"| {line} |")

    print(f"+{sep}+")


def build_eval_dataset(cfg):
    data = load_pt(cfg["data_path"])
    edge_list = get_edge_list(data)
    X_raw = to_tensor(data["X"]).float()[:, :, :2].contiguous()
    YV, YI_branch = get_labels(data)

    YV_dev = compute_cumulative_voltage_deviation(YV)
    YV_worst = compute_worst_voltage_margin(YV, cfg["v_lower"], cfg["v_upper"])
    YI_worst = YI_branch.max(dim=1, keepdim=True).values
    YH_safe = torch.maximum(
        YV_worst / cfg["v_safe_scale"],
        YI_worst / cfg["i_safe_scale"],
    )
    branch_r_ohm = get_branch_resistance(data, edge_list)
    line_max_i_ka = get_line_max_i_ka(data, cfg)
    YP_loss = compute_total_network_loss(YI_branch, branch_r_ohm, line_max_i_ka)

    idx = torch.randperm(len(X_raw), generator=torch.Generator().manual_seed(cfg["seed"]))
    n_train = int(len(X_raw) * cfg["train_ratio"])
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    Xn, YVDevn, YHsafen, YPLossn, norm = normalize_data(
        X_raw,
        YV_dev,
        YH_safe,
        YP_loss,
        train_idx,
    )
    dataset = TensorDataset(
        Xn[test_idx],
        YVDevn[test_idx],
        YHsafen[test_idx],
        YPLossn[test_idx],
        YV_dev[test_idx],
        YH_safe[test_idx],
        YP_loss[test_idx],
    )
    return edge_list, norm, dataset, len(train_idx), len(test_idx)


@torch.no_grad()
def collect_metrics(model, loader, norm, device):
    pred_vdev, true_vdev = [], []
    pred_hsafe, true_hsafe = [], []
    pred_ploss, true_ploss = [], []

    model.eval()
    for batch in loader:
        batch = [item.to(device) for item in batch]
        X = batch[0]
        Vdev_n, Hsafe_n, Ploss_n, *_ = unpack_forward(model, X)
        Vdev, Hsafe, Ploss = denorm_outputs(
            Vdev_n,
            Hsafe_n,
            Ploss_n,
            norm,
        )

        pred_vdev.append(Vdev.cpu())
        pred_hsafe.append(Hsafe.cpu())
        pred_ploss.append(Ploss.cpu())
        true_vdev.append(batch[5].cpu())
        true_hsafe.append(batch[6].cpu())
        true_ploss.append(batch[7].cpu())

    pred_vdev = torch.cat(pred_vdev)
    true_vdev = torch.cat(true_vdev)
    pred_hsafe = torch.cat(pred_hsafe)
    true_hsafe = torch.cat(true_hsafe)
    pred_ploss = torch.cat(pred_ploss)
    true_ploss = torch.cat(true_ploss)

    reg_rows = [
        regression_row("VoltageDeviation", pred_vdev, true_vdev),
        regression_row("Hsafe", pred_hsafe, true_hsafe),
        regression_row("NetworkLoss", pred_ploss, true_ploss),
    ]
    cls_rows = [
        classification_row("Hsafe", pred_hsafe, true_hsafe),
    ]
    return reg_rows, cls_rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate the MLP three-scalar checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to config best_model_name.")
    parser.add_argument("--device", default=None, help="Device, for example cpu or cuda.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size.")
    args = parser.parse_args()

    cfg = get_config()
    configured_data_path = cfg["data_path"]
    checkpoint_path = args.checkpoint or os.path.join(cfg["save_dir"], cfg["best_model_name"])
    checkpoint = load_checkpoint(checkpoint_path)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        cfg.update(checkpoint["config"])
        cfg["data_path"] = configured_data_path
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _, norm, dataset, train_size, test_size = build_eval_dataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, drop_last=False)

    model = build_base_model(cfg).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint does not contain model_state_dict or state_dict.")
    model.load_state_dict(state_dict)

    reg_rows, cls_rows = collect_metrics(model, loader, norm, device)

    print(f"\nExperiment: model.mlp")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {cfg['data_path']}")
    print(f"Split: train={train_size}, test={test_size}")
    print(f"Device: {device}")
    print(f"Binary count estimate: {model.get_binary_count()}")

    print_table("Regression Metrics", reg_rows, ["Target", "MAE", "MAPE(%)", "NRMSE(%)"])
    print_table(
        "Classification Metrics",
        cls_rows,
        ["Target", "MAE", "FalseSafe(%)", "FalseViolate(%)", "SignAcc(%)", "BalancedAcc(%)"],
    )


if __name__ == "__main__":
    main()
