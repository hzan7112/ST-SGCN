import argparse
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from train_mlp import (
    build_base_model,
    denorm_outputs,
    get_config,
    get_edge_list,
    get_labels,
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
    YI_worst = YI_branch.max(dim=1, keepdim=True).values

    idx = torch.randperm(len(X_raw), generator=torch.Generator().manual_seed(cfg["seed"]))
    n_train = int(len(X_raw) * cfg["train_ratio"])
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    Xn, YVn, YIn, norm = normalize_data(X_raw, YV, YI_worst, train_idx)
    dataset = TensorDataset(
        Xn[test_idx],
        YVn[test_idx],
        YIn[test_idx],
        YV[test_idx],
        YI_worst[test_idx],
    )
    return edge_list, norm, dataset, len(train_idx), len(test_idx)


@torch.no_grad()
def collect_metrics(model, loader, norm, cfg, device):
    model.eval()
    V_pred_all, I_pred_all, V_true_all, I_true_all = [], [], [], []

    for batch in loader:
        batch = [item.to(device) for item in batch]
        X, _, _, YV, YI_worst = batch
        Vn, YI_worst_n, *_ = unpack_forward(model, X)
        V, I_pred = denorm_outputs(Vn, YI_worst_n, norm)
        V_pred_all.append(V.cpu())
        I_pred_all.append(I_pred.cpu())
        V_true_all.append(YV.cpu())
        I_true_all.append(YI_worst.cpu())

    Vp = torch.cat(V_pred_all)
    Ip = torch.cat(I_pred_all)
    Vt = torch.cat(V_true_all)
    It = torch.cat(I_true_all)

    v_err = Vp[:, 1:] - Vt[:, 1:]
    i_err = Ip - It
    reg_rows = [
        {
            "Target": "V_nodes",
            "MAE": float(torch.mean(torch.abs(v_err))),
            "RMSE": float(torch.sqrt(torch.mean(v_err.square()))),
            "MaxErr": float(torch.max(torch.abs(v_err))),
        },
        {
            "Target": "YI_worst",
            "MAE": float(torch.mean(torch.abs(i_err))),
            "RMSE": float(torch.sqrt(torch.mean(i_err.square()))),
            "MaxErr": float(torch.max(torch.abs(i_err))),
        },
    ]

    def cls_row(name, true_unsafe, pred_unsafe):
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
            "FalseSafe(%)": fn / unsafe_total * 100.0,
            "FalseViolate(%)": fp / safe_total * 100.0,
            "SignAcc(%)": (tp + tn) / total * 100.0,
            "BalancedAcc(%)": 0.5 * (unsafe_recall + safe_recall),
        }

    v_true_unsafe = (Vt[:, 1:] < cfg["v_lower"]) | (Vt[:, 1:] > cfg["v_upper"])
    v_pred_unsafe = (Vp[:, 1:] < cfg["v_lower"]) | (Vp[:, 1:] > cfg["v_upper"])
    i_true_unsafe = It > 0.0
    i_pred_unsafe = Ip > 0.0
    cls_rows = [
        cls_row("V_nodes", v_true_unsafe, v_pred_unsafe),
        cls_row("YI_worst", i_true_unsafe, i_pred_unsafe),
    ]
    return reg_rows, cls_rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate the MLP Exp17-output checkpoint.")
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
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint does not contain model_state_dict or state_dict.")
    model.load_state_dict(state_dict)

    reg_rows, cls_rows = collect_metrics(model, loader, norm, cfg, device)

    print("\nExperiment: model.mlp_exp1")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {cfg['data_path']}")
    print(f"Split: train={train_size}, test={test_size}")
    print(f"Device: {device}")
    print("Input features: [P_net,Q_net]; no aggregation; no linear prior")
    print(f"Binary count estimate: {model.get_binary_count()}")

    print_table("Regression Metrics", reg_rows, ["Target", "MAE", "RMSE", "MaxErr"])
    print_table(
        "Safety Classification Metrics",
        cls_rows,
        ["Target", "FalseSafe(%)", "FalseViolate(%)", "SignAcc(%)", "BalancedAcc(%)"],
    )


if __name__ == "__main__":
    main()
