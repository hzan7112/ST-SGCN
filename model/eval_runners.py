import argparse
import importlib
import os

import torch
from torch.utils.data import DataLoader, TensorDataset


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
    std = true.std(unbiased=False)
    nrmse = rmse / std.clamp_min(eps) * 100.0
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


def voltage_classification_row(name, pred, true, v_lower, v_upper):
    pred = pred.detach().float().reshape(-1).cpu()
    true = true.detach().float().reshape(-1).cpu()
    true_unsafe = (true < v_lower) | (true > v_upper)
    pred_unsafe = (pred < v_lower) | (pred > v_upper)

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


def split_indices(n, cfg):
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"]))
    n_train = int(n * cfg["train_ratio"])
    return idx[:n_train], idx[n_train:]


def base_dataset(data_mod, cfg):
    data = data_mod.load_pt(cfg["data_path"])
    edge_list = data_mod.get_edge_list(data)
    x_raw = data_mod.to_tensor(data["X"]).float().numpy()
    yv, yi_branch = data_mod.get_labels(data)
    s_down, s_path, _ = data_mod.build_topology_matrices(edge_list, 33)
    x_aug = torch.tensor(
        data_mod.augment_path_power_features(x_raw, s_down, s_path),
        dtype=torch.float32,
    )
    train_idx, test_idx = split_indices(len(x_aug), cfg)
    return data, edge_list, x_aug, yv, yi_branch, train_idx, test_idx


def prepare_voltage_worsti(data_mod, cfg):
    data, edge_list, x_aug, yv, yi_branch, train_idx, test_idx = base_dataset(data_mod, cfg)
    yi_target = yi_branch if cfg.get("predicts_branch_current", False) else yi_branch.max(dim=1, keepdim=True).values
    has_ploss = (
        hasattr(data_mod, "compute_total_network_loss")
        and hasattr(data_mod, "get_branch_resistance")
        and "ploss_loss_weight" in cfg
    )
    if has_ploss:
        branch_r_ohm = data_mod.get_branch_resistance(data, edge_list)
        line_max_i_ka = data_mod.get_line_max_i_ka(
            data,
            cfg.get("default_line_max_i_ka", 0.20),
        )
        yp_loss = data_mod.compute_total_network_loss(
            yi_branch,
            branch_r_ohm,
            line_max_i_ka,
        )
        xn, yvn, yin, ypn, norm = data_mod.normalize_data(
            x_aug,
            yv,
            yi_target,
            yp_loss,
            train_idx,
        )
        dataset = TensorDataset(
            xn[test_idx],
            yvn[test_idx],
            yin[test_idx],
            ypn[test_idx],
            yv[test_idx],
            yi_target[test_idx],
            yp_loss[test_idx],
        )
        return edge_list, norm, dataset, len(train_idx), len(test_idx)

    xn, yvn, yin, norm = data_mod.normalize_data(x_aug, yv, yi_target, train_idx)
    dataset = TensorDataset(
        xn[test_idx],
        yvn[test_idx],
        yin[test_idx],
        yv[test_idx],
        yi_target[test_idx],
    )
    return edge_list, norm, dataset, len(train_idx), len(test_idx)


def prepare_three_scalar(data_mod, cfg):
    _, edge_list, x_aug, yv, yi_branch, train_idx, test_idx = base_dataset(data_mod, cfg)
    yv_dev = data_mod.compute_cumulative_voltage_deviation(yv)
    yv_worst = data_mod.compute_worst_voltage_margin(yv, cfg["v_lower"], cfg["v_upper"])
    yi_worst = yi_branch.max(dim=1, keepdim=True).values
    xn, yvdevn, yvworstn, yin, norm = data_mod.normalize_data(
        x_aug,
        yv_dev,
        yv_worst,
        yi_worst,
        train_idx,
    )
    dataset = TensorDataset(
        xn[test_idx],
        yvdevn[test_idx],
        yvworstn[test_idx],
        yin[test_idx],
        yv_dev[test_idx],
        yv_worst[test_idx],
        yi_worst[test_idx],
    )
    return edge_list, norm, dataset, len(train_idx), len(test_idx)


def prepare_four_scalar(data_mod, cfg):
    data, edge_list, x_aug, yv, yi_branch, train_idx, test_idx = base_dataset(data_mod, cfg)
    yv_dev = data_mod.compute_cumulative_voltage_deviation(yv)
    yv_worst = data_mod.compute_worst_voltage_margin(yv, cfg["v_lower"], cfg["v_upper"])
    yi_worst = yi_branch.max(dim=1, keepdim=True).values
    branch_r_ohm = data_mod.get_branch_resistance(data, edge_list)
    line_max_i_ka = data_mod.get_line_max_i_ka(data, cfg)
    yp_loss = data_mod.compute_total_network_loss(yi_branch, branch_r_ohm, line_max_i_ka)
    if cfg.get("encoder_mode") == "triple_linear_sgc":
        s_down, s_path, parent = data_mod.build_topology_matrices(edge_list, 33)
        x_raw = data_mod.to_tensor(data["X"]).float().numpy()
        x_voltage_raw, x_current_raw, x_loss_raw = data_mod.augment_triple_encoder_features(
            x_raw,
            s_down,
            s_path,
            parent,
        )
        x_voltage = torch.tensor(x_voltage_raw, dtype=torch.float32)
        x_current = torch.tensor(x_current_raw, dtype=torch.float32)
        x_loss = torch.tensor(x_loss_raw, dtype=torch.float32)
        xv_n, xi_n, xl_n, yvdevn, yvworstn, yin, yplossn, norm = data_mod.normalize_data(
            x_voltage,
            x_current,
            x_loss,
            yv_dev,
            yv_worst,
            yi_worst,
            yp_loss,
            train_idx,
        )
        dataset = TensorDataset(
            xv_n[test_idx],
            xi_n[test_idx],
            xl_n[test_idx],
            yvdevn[test_idx],
            yvworstn[test_idx],
            yin[test_idx],
            yplossn[test_idx],
            yv_dev[test_idx],
            yv_worst[test_idx],
            yi_worst[test_idx],
            yp_loss[test_idx],
        )
        return edge_list, norm, dataset, len(train_idx), len(test_idx)

    if cfg.get("encoder_mode") == "dual_linear_sgc":
        s_down, s_path, parent = data_mod.build_topology_matrices(edge_list, 33)
        x_raw = data_mod.to_tensor(data["X"]).float().numpy()
        x_voltage_raw, x_flow_raw = data_mod.augment_dual_encoder_features(
            x_raw,
            s_down,
            s_path,
            parent,
        )
        x_voltage = torch.tensor(x_voltage_raw, dtype=torch.float32)
        x_flow = torch.tensor(x_flow_raw, dtype=torch.float32)
        xv_n, xf_n, yvdevn, yvworstn, yin, yplossn, norm = data_mod.normalize_data(
            x_voltage,
            x_flow,
            yv_dev,
            yv_worst,
            yi_worst,
            yp_loss,
            train_idx,
        )
        dataset = TensorDataset(
            xv_n[test_idx],
            xf_n[test_idx],
            yvdevn[test_idx],
            yvworstn[test_idx],
            yin[test_idx],
            yplossn[test_idx],
            yv_dev[test_idx],
            yv_worst[test_idx],
            yi_worst[test_idx],
            yp_loss[test_idx],
        )
        return edge_list, norm, dataset, len(train_idx), len(test_idx)

    xn, yvdevn, yvworstn, yin, yplossn, norm = data_mod.normalize_data(
        x_aug,
        yv_dev,
        yv_worst,
        yi_worst,
        yp_loss,
        train_idx,
    )
    dataset = TensorDataset(
        xn[test_idx],
        yvdevn[test_idx],
        yvworstn[test_idx],
        yin[test_idx],
        yplossn[test_idx],
        yv_dev[test_idx],
        yv_worst[test_idx],
        yi_worst[test_idx],
        yp_loss[test_idx],
    )
    return edge_list, norm, dataset, len(train_idx), len(test_idx)


def prepare_hsafe_three_scalar(data_mod, cfg):
    data, edge_list, x_aug, yv, yi_branch, train_idx, test_idx = base_dataset(data_mod, cfg)
    yv_dev = data_mod.compute_cumulative_voltage_deviation(yv)
    yv_worst = data_mod.compute_worst_voltage_margin(yv, cfg["v_lower"], cfg["v_upper"])
    yi_worst = yi_branch.max(dim=1, keepdim=True).values
    yh_safe = torch.maximum(
        yv_worst / cfg["v_safe_scale"],
        yi_worst / cfg["i_safe_scale"],
    )
    branch_r_ohm = data_mod.get_branch_resistance(data, edge_list)
    line_max_i_ka = data_mod.get_line_max_i_ka(data, cfg)
    yp_loss = data_mod.compute_total_network_loss(yi_branch, branch_r_ohm, line_max_i_ka)
    xn, yvdevn, yhsafen, yplossn, norm = data_mod.normalize_data(
        x_aug,
        yv_dev,
        yh_safe,
        yp_loss,
        train_idx,
    )
    dataset = TensorDataset(
        xn[test_idx],
        yvdevn[test_idx],
        yhsafen[test_idx],
        yplossn[test_idx],
        yv_dev[test_idx],
        yh_safe[test_idx],
        yp_loss[test_idx],
    )
    return edge_list, norm, dataset, len(train_idx), len(test_idx)


def experiment_mode(model_mod):
    if (
        hasattr(model_mod, "STSGCNThreeDirectScalars")
        and "hsafe_head_dim" in model_mod.STSGCNThreeDirectScalars.__init__.__code__.co_varnames
    ):
        return "hsafe_three_scalar"
    if hasattr(model_mod, "DirectedHeteroTaskReadoutFourScalars"):
        return "hetero_task_readout_four_scalar"
    if hasattr(model_mod, "DirectedHeteroSGCNFourScalars"):
        return "hetero_four_scalar"
    if hasattr(model_mod, "STSGCNPairedFourScalars"):
        return "paired_four_scalar"
    if hasattr(model_mod, "STSGCNFourDirectScalars"):
        return "four_scalar"
    if hasattr(model_mod, "STSGCNThreeDirectScalars"):
        return "three_scalar"
    if hasattr(model_mod, "STSGCNVdevVworstIworst"):
        return "wrapped_three_scalar"
    if hasattr(model_mod, "STSGCNWorstI"):
        return "voltage_worsti"
    raise RuntimeError("Unsupported experiment model module.")


def build_model(model_mod, cfg, edge_list, checkpoint, mode, device):
    if mode == "hetero_task_readout_four_scalar":
        model = model_mod.DirectedHeteroTaskReadoutFourScalars(
            in_features=6,
            bus_hidden_dim=cfg["bus_hidden_dim"],
            branch_hidden_dim=cfg["branch_hidden_dim"],
            K=cfg["K"],
            edge_list=edge_list,
            num_nodes=33,
            vdev_local_relu_dim=cfg["vdev_local_relu_dim"],
            ploss_local_relu_dim=cfg["ploss_local_relu_dim"],
            include_order0=cfg["include_order0"],
            use_sgc_relu=cfg["use_sgc_relu"],
            use_linear_skip=cfg["use_linear_skip"],
        )
    elif mode == "hetero_four_scalar":
        model = model_mod.DirectedHeteroSGCNFourScalars(
            in_features=6,
            bus_hidden_dim=cfg["bus_hidden_dim"],
            branch_hidden_dim=cfg["branch_hidden_dim"],
            K=cfg["K"],
            edge_list=edge_list,
            num_nodes=33,
            voltage_shared_head_dim=cfg["voltage_shared_head_dim"],
            vdev_private_head_dim=cfg["vdev_private_head_dim"],
            vworst_private_head_dim=cfg["vworst_private_head_dim"],
            current_loss_pair_head_dim=cfg["current_loss_pair_head_dim"],
            include_order0=cfg["include_order0"],
            use_sgc_relu=cfg["use_sgc_relu"],
            use_linear_skip=cfg["use_linear_skip"],
        )
    elif mode == "paired_four_scalar":
        model = model_mod.STSGCNPairedFourScalars(
            in_features=6,
            hidden_dim=cfg["hidden_dim"],
            K=cfg["K"],
            edge_list=edge_list,
            num_nodes=33,
            voltage_shared_head_dim=cfg["voltage_shared_head_dim"],
            vdev_private_head_dim=cfg["vdev_private_head_dim"],
            vworst_private_head_dim=cfg["vworst_private_head_dim"],
            current_loss_pair_head_dim=cfg["current_loss_pair_head_dim"],
            include_order0=cfg["include_order0"],
            use_sgc_relu=cfg["use_sgc_relu"],
            use_linear_skip=cfg["use_linear_skip"],
        )
    elif mode == "four_scalar":
        kwargs = {
            "hidden_dim": cfg["hidden_dim"],
            "K": cfg["K"],
            "edge_list": edge_list,
            "num_nodes": 33,
            "vdev_head_dim": cfg["vdev_head_dim"],
            "vworst_head_dim": cfg["vworst_head_dim"],
            "iworst_head_dim": cfg["iworst_head_dim"],
            "ploss_head_dim": cfg["ploss_head_dim"],
            "include_order0": cfg["include_order0"],
            "use_sgc_relu": cfg["use_sgc_relu"],
            "use_linear_skip": cfg["use_linear_skip"],
        }
        init_vars = model_mod.STSGCNFourDirectScalars.__init__.__code__.co_varnames
        if "voltage_in_features" in init_vars:
            kwargs["voltage_in_features"] = cfg.get("voltage_in_features", 6)
            if "current_in_features" in init_vars:
                kwargs["current_in_features"] = cfg.get("current_in_features", 6)
                kwargs["loss_in_features"] = cfg.get("loss_in_features", 6)
            else:
                kwargs["flow_in_features"] = cfg.get("flow_in_features", 6)
        else:
            kwargs["in_features"] = cfg.get("in_features", 6)
        if "ploss_head_dim2" in init_vars and "ploss_head_dim2" in cfg:
            kwargs["ploss_head_dim2"] = cfg["ploss_head_dim2"]
        model = model_mod.STSGCNFourDirectScalars(**kwargs)
    elif mode == "hsafe_three_scalar":
        model = model_mod.STSGCNThreeDirectScalars(
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
        )
    elif mode == "three_scalar":
        model = model_mod.STSGCNThreeDirectScalars(
            in_features=6,
            hidden_dim=cfg["hidden_dim"],
            K=cfg["K"],
            edge_list=edge_list,
            num_nodes=33,
            vdev_head_dim=cfg["vdev_head_dim"],
            vworst_head_dim=cfg["vworst_head_dim"],
            iworst_head_dim=cfg["iworst_head_dim"],
            include_order0=cfg["include_order0"],
            use_sgc_relu=cfg["use_sgc_relu"],
            use_linear_skip=cfg["use_linear_skip"],
        )
    elif mode == "wrapped_three_scalar":
        base_model = model_mod.STSGCNVdevVworstIworst(
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
        model = model_mod.VoltageLinearResidualWrapper(
            base_model=base_model,
            W_vlin=checkpoint["voltage_linear_prior_W"],
            b_vlin=checkpoint["voltage_linear_prior_b"],
        )
    else:
        base_model = model_mod.STSGCNWorstI(
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
        model = model_mod.VoltageLinearResidualWrapper(
            base_model=base_model,
            W_vlin=checkpoint["voltage_linear_prior_W"],
            b_vlin=checkpoint["voltage_linear_prior_b"],
        )

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("wrapper_state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint does not contain a model state dict.")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_voltage_worsti(
    model,
    model_mod,
    loader,
    norm,
    cfg,
    device,
    include_voltage_linear_prior=False,
):
    pred_v, true_v, pred_i, true_i = [], [], [], []
    pred_ploss, true_ploss = [], []
    pred_vlin = []
    for batch in loader:
        batch = [item.to(device) for item in batch]
        has_ploss = len(batch) >= 7 and "YP_loss_mean" in norm
        if has_ploss:
            x, _, _, _, yv, yi_worst, yp_loss = batch
            vn, inorm, plossn, *_ = model_mod.unpack_forward(model, x)
            v, iworst, ploss = model_mod.denorm_outputs(vn, inorm, plossn, norm)
            pred_ploss.append(ploss.cpu())
            true_ploss.append(yp_loss.cpu())
        else:
            x, _, _, yv, yi_worst = batch
            vn, inorm, *_ = model_mod.unpack_forward(model, x)
            v, iworst = model_mod.denorm_outputs(vn, inorm, norm)
        pred_v.append(v.cpu())
        true_v.append(yv.cpu())
        pred_i.append(iworst.cpu())
        true_i.append(yi_worst.cpu())

        if include_voltage_linear_prior:
            if not hasattr(model, "W_vlin") or not hasattr(model, "b_vlin"):
                raise RuntimeError("This model does not expose W_vlin/b_vlin for voltage prior evaluation.")

            vlin_n = x.reshape(x.size(0), -1) @ model.W_vlin.T + model.b_vlin
            zeros_i = torch.zeros_like(inorm)
            if has_ploss:
                zeros_p = torch.zeros((x.size(0), 1), dtype=vlin_n.dtype, device=device)
                vlin, _, _ = model_mod.denorm_outputs(vlin_n, zeros_i, zeros_p, norm)
            else:
                vlin, _ = model_mod.denorm_outputs(vlin_n, zeros_i, norm)
            pred_vlin.append(vlin.cpu())

    pred_v = torch.cat(pred_v)
    true_v = torch.cat(true_v)
    pred_i = torch.cat(pred_i)
    true_i = torch.cat(true_i)

    current_label = "BranchI" if pred_i.shape[1] > 1 else "WorstI"
    reg_rows = [
        regression_row("Voltage", pred_v[:, 1:], true_v[:, 1:]),
        regression_row(current_label, pred_i, true_i),
    ]
    if pred_ploss:
        reg_rows.append(
            regression_row(
                "NetworkLoss",
                torch.cat(pred_ploss),
                torch.cat(true_ploss),
            )
        )
    if include_voltage_linear_prior:
        pred_vlin = torch.cat(pred_vlin)
        reg_rows.append(
            regression_row("VoltageLinearPrior", pred_vlin[:, 1:], true_v[:, 1:])
        )
    cls_rows = [
        voltage_classification_row(
            "Voltage",
            pred_v[:, 1:],
            true_v[:, 1:],
            cfg["v_lower"],
            cfg["v_upper"],
        ),
        classification_row(current_label, pred_i, true_i),
    ]
    return reg_rows, cls_rows


@torch.no_grad()
def collect_three_scalar(model, model_mod, loader, norm, device):
    pred_vdev, true_vdev = [], []
    pred_vworst, true_vworst = [], []
    pred_i, true_i = [], []

    for batch in loader:
        batch = [item.to(device) for item in batch]
        x = batch[0]
        vdevn, vworstn, inorm, *_ = model_mod.unpack_forward(model, x)
        vdev, vworst, iworst = model_mod.denorm_outputs(vdevn, vworstn, inorm, norm)
        pred_vdev.append(vdev.cpu())
        pred_vworst.append(vworst.cpu())
        pred_i.append(iworst.cpu())
        true_vdev.append(batch[4].cpu())
        true_vworst.append(batch[5].cpu())
        true_i.append(batch[6].cpu())

    pred_vdev = torch.cat(pred_vdev)
    true_vdev = torch.cat(true_vdev)
    pred_vworst = torch.cat(pred_vworst)
    true_vworst = torch.cat(true_vworst)
    pred_i = torch.cat(pred_i)
    true_i = torch.cat(true_i)

    reg_rows = [
        regression_row("VoltageDeviation", pred_vdev, true_vdev),
    ]
    cls_rows = [
        classification_row("Vworst", pred_vworst, true_vworst),
        classification_row("WorstI", pred_i, true_i),
    ]
    return reg_rows, cls_rows


@torch.no_grad()
def collect_four_scalar(model, model_mod, loader, norm, device):
    pred_vdev, true_vdev = [], []
    pred_vworst, true_vworst = [], []
    pred_i, true_i = [], []
    pred_ploss, true_ploss = [], []

    for batch in loader:
        batch = [item.to(device) for item in batch]
        if len(batch) == 11:
            x_voltage, x_current, x_loss = batch[0], batch[1], batch[2]
            vdevn, vworstn, inorm, plossn, *_ = model_mod.unpack_forward(
                model,
                x_voltage,
                x_current,
                x_loss,
            )
            target_offset = 2
        elif len(batch) == 10:
            x_voltage, x_flow = batch[0], batch[1]
            vdevn, vworstn, inorm, plossn, *_ = model_mod.unpack_forward(
                model,
                x_voltage,
                x_flow,
            )
            target_offset = 1
        else:
            x = batch[0]
            vdevn, vworstn, inorm, plossn, *_ = model_mod.unpack_forward(model, x)
            target_offset = 0
        vdev, vworst, iworst, ploss = model_mod.denorm_outputs(
            vdevn,
            vworstn,
            inorm,
            plossn,
            norm,
        )
        pred_vdev.append(vdev.cpu())
        pred_vworst.append(vworst.cpu())
        pred_i.append(iworst.cpu())
        pred_ploss.append(ploss.cpu())
        true_vdev.append(batch[5 + target_offset].cpu())
        true_vworst.append(batch[6 + target_offset].cpu())
        true_i.append(batch[7 + target_offset].cpu())
        true_ploss.append(batch[8 + target_offset].cpu())

    pred_vdev = torch.cat(pred_vdev)
    true_vdev = torch.cat(true_vdev)
    pred_vworst = torch.cat(pred_vworst)
    true_vworst = torch.cat(true_vworst)
    pred_i = torch.cat(pred_i)
    true_i = torch.cat(true_i)
    pred_ploss = torch.cat(pred_ploss)
    true_ploss = torch.cat(true_ploss)

    reg_rows = [
        regression_row("VoltageDeviation", pred_vdev, true_vdev),
        regression_row("NetworkLoss", pred_ploss, true_ploss),
    ]
    cls_rows = [
        classification_row("Vworst", pred_vworst, true_vworst),
        classification_row("WorstI", pred_i, true_i),
    ]
    return reg_rows, cls_rows


@torch.no_grad()
def collect_hsafe_three_scalar(model, model_mod, loader, norm, device):
    pred_vdev, true_vdev = [], []
    pred_hsafe, true_hsafe = [], []
    pred_ploss, true_ploss = [], []

    for batch in loader:
        batch = [item.to(device) for item in batch]
        x = batch[0]
        vdevn, hsafen, plossn, *_ = model_mod.unpack_forward(model, x)
        vdev, hsafe, ploss = model_mod.denorm_outputs(
            vdevn,
            hsafen,
            plossn,
            norm,
        )
        pred_vdev.append(vdev.cpu())
        pred_hsafe.append(hsafe.cpu())
        pred_ploss.append(ploss.cpu())
        true_vdev.append(batch[4].cpu())
        true_hsafe.append(batch[5].cpu())
        true_ploss.append(batch[6].cpu())

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


def run_eval(package_name, include_voltage_linear_prior=False):
    parser = argparse.ArgumentParser(description="Evaluate an ST-SGCN experiment checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to config best_model_name.")
    parser.add_argument("--device", default=None, help="Device, for example cpu or cuda.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size.")
    args = parser.parse_args()

    data_mod = importlib.import_module(f"{package_name}.data")
    model_mod = importlib.import_module(f"{package_name}.model")
    train_mod = importlib.import_module(f"{package_name}.train")

    cfg = train_mod.get_config()
    configured_data_path = cfg["data_path"]
    checkpoint_path = args.checkpoint or os.path.join(cfg["save_dir"], cfg["best_model_name"])
    checkpoint = load_checkpoint(checkpoint_path)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        cfg.update(checkpoint["config"])
        cfg["data_path"] = configured_data_path
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mode = experiment_mode(model_mod)

    if mode == "voltage_worsti":
        edge_list, norm, dataset, train_size, test_size = prepare_voltage_worsti(data_mod, cfg)
    elif mode == "hsafe_three_scalar":
        edge_list, norm, dataset, train_size, test_size = prepare_hsafe_three_scalar(data_mod, cfg)
    elif mode in {"three_scalar", "wrapped_three_scalar"}:
        edge_list, norm, dataset, train_size, test_size = prepare_three_scalar(data_mod, cfg)
    else:
        edge_list, norm, dataset, train_size, test_size = prepare_four_scalar(data_mod, cfg)

    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, drop_last=False)
    model = build_model(model_mod, cfg, edge_list, checkpoint, mode, device)

    if mode == "voltage_worsti":
        reg_rows, cls_rows = collect_voltage_worsti(
            model,
            model_mod,
            loader,
            norm,
            cfg,
            device,
            include_voltage_linear_prior=include_voltage_linear_prior,
        )
    elif mode == "hsafe_three_scalar":
        reg_rows, cls_rows = collect_hsafe_three_scalar(model, model_mod, loader, norm, device)
    elif mode in {"three_scalar", "wrapped_three_scalar"}:
        reg_rows, cls_rows = collect_three_scalar(model, model_mod, loader, norm, device)
    else:
        reg_rows, cls_rows = collect_four_scalar(model, model_mod, loader, norm, device)

    print(f"\nExperiment: {package_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {cfg['data_path']}")
    print(f"Split: train={train_size}, test={test_size}")
    print(f"Device: {device}")

    print_table("Regression Metrics", reg_rows, ["Target", "MAE", "MAPE(%)", "NRMSE(%)"])
    print_table(
        "Classification Metrics",
        cls_rows,
        ["Target", "MAE", "FalseSafe(%)", "FalseViolate(%)", "SignAcc(%)", "BalancedAcc(%)"],
    )
