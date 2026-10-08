import torch
import torch.nn.functional as F

from .model import denorm_outputs, unpack_forward


def masked_mean(x, mask):
    mask = mask.float()
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


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

    node_mse = torch.mean(v_weight * (Vn_use - YVn_use) ** 2)
    worst_i_mse = torch.mean(i_weight * (YI_worst_n - YIn) ** 2)

    base = cfg["voltage_loss_weight"] * node_mse + cfg["worst_i_loss_weight"] * worst_i_mse

    # 旧的单侧电流假安全惩罚，保留用于兼容和日志；
    # 在当前配置中 worst_i_false_safe_lambda=0.0，不参与主损失。
    i_pen = masked_mean(
        torch.relu(cfg["worst_i_false_safe_margin"] - i_pred) ** 2,
        i_unsafe,
    )

    # 电流符号分类增强损失。
    # YI_worst > 0 表示电流越限，希望预测值 > +margin；
    # YI_worst <= 0 表示电流安全，希望预测值 < -margin。
    i_safe = ~i_unsafe
    i_margin = cfg["worst_i_sign_margin"]

    i_sign_fs_loss = masked_mean(
        torch.relu(i_margin - i_pred) ** 2,
        i_unsafe,
    )

    i_sign_fv_loss = masked_mean(
        torch.relu(i_pred + i_margin) ** 2,
        i_safe,
    )

    i_sign_loss = i_sign_fs_loss + i_sign_fv_loss

    low_mask = v_true < cfg["v_lower"]
    high_mask = v_true > cfg["v_upper"]

    low_pen = masked_mean(
        torch.relu(v_pred - (cfg["v_lower"] - cfg["v_false_safe_guard"])) ** 2,
        low_mask,
    )

    high_pen = masked_mean(
        torch.relu((cfg["v_upper"] + cfg["v_false_safe_guard"]) - v_pred) ** 2,
        high_mask,
    )

    v_pen = low_pen + high_pen

    total = base + ramp * (
        cfg["worst_i_false_safe_lambda"] * i_pen
        + cfg["worst_i_sign_loss_weight"] * i_sign_loss
        + cfg["v_false_safe_lambda"] * v_pen
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
    }

