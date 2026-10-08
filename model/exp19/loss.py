import torch
import torch.nn.functional as F

from .model import denorm_outputs, unpack_forward


def masked_mean(x, mask):
    mask = mask.float()
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def weighted_regression_loss(pred, true, weight, cfg):
    huber = F.smooth_l1_loss(
        pred,
        true,
        beta=cfg["huber_beta"],
        reduction="none",
    )
    mse = (pred - true).square()
    loss = huber + cfg["mse_mix"] * mse
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def sign_margin_loss(true, pred, margin, scale):
    unsafe = true > 0.0
    safe = ~unsafe

    false_safe_loss = masked_mean(
        (F.relu(float(margin) - pred) / float(scale)).square(),
        unsafe,
    )
    false_violate_loss = masked_mean(
        (F.relu(pred + float(margin)) / float(scale)).square(),
        safe,
    )

    return (
        false_safe_loss + false_violate_loss,
        false_safe_loss,
        false_violate_loss,
    )


def compute_loss(model, batch, norm, cfg, ramp=1.0):
    X, YVDevn, YVWorstn, YIn, YV_dev, YV_worst, YI_worst = batch

    Vdev_n, Vworst_n, Iworst_n, *_ = unpack_forward(model, X)
    Vdev_pred, Vworst_pred, Iworst_pred = denorm_outputs(
        Vdev_n,
        Vworst_n,
        Iworst_n,
        norm,
    )

    v_unsafe = YV_worst > 0.0
    i_unsafe = YI_worst > 0.0

    v_boundary = torch.exp(
        -torch.abs(YV_worst) / cfg["boundary_tau"]
    )
    i_boundary = torch.exp(
        -torch.abs(YI_worst) / cfg["boundary_tau"]
    )

    vdev_weight = torch.ones_like(YV_dev)
    vworst_weight = (
        1.0
        + cfg["vworst_unsafe_extra"] * v_unsafe.float()
        + cfg["boundary_extra"] * v_boundary
    )
    iworst_weight = (
        1.0
        + cfg["iworst_unsafe_extra"] * i_unsafe.float()
        + cfg["boundary_extra"] * i_boundary
    )

    vdev_reg = weighted_regression_loss(
        Vdev_n,
        YVDevn,
        vdev_weight,
        cfg,
    )
    vworst_reg = weighted_regression_loss(
        Vworst_n,
        YVWorstn,
        vworst_weight,
        cfg,
    )
    iworst_reg = weighted_regression_loss(
        Iworst_n,
        YIn,
        iworst_weight,
        cfg,
    )

    base = (
        cfg["vdev_loss_weight"] * vdev_reg
        + cfg["vworst_loss_weight"] * vworst_reg
        + cfg["iworst_loss_weight"] * iworst_reg
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
        "v_sign": v_sign.detach(),
        "v_sign_fs": v_sign_fs.detach(),
        "v_sign_fv": v_sign_fv.detach(),
        "i_sign": i_sign.detach(),
        "i_sign_fs": i_sign_fs.detach(),
        "i_sign_fv": i_sign_fv.detach(),
    }

