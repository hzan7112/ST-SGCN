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
    (
        X,
        YVDevn,
        YHsafe_n,
        YPLossn,
        YV_dev,
        YH_safe,
        YP_loss,
    ) = batch

    Vdev_n, Hsafe_n, Ploss_n, *_ = unpack_forward(model, X)
    _, Hsafe_pred, _ = denorm_outputs(
        Vdev_n,
        Hsafe_n,
        Ploss_n,
        norm,
    )

    vdev_weight = torch.ones_like(YV_dev)
    hsafe_weight = torch.ones_like(YH_safe)
    ploss_weight = torch.ones_like(YP_loss)

    vdev_reg = weighted_regression_loss(
        Vdev_n,
        YVDevn,
        vdev_weight,
        cfg,
    )
    hsafe_reg = weighted_regression_loss(
        Hsafe_n,
        YHsafe_n,
        hsafe_weight,
        cfg,
    )
    ploss_reg = weighted_regression_loss(
        Ploss_n,
        YPLossn,
        ploss_weight,
        cfg,
    )

    base = (
        cfg["vdev_loss_weight"] * vdev_reg
        + cfg["hsafe_loss_weight"] * hsafe_reg
        + cfg["ploss_loss_weight"] * ploss_reg
    )

    h_sign, h_sign_fs, h_sign_fv = sign_margin_loss(
        YH_safe,
        Hsafe_pred,
        0.0,
        cfg["hsafe_sign_scale"],
    )

    total = base + ramp * (
        cfg["hsafe_sign_loss_weight"] * h_sign
    )

    return total, {
        "base": base.detach(),
        "vdev_reg": vdev_reg.detach(),
        "hsafe_reg": hsafe_reg.detach(),
        "ploss_reg": ploss_reg.detach(),
        "h_sign": h_sign.detach(),
        "h_sign_fs": h_sign_fs.detach(),
        "h_sign_fv": h_sign_fv.detach(),
    }
