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


def balanced_focal_bce_loss(
    true,
    pred,
    temperature,
    gamma,
    boundary_tau,
    boundary_extra,
    false_safe_weight=1.0,
    false_violate_weight=1.0,
):
    label = (true > 0.0).float()
    logits = pred / float(temperature)

    pos_count = label.sum().clamp_min(1.0)
    neg_count = (1.0 - label).sum().clamp_min(1.0)
    total = pos_count + neg_count
    class_weight = torch.where(
        label > 0.5,
        total / (2.0 * pos_count),
        total / (2.0 * neg_count),
    )
    side_weight = torch.where(
        label > 0.5,
        torch.full_like(label, float(false_safe_weight)),
        torch.full_like(label, float(false_violate_weight)),
    )
    boundary_weight = 1.0 + float(boundary_extra) * torch.exp(
        -torch.abs(true) / float(boundary_tau)
    )

    bce = F.binary_cross_entropy_with_logits(
        logits,
        label,
        reduction="none",
    )
    prob = torch.sigmoid(logits)
    pt = torch.where(label > 0.5, prob, 1.0 - prob)
    focal = (1.0 - pt).clamp_min(0.0).pow(float(gamma))

    weight = class_weight * side_weight * boundary_weight
    return (bce * focal * weight).sum() / weight.sum().clamp_min(1.0)


def compute_loss(model, batch, norm, cfg, ramp=1.0):
    (
        X_voltage,
        X_current,
        X_loss,
        YVDevn,
        YVWorstn,
        YIn,
        YPLossn,
        YV_dev,
        YV_worst,
        YI_worst,
        YP_loss,
    ) = batch

    Vdev_n, Vworst_n, Iworst_n, Ploss_n, *_ = unpack_forward(
        model,
        X_voltage,
        X_current,
        X_loss,
    )
    _, Vworst_pred, Iworst_pred, Ploss_pred = denorm_outputs(
        Vdev_n,
        Vworst_n,
        Iworst_n,
        Ploss_n,
        norm,
    )

    v_unsafe = YV_worst > 0.0
    i_unsafe = YI_worst > 0.0

    v_boundary = torch.exp(
        -torch.abs(YV_worst) / cfg["vworst_boundary_tau"]
    )
    i_boundary = torch.exp(
        -torch.abs(YI_worst) / cfg["iworst_boundary_tau"]
    )

    vdev_weight = torch.ones_like(YV_dev)
    ploss_weight = torch.ones_like(YP_loss)
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
    ploss_reg = weighted_regression_loss(
        Ploss_n,
        YPLossn,
        ploss_weight,
        cfg,
    )
    ploss_relative_error = (
        (Ploss_pred - YP_loss)
        / YP_loss.detach().clamp_min(cfg["ploss_relative_floor"])
    )
    ploss_relative = F.smooth_l1_loss(
        ploss_relative_error,
        torch.zeros_like(ploss_relative_error),
        beta=cfg["ploss_relative_beta"],
    )
    relative_weight = float(cfg["ploss_relative_weight"])
    ploss_objective = (
        (1.0 - relative_weight) * ploss_reg
        + relative_weight * ploss_relative
    )

    base = (
        cfg["vdev_loss_weight"] * vdev_reg
        + cfg["vworst_loss_weight"] * vworst_reg
        + cfg["iworst_loss_weight"] * iworst_reg
        + cfg["ploss_loss_weight"] * ploss_objective
    )

    v_sign, v_sign_fs, v_sign_fv = sign_margin_loss(
        YV_worst,
        Vworst_pred,
        cfg["vworst_sign_margin"],
        cfg["vworst_sign_scale"],
    )
    i_sign, i_sign_fs, i_sign_fv = sign_margin_loss(
        YI_worst,
        Iworst_pred,
        cfg["iworst_sign_margin"],
        cfg["iworst_sign_scale"],
    )
    v_cls = balanced_focal_bce_loss(
        YV_worst,
        Vworst_pred,
        cfg["vworst_cls_temperature"],
        cfg["cls_focal_gamma"],
        cfg["cls_boundary_tau"],
        cfg["cls_boundary_extra"],
        false_safe_weight=cfg["vworst_cls_false_safe_weight"],
        false_violate_weight=cfg["vworst_cls_false_violate_weight"],
    )
    if cfg["iworst_cls_loss_weight"] > 0.0:
        i_cls = balanced_focal_bce_loss(
            YI_worst,
            Iworst_pred,
            cfg["iworst_cls_temperature"],
            cfg["cls_focal_gamma"],
            cfg["cls_boundary_tau"],
            cfg["cls_boundary_extra"],
            false_safe_weight=cfg["iworst_cls_false_safe_weight"],
            false_violate_weight=cfg["iworst_cls_false_violate_weight"],
        )
    else:
        i_cls = Iworst_pred.new_zeros(())

    total = base + ramp * (
        cfg["vworst_sign_loss_weight"] * v_sign
        + cfg["iworst_sign_loss_weight"] * i_sign
        + cfg["vworst_cls_loss_weight"] * v_cls
        + cfg["iworst_cls_loss_weight"] * i_cls
    )

    return total, {
        "base": base.detach(),
        "vdev_reg": vdev_reg.detach(),
        "vworst_reg": vworst_reg.detach(),
        "iworst_reg": iworst_reg.detach(),
        "ploss_reg": ploss_reg.detach(),
        "ploss_relative": ploss_relative.detach(),
        "ploss_objective": ploss_objective.detach(),
        "v_sign": v_sign.detach(),
        "v_sign_fs": v_sign_fs.detach(),
        "v_sign_fv": v_sign_fv.detach(),
        "i_sign": i_sign.detach(),
        "i_sign_fs": i_sign_fs.detach(),
        "i_sign_fv": i_sign_fv.detach(),
        "v_cls": v_cls.detach(),
        "i_cls": i_cls.detach(),
    }

