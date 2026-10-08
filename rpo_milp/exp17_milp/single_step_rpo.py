"""Single-step reactive power optimization using the Exp17 SGCN-MILP.

This script mirrors the older ``rpo_milp/src/RPO_MILP.py`` workflow, but uses
the Exp17 nodal-voltage and worst-current surrogate:

    V_nodes, Vdev_total, Vworst, WorstI

The decision variables are the reactive injections of PV inverters, ESS
inverters, and discrete/continuous reactive devices present in the selected
operating point. Exp17 has no learned loss head, so the objective uses only
the learned cumulative voltage deviation:

    Vdev_total

The surrogate safety constraints are:

    Vworst <= voltage_margin
    WorstI <= current_margin

where zero means the predicted operating point is exactly at the learned safety
boundary.

The voltage and current heads are learned. Ploss_total is intentionally not
used in Exp17 RPO because the model did not learn total network loss and this
workflow does not assume exact line impedance parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Keep numerical libraries quiet before importing numpy/torch/gurobi.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ENGINE_PATH = (
    REPO_ROOT
    / "checkpoints"
    / "st_sgcn_k4_h24_n2_g32_exp17_current_sign_voltage_safety_milp_engine.pt"
)

gp = None
GRB = None


def require_gurobi():
    global gp, GRB
    if gp is not None and GRB is not None:
        return gp, GRB
    try:
        import gurobipy as gp_mod
        from gurobipy import GRB as grb_mod
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gurobipy is required to solve the Exp17 MILP. "
            "Install Gurobi and run this script in the environment where "
            "gurobipy is available."
        ) from exc
    gp = gp_mod
    GRB = grb_mod
    return gp, GRB


def get_converter_class():
    require_gurobi()
    from rpo_milp.exp17_milp.sgcn_milp_converter import Exp17SGCNMILPConverter

    return Exp17SGCNMILPConverter


def prepare_base_profiles():
    """Return IEEE-33 base load and default PV configuration."""
    pload_standard_kw = np.array(
        [
            0, 100, 90, 120, 60, 60, 200, 200, 60, 60,
            45, 60, 60, 120, 60, 60, 60, 90, 90, 90,
            90, 90, 90, 420, 420, 60, 60, 60, 420, 400,
            450, 410, 60,
        ],
        dtype=float,
    )
    qload_standard_kvar = np.array(
        [
            0, 60, 40, 80, 30, 20, 100, 100, 20, 20,
            30, 35, 35, 80, 10, 20, 20, 40, 40, 40,
            40, 40, 50, 200, 200, 25, 25, 20, 70, 600,
            70, 100, 40,
        ],
        dtype=float,
    )

    p_load_mw = pload_standard_kw / 1000.0
    q_load_mvar = qload_standard_kvar / 1000.0
    pv_nodes = np.array([7, 14, 22, 29], dtype=int)
    s_rated_mva = np.array([1.5, 2.0, 3.0, 3.5], dtype=float)
    return p_load_mw, q_load_mvar, pv_nodes, s_rated_mva


def get_standard_radial_topology():
    """Exp17 expects the standard IEEE-33 radial topology."""
    topo_mask = np.zeros(37, dtype=bool)
    topo_mask[:32] = True
    return topo_mask


def get_default_operating_point(pv_pu: float = 0.8):
    p_load, q_load, pv_nodes, s_rated = prepare_base_profiles()
    p_pv_available = float(pv_pu) * s_rated
    topo_mask = get_standard_radial_topology()
    return p_load, q_load, p_pv_available, topo_mask, pv_nodes, s_rated


def safe_torch_load(path: str | Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_numpy(value, *, dtype=float):
    if value is None:
        return np.array([], dtype=dtype)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def worst_voltage_margin(yv, v_lower=0.95, v_upper=1.05):
    import torch

    over = yv.max(dim=1, keepdim=True).values - float(v_upper)
    under = float(v_lower) - yv.min(dim=1, keepdim=True).values
    return torch.maximum(over, under)


def choose_dataset_sample(data, cfg, requested_index: int):
    import torch

    n = int(data["X"].shape[0])
    if requested_index >= 0:
        if requested_index >= n:
            raise ValueError(f"sample index {requested_index} out of range 0..{n - 1}")
        return int(requested_index), "requested"

    yv_worst = worst_voltage_margin(
        data["Y_V"].float(),
        cfg.get("v_lower", 0.95),
        cfg.get("v_upper", 1.05),
    ).reshape(-1)
    yi_worst = data["Y_I"].float().max(dim=1).values.reshape(-1)
    pv_sum = data["pv_p"].float().sum(dim=1)

    safe = (yv_worst <= 0.0) & (yi_worst <= 0.0)
    with_pv = pv_sum > 1e-6
    candidates = torch.where(safe & with_pv)[0]
    reason = "auto_safe_with_pv"
    if candidates.numel() == 0:
        candidates = torch.where(safe)[0]
        reason = "auto_safe"
    if candidates.numel() == 0:
        candidates = torch.arange(n)
        reason = "auto_any"

    candidate_pv = pv_sum[candidates]
    target = torch.median(candidate_pv)
    local = torch.argmin(torch.abs(candidate_pv - target))
    return int(candidates[local].item()), reason


def get_dataset_operating_point(
    data_path: str | Path,
    cfg: dict,
    *,
    sample_index: int = -1,
):
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    data = safe_torch_load(data_path)
    if not isinstance(data, dict):
        raise TypeError(f"dataset must be a dict: {data_path}")

    required = ["X", "pv_nodes", "S_pv_mva", "pv_p", "pv_q", "Y_V", "Y_I"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"dataset missing required keys for dataset source: {missing}")

    sample_idx, reason = choose_dataset_sample(data, cfg, sample_index)
    x_base = data["X"][sample_idx].detach().cpu().float().numpy()[:, :2]
    pv_nodes = data["pv_nodes"].detach().cpu().numpy().astype(int).reshape(-1)
    s_rated = data["S_pv_mva"].detach().cpu().float().numpy().reshape(-1)
    p_pv = data["pv_p"][sample_idx].detach().cpu().float().numpy().reshape(-1)
    pv_q_base = data["pv_q"][sample_idx].detach().cpu().float().numpy().reshape(-1)
    ess_nodes = to_numpy(data.get("ess_nodes"), dtype=int).reshape(-1)
    ess_s_rated = to_numpy(data.get("S_ess_mva")).reshape(-1)
    ess_p_base = (
        to_numpy(data["ess_p"][sample_idx]).reshape(-1)
        if "ess_p" in data
        else np.array([], dtype=float)
    )
    ess_q_base = (
        to_numpy(data["ess_q"][sample_idx]).reshape(-1)
        if "ess_q" in data
        else np.array([], dtype=float)
    )
    q_device_nodes = to_numpy(data.get("q_device_nodes"), dtype=int).reshape(-1)
    q_device_min = to_numpy(data.get("q_device_min")).reshape(-1)
    q_device_max = to_numpy(data.get("q_device_max")).reshape(-1)
    q_device_q_base = (
        to_numpy(data["qdev_q"][sample_idx]).reshape(-1)
        if "qdev_q" in data
        else np.array([], dtype=float)
    )
    q_device_names = (
        [str(name) for name in data.get("q_device_names", [])]
        if q_device_nodes.size
        else []
    )
    topo_mask = get_standard_radial_topology()

    meta = {
        "data_path": str(data_path),
        "sample_index": sample_idx,
        "sample_reason": reason,
        "hour": int(data["hour"][sample_idx].item()) if "hour" in data else None,
        "mode": int(data["mode"][sample_idx].item()) if "mode" in data else None,
        "Pload_sum": float(data["Pload"][sample_idx].sum().item()) if "Pload" in data else None,
        "Qload_sum": float(data["Qload"][sample_idx].sum().item()) if "Qload" in data else None,
        "X_P_sum": float(x_base[:, 0].sum()),
        "X_Q_sum": float(x_base[:, 1].sum()),
        "pv_q_base": pv_q_base,
        "ess_nodes": ess_nodes,
        "S_ess_mva": ess_s_rated,
        "ess_p_base": ess_p_base,
        "ess_q_base": ess_q_base,
        "q_device_nodes": q_device_nodes,
        "q_device_min": q_device_min,
        "q_device_max": q_device_max,
        "q_device_q_base": q_device_q_base,
        "q_device_names": q_device_names,
    }
    return x_base, p_pv, pv_q_base, topo_mask, pv_nodes, s_rated, meta


def parse_vector_arg(value: str | None, *, expected: int, name: str):
    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    else:
        data = json.loads(value)
    array = np.asarray(data, dtype=float).reshape(-1)
    if array.size != expected:
        raise ValueError(f"{name} must have length {expected}, got {array.size}")
    return array


def normalize_optional_vector(value, *, dtype=float):
    if value is None:
        return np.array([], dtype=dtype)
    return np.asarray(value, dtype=dtype).reshape(-1)


def validate_inputs(
    p_load,
    q_load,
    p_pv_available,
    topo_mask,
    pv_nodes,
    s_rated,
):
    p_load = np.asarray(p_load, dtype=float).reshape(-1)
    q_load = np.asarray(q_load, dtype=float).reshape(-1)
    p_pv_available = np.asarray(p_pv_available, dtype=float).reshape(-1)
    topo_mask = np.asarray(topo_mask, dtype=bool).reshape(-1)
    pv_nodes = np.asarray(pv_nodes, dtype=int).reshape(-1)
    s_rated = np.asarray(s_rated, dtype=float).reshape(-1)

    if p_load.size != 33:
        raise ValueError(f"P_load length must be 33, got {p_load.size}")
    if q_load.size != 33:
        raise ValueError(f"Q_load length must be 33, got {q_load.size}")
    if p_pv_available.size != pv_nodes.size:
        raise ValueError(
            f"P_pv_available length must be {pv_nodes.size}, got {p_pv_available.size}"
        )
    if s_rated.size != pv_nodes.size:
        raise ValueError(f"S_rated length must be {pv_nodes.size}, got {s_rated.size}")
    if topo_mask.size != 37:
        raise ValueError(f"topo_mask length must be 37, got {topo_mask.size}")
    if np.any(p_pv_available < -1e-12):
        raise ValueError("P_pv_available cannot contain negative values")
    if np.any(s_rated <= 0.0):
        raise ValueError("S_rated must be positive")
    if np.any(p_pv_available > s_rated + 1e-9):
        bad = np.where(p_pv_available > s_rated + 1e-9)[0][0]
        raise ValueError(
            f"PV at bus {pv_nodes[bad]} has P={p_pv_available[bad]:.6f} MW "
            f"> S={s_rated[bad]:.6f} MVA"
        )

    return p_load, q_load, p_pv_available, topo_mask, pv_nodes, s_rated


def build_injection_expressions(model, q_pv, p_load, q_load, p_pv_available, pv_nodes):
    """Build 33x2 [P_net, Q_net] expressions for the Exp17 converter."""
    pv_to_idx = {int(bus): idx for idx, bus in enumerate(pv_nodes)}
    x_vars = [[None, None] for _ in range(33)]

    for bus in range(33):
        p_inj = -float(p_load[bus])
        q_inj = -float(q_load[bus])
        if bus in pv_to_idx:
            idx = pv_to_idx[bus]
            p_inj += float(p_pv_available[idx])
            x_vars[bus][0] = gp.LinExpr(p_inj)
            x_vars[bus][1] = gp.LinExpr(q_inj) + q_pv[idx]
        else:
            x_vars[bus][0] = gp.LinExpr(p_inj)
            x_vars[bus][1] = gp.LinExpr(q_inj)

    return x_vars


def build_dataset_injection_expressions(
    model,
    q_pv,
    q_ess,
    q_device,
    x_base,
    pv_q_base,
    pv_nodes,
    ess_q_base,
    ess_nodes,
    q_device_q_base,
    q_device_nodes,
):
    """Build [P_net, Q_net] expressions around a dataset sample.

    The dataset ``X`` already contains the net injection seen during training,
    including load, PV, ESS, and other reactive devices. We keep fixed loads and
    active power unchanged, subtract the base controllable reactive injections,
    and add the optimized controllable reactive variables.
    """
    x_base = np.asarray(x_base, dtype=float)
    if x_base.shape != (33, 2):
        raise ValueError(f"x_base must have shape (33, 2), got {x_base.shape}")
    pv_q_base = np.asarray(pv_q_base, dtype=float).reshape(-1)
    pv_to_idx = {int(bus): idx for idx, bus in enumerate(pv_nodes)}
    ess_to_idx = {int(bus): idx for idx, bus in enumerate(ess_nodes)}
    qdev_to_idx = {int(bus): idx for idx, bus in enumerate(q_device_nodes)}
    x_vars = [[None, None] for _ in range(33)]

    for bus in range(33):
        x_vars[bus][0] = gp.LinExpr(float(x_base[bus, 0]))
        q_expr = gp.LinExpr(float(x_base[bus, 1]))
        if bus in pv_to_idx:
            idx = pv_to_idx[bus]
            q_expr += -float(pv_q_base[idx]) + q_pv[idx]
        if bus in ess_to_idx:
            idx = ess_to_idx[bus]
            q_expr += -float(ess_q_base[idx]) + q_ess[idx]
        if bus in qdev_to_idx:
            idx = qdev_to_idx[bus]
            q_expr += -float(q_device_q_base[idx]) + q_device[idx]
        x_vars[bus][1] = q_expr

    return x_vars


def add_pv_capacity_constraints(model, q_pv, p_pv_available, pv_nodes, s_rated):
    q_caps = []
    for idx, (bus, p_avail, s_max) in enumerate(
        zip(pv_nodes, p_pv_available, s_rated)
    ):
        q_cap_sq = max(float(s_max) ** 2 - float(p_avail) ** 2, 0.0)
        q_cap = float(np.sqrt(q_cap_sq))
        q_caps.append(q_cap)
        # Bounds help presolve, while the quadratic constraint keeps the
        # apparent-power model explicit for compatibility with older scripts.
        q_pv[idx].LB = -q_cap
        q_pv[idx].UB = q_cap
        model.addQConstr(
            q_pv[idx] * q_pv[idx] <= q_cap_sq,
            name=f"PV_Cap_{int(bus)}",
        )
    return np.asarray(q_caps, dtype=float)


def add_ess_capacity_constraints(model, q_ess, ess_p_base, ess_nodes, ess_s_rated):
    q_caps = []
    for idx, (bus, p_base, s_max) in enumerate(
        zip(ess_nodes, ess_p_base, ess_s_rated)
    ):
        q_cap_sq = max(float(s_max) ** 2 - float(p_base) ** 2, 0.0)
        q_cap = float(np.sqrt(q_cap_sq))
        q_caps.append(q_cap)
        q_ess[idx].LB = -q_cap
        q_ess[idx].UB = q_cap
        model.addQConstr(
            q_ess[idx] * q_ess[idx] <= q_cap_sq,
            name=f"ESS_Cap_{int(bus)}",
        )
    return np.asarray(q_caps, dtype=float)


def add_q_device_bounds(model, q_device, q_device_nodes, q_device_min, q_device_max):
    for idx, bus in enumerate(q_device_nodes):
        q_device[idx].LB = float(q_device_min[idx])
        q_device[idx].UB = float(q_device_max[idx])
    return np.asarray(q_device_min, dtype=float), np.asarray(q_device_max, dtype=float)


def build_control_labels(pv_nodes, ess_nodes, q_device_nodes, q_device_names):
    labels = [f"PV@Bus{int(bus) + 1}" for bus in pv_nodes]
    labels.extend(f"ESS@Bus{int(bus) + 1}" for bus in ess_nodes)
    for idx, bus in enumerate(q_device_nodes):
        name = q_device_names[idx] if idx < len(q_device_names) else f"QDev{idx + 1}"
        labels.append(f"{name}@Bus{int(bus) + 1}")
    return labels


def build_control_base_vector(
    pv_q_base,
    ess_q_base,
    q_device_q_base,
    *,
    n_pv: int,
    n_ess: int,
    n_qdev: int,
):
    pv_ref = np.zeros(n_pv, dtype=float) if pv_q_base is None else np.asarray(pv_q_base, dtype=float).reshape(-1)
    ess_ref = np.asarray(ess_q_base, dtype=float).reshape(-1)
    qdev_ref = np.asarray(q_device_q_base, dtype=float).reshape(-1)
    if pv_ref.size != n_pv:
        raise ValueError(f"PV Q reference length must be {n_pv}, got {pv_ref.size}")
    if ess_ref.size != n_ess:
        raise ValueError(f"ESS Q reference length must be {n_ess}, got {ess_ref.size}")
    if qdev_ref.size != n_qdev:
        raise ValueError(f"Q-device reference length must be {n_qdev}, got {qdev_ref.size}")
    return np.concatenate([pv_ref, ess_ref, qdev_ref])


def control_variable_list(q_pv, q_ess, q_device, n_pv: int, n_ess: int, n_qdev: int):
    return (
        [q_pv[idx] for idx in range(n_pv)]
        + [q_ess[idx] for idx in range(n_ess)]
        + [q_device[idx] for idx in range(n_qdev)]
    )


def build_control_node_vector(pv_nodes, ess_nodes, q_device_nodes):
    return np.concatenate(
        [
            np.asarray(pv_nodes, dtype=int).reshape(-1),
            np.asarray(ess_nodes, dtype=int).reshape(-1),
            np.asarray(q_device_nodes, dtype=int).reshape(-1),
        ]
    )


def build_control_bound_vectors(q_caps, ess_q_caps, qdev_min, qdev_max):
    q_caps = np.asarray(q_caps, dtype=float).reshape(-1)
    ess_q_caps = np.asarray(ess_q_caps, dtype=float).reshape(-1)
    qdev_min = np.asarray(qdev_min, dtype=float).reshape(-1)
    qdev_max = np.asarray(qdev_max, dtype=float).reshape(-1)
    lower = np.concatenate([-q_caps, -ess_q_caps, qdev_min])
    upper = np.concatenate([q_caps, ess_q_caps, qdev_max])
    if lower.shape != upper.shape:
        raise ValueError("Control lower/upper bound vectors have different shapes")
    if np.any(upper < lower):
        bad = int(np.where(upper < lower)[0][0])
        raise ValueError(
            f"Invalid control bounds at index {bad}: lower={lower[bad]}, upper={upper[bad]}"
        )
    return lower, upper


def add_conditional_qnet_bounds(
    model,
    x_vars,
    *,
    x_base,
    q_load,
    control_nodes,
    control_base,
    control_lower,
    control_upper,
):
    """Constrain controllable-node Q_net to the current operating-point domain.

    The bounds are conditional on the current fixed net injection and the
    current device capability:

        Q_net = Q_fixed_noncontrol + sum(Q_control_at_node)

    For dataset samples, ``x_base[:, 1]`` already includes the sampled
    controllable Q. For profile inputs, the fixed term is ``-q_load``.
    """
    control_nodes = np.asarray(control_nodes, dtype=int).reshape(-1)
    control_base = np.asarray(control_base, dtype=float).reshape(-1)
    control_lower = np.asarray(control_lower, dtype=float).reshape(-1)
    control_upper = np.asarray(control_upper, dtype=float).reshape(-1)

    if not (
        control_nodes.size
        == control_base.size
        == control_lower.size
        == control_upper.size
    ):
        raise ValueError("Control nodes, base values, and bounds must have the same length")

    if control_nodes.size == 0:
        return {
            "enabled": True,
            "nodes": [],
            "lower": [],
            "upper": [],
            "fixed_q": [],
        }

    if np.any((control_nodes < 0) | (control_nodes >= 33)):
        bad = int(control_nodes[np.where((control_nodes < 0) | (control_nodes >= 33))[0][0]])
        raise ValueError(f"Control node index out of range: {bad}")

    if x_base is not None:
        x_base = np.asarray(x_base, dtype=float)
        if x_base.shape != (33, 2):
            raise ValueError(f"x_base must have shape (33, 2), got {x_base.shape}")
        fixed_q = x_base[:, 1].astype(float).copy()
    else:
        q_load = np.asarray(q_load, dtype=float).reshape(-1)
        if q_load.size != 33:
            raise ValueError(f"q_load must have length 33 when x_base is None, got {q_load.size}")
        fixed_q = -q_load.astype(float).copy()

    has_control = np.zeros(33, dtype=bool)
    qnet_lower = fixed_q.copy()
    qnet_upper = fixed_q.copy()

    for idx, bus in enumerate(control_nodes):
        bus = int(bus)
        has_control[bus] = True
        fixed_q[bus] -= float(control_base[idx])
        qnet_lower[bus] -= float(control_base[idx])
        qnet_upper[bus] -= float(control_base[idx])

    for idx, bus in enumerate(control_nodes):
        bus = int(bus)
        qnet_lower[bus] += float(control_lower[idx])
        qnet_upper[bus] += float(control_upper[idx])

    nodes = np.where(has_control)[0]
    for bus in nodes:
        if qnet_upper[bus] < qnet_lower[bus]:
            raise ValueError(
                f"Invalid conditional Q_net bounds at bus {int(bus) + 1}: "
                f"lower={qnet_lower[bus]}, upper={qnet_upper[bus]}"
            )
        model.addConstr(
            x_vars[int(bus)][1] >= float(qnet_lower[bus]),
            name=f"Conditional_Qnet_lb_bus{int(bus) + 1}",
        )
        model.addConstr(
            x_vars[int(bus)][1] <= float(qnet_upper[bus]),
            name=f"Conditional_Qnet_ub_bus{int(bus) + 1}",
        )

    return {
        "enabled": True,
        "nodes": nodes.astype(int).tolist(),
        "lower": qnet_lower[nodes].astype(float).tolist(),
        "upper": qnet_upper[nodes].astype(float).tolist(),
        "fixed_q": fixed_q[nodes].astype(float).tolist(),
    }


def sanitize_name(value: str) -> str:
    keep = []
    for char in str(value):
        keep.append(char if char.isalnum() else "_")
    return "".join(keep).strip("_") or "control"


def add_surrogate_physical_output_bounds(model, outputs, *, enabled: bool):
    if not enabled:
        return {}
    bounds = {
        "Vdev_total_min": 0.0,
        "WorstI_min": -1.0,
        "Vworst_min": -0.05,
    }
    model.addConstr(outputs.Vdev_total >= bounds["Vdev_total_min"], name="Physical_Vdev_total_nonnegative")
    model.addConstr(outputs.WorstI >= bounds["WorstI_min"], name="Physical_WorstI_lower")
    model.addConstr(outputs.Vworst >= bounds["Vworst_min"], name="Physical_Vworst_lower")
    return bounds


def add_control_trust_region(
    model,
    control_vars,
    q_reference,
    lower,
    upper,
    labels,
    *,
    fraction: float,
):
    q_reference = np.asarray(q_reference, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    if not (control_vars and q_reference.size == lower.size == upper.size == len(control_vars)):
        if len(control_vars) == 0 and q_reference.size == lower.size == upper.size == 0:
            return q_reference, np.array([], dtype=float)
        raise ValueError("Control vars, references, and bounds must have the same length")
    if fraction < 0.0:
        raise ValueError("trust_region_fraction must be nonnegative")

    q_ref = np.clip(q_reference, lower, upper)
    width = np.maximum(upper - lower, 0.0)
    radius = float(fraction) * width
    if fraction >= 1.0:
        return q_ref, radius

    for idx, var in enumerate(control_vars):
        name = sanitize_name(labels[idx] if idx < len(labels) else f"Q_{idx}")
        model.addConstr(var <= float(q_ref[idx] + radius[idx]), name=f"Trust_ub_{idx}_{name}")
        model.addConstr(var >= float(q_ref[idx] - radius[idx]), name=f"Trust_lb_{idx}_{name}")
    return q_ref, radius


def add_control_deviation_penalty(
    model,
    control_vars,
    q_reference,
    labels,
    *,
    weight: float,
):
    penalty = gp.LinExpr(0.0)
    aux_vars = []
    if weight <= 0.0:
        return penalty, aux_vars
    for idx, var in enumerate(control_vars):
        name = sanitize_name(labels[idx] if idx < len(labels) else f"Q_{idx}")
        dev = model.addVar(lb=0.0, name=f"AbsDev_{idx}_{name}")
        ref = float(q_reference[idx])
        model.addConstr(dev >= var - ref, name=f"AbsDev_pos_{idx}_{name}")
        model.addConstr(dev >= ref - var, name=f"AbsDev_neg_{idx}_{name}")
        penalty += float(weight) * dev
        aux_vars.append(dev)
    return penalty, aux_vars


def build_objective_expression(
    outputs,
    objective: str,
    *,
    loss_weight: float,
    voltage_weight: float,
    loss_expr=None,
):
    if objective == "vdev":
        return outputs.Vdev_total
    raise ValueError("Exp17 has no learned Ploss head; use objective='vdev'")


def objective_value_from_components(
    output_values,
    objective: str,
    *,
    loss_weight: float,
    voltage_weight: float,
    loss_value=None,
):
    if output_values is None:
        return None
    vdev = float(output_values["Vdev_total"])
    if objective == "vdev":
        return vdev
    raise ValueError("Exp17 has no learned Ploss head; use objective='vdev'")


def select_loss_objective_expression(
    converter,
    outputs,
    *,
    loss_objective_model: str,
    base_kv: float,
    base_mva: float,
    surrogate_loss_weight: float,
):
    if loss_objective_model in {"none", "disabled"}:
        return None
    raise ValueError("Exp17 has no learned Ploss head; use --loss-objective-model none")


def add_surrogate_loss_consistency_constraints(
    model,
    converter,
    outputs,
    *,
    loss_objective_model: str,
    base_kv: float,
    base_mva: float,
    relative_tol: float,
    absolute_tol: float,
):
    if loss_objective_model not in {"none", "disabled"}:
        raise ValueError("Exp17 loss consistency is unavailable without a learned Ploss head")
    return None


def add_objective(
    model,
    outputs,
    objective: str,
    loss_weight: float,
    voltage_weight: float,
):
    expr = build_objective_expression(
        outputs,
        objective,
        loss_weight=loss_weight,
        voltage_weight=voltage_weight,
    )
    model.setObjective(expr, GRB.MINIMIZE)


def add_surrogate_safety_constraints(
    model,
    outputs,
    *,
    voltage_margin: float,
    current_margin: float,
    diagnose_slack: bool,
    slack_penalty: float,
    base_objective,
):
    if not diagnose_slack:
        model.addConstr(
            outputs.Vworst <= float(voltage_margin),
            name="Surrogate_Vworst_safe",
        )
        model.addConstr(
            outputs.WorstI <= float(current_margin),
            name="Surrogate_WorstI_safe",
        )
        return None, None

    s_v = model.addVar(lb=0.0, name="slack_Vworst_safe")
    s_i = model.addVar(lb=0.0, name="slack_WorstI_safe")
    model.addConstr(
        outputs.Vworst <= float(voltage_margin) + s_v,
        name="Surrogate_Vworst_safe_soft",
    )
    model.addConstr(
        outputs.WorstI <= float(current_margin) + s_i,
        name="Surrogate_WorstI_safe_soft",
    )
    model.setObjective(
        float(slack_penalty) * (s_v + s_i) + base_objective,
        GRB.MINIMIZE,
    )
    return s_v, s_i


def run_single_step_rpo(
    p_load,
    q_load,
    p_pv_available,
    topo_mask,
    *,
    pv_nodes,
    s_rated,
    engine_path=DEFAULT_ENGINE_PATH,
    relu_formulation="big_m",
    big_m_scale=1.0,
    fallback_big_m=1e3,
    objective="vdev",
    loss_objective_model="none",
    surrogate_loss_weight=0.5,
    surrogate_loss_consistency_rel=0.5,
    surrogate_loss_consistency_abs=0.002,
    loss_weight=0.0,
    voltage_weight=1e-3,
    lambda_q=0.0,
    voltage_margin=0.0,
    current_margin=0.0,
    physical_output_bounds=True,
    trust_region_fraction=1.0,
    control_deviation_penalty=0.0,
    base_kv=12.66,
    base_mva=1.0,
    time_limit=300.0,
    mip_gap=0.01,
    dual_reductions=0,
    diagnose_slack=False,
    slack_penalty=1e4,
    output_flag=1,
    export_lp=None,
    iis_path=None,
    x_base=None,
    pv_q_base=None,
    ess_nodes=None,
    ess_s_rated=None,
    ess_p_base=None,
    ess_q_base=None,
    q_device_nodes=None,
    q_device_min=None,
    q_device_max=None,
    q_device_q_base=None,
    q_device_names=None,
):
    gp_mod, grb_mod = require_gurobi()
    converter_class = get_converter_class()

    p_load, q_load, p_pv_available, topo_mask, pv_nodes, s_rated = validate_inputs(
        p_load,
        q_load,
        p_pv_available,
        topo_mask,
        pv_nodes,
        s_rated,
    )
    ess_nodes = normalize_optional_vector(ess_nodes, dtype=int)
    ess_s_rated = normalize_optional_vector(ess_s_rated)
    ess_p_base = normalize_optional_vector(ess_p_base)
    ess_q_base = normalize_optional_vector(ess_q_base)
    q_device_nodes = normalize_optional_vector(q_device_nodes, dtype=int)
    q_device_min = normalize_optional_vector(q_device_min)
    q_device_max = normalize_optional_vector(q_device_max)
    q_device_q_base = normalize_optional_vector(q_device_q_base)
    q_device_names = [] if q_device_names is None else [str(x) for x in q_device_names]
    if float(lambda_q) != 0.0:
        raise ValueError("Exp17 objective is voltage-only; set lambda_q=0")

    if len(ess_nodes) not in {0, len(ess_s_rated), len(ess_p_base), len(ess_q_base)}:
        raise ValueError("ESS node, rating, active-power, and reactive-power arrays must have the same length")
    if len(ess_nodes) and not (
        len(ess_s_rated) == len(ess_p_base) == len(ess_q_base) == len(ess_nodes)
    ):
        raise ValueError("ESS node, rating, active-power, and reactive-power arrays must have the same length")
    if len(q_device_nodes) and not (
        len(q_device_min) == len(q_device_max) == len(q_device_q_base) == len(q_device_nodes)
    ):
        raise ValueError("Q-device node, min, max, and base reactive arrays must have the same length")

    model = gp_mod.Model("Exp17_SingleStep_RPO")
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = float(mip_gap)
    model.Params.OutputFlag = int(output_flag)
    model.Params.NonConvex = 2
    model.Params.DualReductions = int(dual_reductions)

    converter = converter_class(
        engine_path,
        relu_formulation=relu_formulation,
        fallback_big_m=fallback_big_m,
        big_m_scale=big_m_scale,
    )

    q_pv = model.addVars(len(pv_nodes), lb=-grb_mod.INFINITY, ub=grb_mod.INFINITY, name="Q_pv")
    q_ess = model.addVars(len(ess_nodes), lb=-grb_mod.INFINITY, ub=grb_mod.INFINITY, name="Q_ess")
    q_device = model.addVars(
        len(q_device_nodes),
        lb=-grb_mod.INFINITY,
        ub=grb_mod.INFINITY,
        name="Q_device",
    )
    q_caps = add_pv_capacity_constraints(
        model,
        q_pv,
        p_pv_available,
        pv_nodes,
        s_rated,
    )
    ess_q_caps = add_ess_capacity_constraints(
        model,
        q_ess,
        ess_p_base,
        ess_nodes,
        ess_s_rated,
    )
    qdev_min, qdev_max = add_q_device_bounds(
        model,
        q_device,
        q_device_nodes,
        q_device_min,
        q_device_max,
    )
    labels = build_control_labels(
        pv_nodes,
        ess_nodes,
        q_device_nodes,
        q_device_names,
    )
    control_vars = control_variable_list(
        q_pv,
        q_ess,
        q_device,
        len(pv_nodes),
        len(ess_nodes),
        len(q_device_nodes),
    )
    control_nodes = build_control_node_vector(
        pv_nodes,
        ess_nodes,
        q_device_nodes,
    )
    control_lower, control_upper = build_control_bound_vectors(
        q_caps,
        ess_q_caps,
        qdev_min,
        qdev_max,
    )
    control_base = build_control_base_vector(
        pv_q_base if x_base is not None else None,
        ess_q_base,
        q_device_q_base,
        n_pv=len(pv_nodes),
        n_ess=len(ess_nodes),
        n_qdev=len(q_device_nodes),
    )
    q_reference = control_base.copy()
    q_reference, trust_radius = add_control_trust_region(
        model,
        control_vars,
        q_reference,
        control_lower,
        control_upper,
        labels,
        fraction=float(trust_region_fraction),
    )
    if x_base is None:
        x_vars = build_injection_expressions(
            model,
            q_pv,
            p_load,
            q_load,
            p_pv_available,
            pv_nodes,
        )
    else:
        if pv_q_base is None:
            raise ValueError("pv_q_base is required when x_base is provided")
        x_vars = build_dataset_injection_expressions(
            model,
            q_pv,
            q_ess,
            q_device,
            x_base,
            pv_q_base,
            pv_nodes,
            ess_q_base,
            ess_nodes,
            q_device_q_base,
            q_device_nodes,
        )

    conditional_qnet_bounds = add_conditional_qnet_bounds(
        model,
        x_vars,
        x_base=x_base,
        q_load=q_load,
        control_nodes=control_nodes,
        control_base=control_base,
        control_lower=control_lower,
        control_upper=control_upper,
    )

    print("Embedding Exp17 SGCN surrogate constraints...")
    outputs = converter.embed_sgcn_constraints(
        model,
        x_vars,
        topo_mask=topo_mask,
        name_prefix="exp17_rpo",
    )

    output_physical_bounds = add_surrogate_physical_output_bounds(
        model,
        outputs,
        enabled=bool(physical_output_bounds),
    )
    loss_objective_expr = select_loss_objective_expression(
        converter,
        outputs,
        loss_objective_model=str(loss_objective_model),
        base_kv=float(base_kv),
        base_mva=float(base_mva),
        surrogate_loss_weight=float(surrogate_loss_weight),
    )
    loss_consistency = add_surrogate_loss_consistency_constraints(
        model,
        converter,
        outputs,
        loss_objective_model=str(loss_objective_model),
        base_kv=float(base_kv),
        base_mva=float(base_mva),
        relative_tol=float(surrogate_loss_consistency_rel),
        absolute_tol=float(surrogate_loss_consistency_abs),
    )
    base_objective = build_objective_expression(
        outputs,
        objective,
        loss_weight=loss_weight,
        voltage_weight=voltage_weight,
        loss_expr=loss_objective_expr,
    )
    deviation_penalty, deviation_aux = add_control_deviation_penalty(
        model,
        control_vars,
        q_reference,
        labels,
        weight=float(control_deviation_penalty),
    )
    reactive_flow_penalty = gp_mod.QuadExpr()
    protected_objective = base_objective + deviation_penalty + reactive_flow_penalty
    model.setObjective(protected_objective, grb_mod.MINIMIZE)
    slack_v, slack_i = add_surrogate_safety_constraints(
        model,
        outputs,
        voltage_margin=voltage_margin,
        current_margin=current_margin,
        diagnose_slack=diagnose_slack,
        slack_penalty=slack_penalty,
        base_objective=protected_objective,
    )

    if export_lp:
        export_path = Path(export_lp)
        if not export_path.is_absolute():
            export_path = REPO_ROOT / export_path
        export_path.parent.mkdir(parents=True, exist_ok=True)
        model.write(str(export_path))
        print(f"LP/MPS written before solve: {export_path}")

    print("Starting Gurobi solve...")
    model.optimize()

    result = {
        "status": int(model.status),
        "status_name": status_name(model.status),
        "objective": None,
        "objective_mode": objective,
        "loss_objective_model": str(loss_objective_model),
        "surrogate_loss_weight": float(surrogate_loss_weight),
        "surrogate_loss_consistency": loss_consistency,
        "loss_weight": float(loss_weight),
        "voltage_weight": float(voltage_weight),
        "Q_pv_opt": None,
        "Q_pv_caps": q_caps,
        "Q_ess_opt": None,
        "Q_ess_caps": ess_q_caps,
        "Q_device_opt": None,
        "Q_device_min": qdev_min,
        "Q_device_max": qdev_max,
        "Q_control_opt": None,
        "Q_control_labels": labels,
        "Q_control_reference": q_reference,
        "Q_control_lower": control_lower,
        "Q_control_upper": control_upper,
        "Q_control_trust_radius": trust_radius,
        "conditional_qnet_bounds": conditional_qnet_bounds,
        "trust_region_fraction": float(trust_region_fraction),
        "physical_output_bounds": output_physical_bounds,
        "lambda_q": float(lambda_q),
        "q_flow_regularization": None,
        "q_flow_penalty": None,
        "q_flow_branch_values": None,
        "control_deviation_penalty_weight": float(control_deviation_penalty),
        "control_deviation_abs": None,
        "loss_objective_value": None,
        "objective_components": None,
        "outputs": None,
        "safety_slack": None,
        "binary_count_estimate": converter.binary_count,
        "sol_count": int(model.SolCount),
    }

    if model.SolCount > 0:
        q_solution = np.array([q_pv[idx].X for idx in range(len(pv_nodes))], dtype=float)
        q_ess_solution = np.array(
            [q_ess[idx].X for idx in range(len(ess_nodes))],
            dtype=float,
        )
        q_device_solution = np.array(
            [q_device[idx].X for idx in range(len(q_device_nodes))],
            dtype=float,
        )
        q_control_solution = np.concatenate(
            [q_solution, q_ess_solution, q_device_solution]
        )
        output_values = {
            "Vdev_total": float(outputs.Vdev_total.X),
            "Vworst": float(outputs.Vworst.X),
            "WorstI": float(outputs.WorstI.X),
            "Ploss_total": None,
            "V_nodes": np.array([var.X for var in outputs.V_nodes], dtype=float),
        }
        loss_objective_value = None
        deviation_values = np.array([dev.X for dev in deviation_aux], dtype=float)
        guarded_objective = objective_value_from_components(
            output_values,
            objective,
            loss_weight=loss_weight,
            voltage_weight=voltage_weight,
            loss_value=loss_objective_value,
        )
        deviation_penalty_value = float(control_deviation_penalty) * float(
            deviation_values.sum()
        )
        q_flow_regularization = None
        q_flow_penalty_value = None
        q_flow_branch_values = None
        objective_components = {
            "guarded_objective": guarded_objective,
            "loss_objective": None,
            "surrogate_ploss": None,
            "q_flow_regularization": q_flow_regularization,
            "q_flow_penalty": q_flow_penalty_value,
            "control_deviation_penalty": deviation_penalty_value,
            "slack_penalty": None,
        }
        slack_values = None
        if slack_v is not None and slack_i is not None:
            slack_values = {
                "Vworst": float(slack_v.X),
                "WorstI": float(slack_i.X),
            }
            objective_components["slack_penalty"] = float(slack_penalty) * (
                slack_values["Vworst"] + slack_values["WorstI"]
            )
        result.update(
            {
                "objective": float(model.ObjVal),
                "Q_pv_opt": q_solution,
                "Q_ess_opt": q_ess_solution,
                "Q_device_opt": q_device_solution,
                "Q_control_opt": q_control_solution,
                "control_deviation_abs": deviation_values,
                "q_flow_regularization": q_flow_regularization,
                "q_flow_penalty": q_flow_penalty_value,
                "q_flow_branch_values": q_flow_branch_values,
                "loss_objective_value": loss_objective_value,
                "objective_components": objective_components,
                "outputs": output_values,
                "safety_slack": slack_values,
            }
        )
        print_solution(
            result,
            pv_nodes=pv_nodes,
            p_pv_available=p_pv_available,
            ess_nodes=ess_nodes,
            q_device_nodes=q_device_nodes,
            q_device_names=q_device_names,
        )
    elif model.status == grb_mod.INFEASIBLE and iis_path:
        iis_out = Path(iis_path)
        if not iis_out.is_absolute():
            iis_out = REPO_ROOT / iis_out
        iis_out.parent.mkdir(parents=True, exist_ok=True)
        print("Model infeasible; computing IIS...")
        model.computeIIS()
        model.write(str(iis_out))
        print(f"IIS written: {iis_out}")
    else:
        print(f"No feasible solution. Status={model.status} ({status_name(model.status)})")
        if model.status == grb_mod.INF_OR_UNBD and int(dual_reductions) != 0:
            print("Hint: rerun with --dual-reductions 0 to distinguish infeasible vs unbounded.")
        if not diagnose_slack:
            print("Hint: rerun with --diagnose-slack to minimize safety-constraint violations.")

    return result


def status_name(status_code: int) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    return names.get(status_code, f"STATUS_{status_code}")


def print_solution(
    result,
    *,
    pv_nodes,
    p_pv_available,
    ess_nodes,
    q_device_nodes,
    q_device_names,
):
    print()
    print("Single-step reactive power optimization result")
    print(f"Status: {result['status_name']} ({result['status']})")
    print(f"Objective: {result['objective']:.8f}")
    if result.get("q_flow_regularization") is not None:
        print(
            "Reactive-flow regularizer: "
            f"J_Q_flow={result['q_flow_regularization']:.8f}, "
            f"lambda_Q*J_Q_flow={result['q_flow_penalty']:.8f}"
        )
    print(f"Estimated surrogate ReLU binaries: {result['binary_count_estimate']}")
    print()
    print("PV dispatch:")
    q_solution = result["Q_pv_opt"]
    q_caps = result["Q_pv_caps"]
    for idx, bus in enumerate(pv_nodes):
        print(
            f"  bus {int(bus):02d}: P={p_pv_available[idx]:.6f} MW, "
            f"Q={q_solution[idx]: .6f} MVar, |Q|max={q_caps[idx]:.6f}"
        )
    if len(ess_nodes):
        print()
        print("ESS dispatch:")
        q_ess_solution = result["Q_ess_opt"]
        q_ess_caps = result["Q_ess_caps"]
        for idx, bus in enumerate(ess_nodes):
            print(
                f"  bus {int(bus):02d}: "
                f"Q={q_ess_solution[idx]: .6f} MVar, |Q|max={q_ess_caps[idx]:.6f}"
            )
    if len(q_device_nodes):
        print()
        print("Reactive device dispatch:")
        q_device_solution = result["Q_device_opt"]
        q_device_min = result["Q_device_min"]
        q_device_max = result["Q_device_max"]
        for idx, bus in enumerate(q_device_nodes):
            name = q_device_names[idx] if idx < len(q_device_names) else f"QDev{idx + 1}"
            print(
                f"  {name} bus {int(bus):02d}: "
                f"Q={q_device_solution[idx]: .6f} MVar, "
                f"bounds=[{q_device_min[idx]:.6f}, {q_device_max[idx]:.6f}]"
            )
    print()
    print("Surrogate outputs:")
    for key, value in result["outputs"].items():
        if value is None:
            print(f"  {key}: unavailable")
        elif isinstance(value, np.ndarray):
            print(
                f"  {key}: array shape={value.shape}, "
                f"min={float(np.min(value)):.8f}, max={float(np.max(value)):.8f}"
            )
        else:
            print(f"  {key}: {value:.8f}")
    if result.get("safety_slack") is not None:
        print()
        print("Safety slack diagnostics:")
        for key, value in result["safety_slack"].items():
            print(f"  {key}: {value:.8f}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Single-step reactive power optimization with Exp17 SGCN-MILP."
    )
    parser.add_argument("--engine", default=str(DEFAULT_ENGINE_PATH))
    parser.add_argument(
        "--source",
        choices=["dataset", "profile"],
        default="dataset",
        help="dataset uses one sample from the current training dataset; profile uses the old IEEE33 default profile.",
    )
    parser.add_argument(
        "--data",
        default="data/ieee33_nodal_pq_correlated_raw_pool_50k.pt",
        help="Dataset path when --source dataset is used.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=-1,
        help="Dataset sample index. Use -1 for deterministic auto selection.",
    )
    parser.add_argument("--pv-pu", type=float, default=0.8)
    parser.add_argument(
        "--p-load",
        default=None,
        help="JSON list or JSON file containing 33 MW loads.",
    )
    parser.add_argument(
        "--q-load",
        default=None,
        help="JSON list or JSON file containing 33 MVar loads.",
    )
    parser.add_argument(
        "--p-pv",
        default=None,
        help="JSON list or JSON file containing active PV output at the PV buses.",
    )
    parser.add_argument(
        "--objective",
        choices=["vdev"],
        default="vdev",
        help=(
            "Exp17 has no learned Ploss head, so the MILP objective is the "
            "learned cumulative voltage deviation."
        ),
    )
    parser.add_argument(
        "--loss-objective-model",
        choices=["none"],
        default="none",
        help=(
            "No loss term is available in Exp17. Kept for CLI compatibility."
        ),
    )
    parser.add_argument(
        "--surrogate-loss-weight",
        type=float,
        default=0.5,
        help="Unused for Exp17; kept for CLI compatibility.",
    )
    parser.add_argument(
        "--surrogate-loss-consistency-rel",
        type=float,
        default=0.5,
        help=(
            "Unused for Exp17; kept for CLI compatibility."
        ),
    )
    parser.add_argument(
        "--surrogate-loss-consistency-abs",
        type=float,
        default=0.002,
        help="Unused for Exp17; kept for CLI compatibility.",
    )
    parser.add_argument("--loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--voltage-weight",
        type=float,
        default=1e-3,
        help="Unused for Exp17; kept for CLI compatibility.",
    )
    parser.add_argument(
        "--lambda-q",
        type=float,
        default=0.0,
        help="Unused for Exp17 voltage-only objective; must remain 0.",
    )
    parser.add_argument("--voltage-margin", type=float, default=0.0)
    parser.add_argument("--current-margin", type=float, default=0.0)
    parser.add_argument("--base-kv", type=float, default=12.66, help="Unused for Exp17.")
    parser.add_argument("--base-mva", type=float, default=1.0, help="Unused for Exp17.")
    parser.add_argument(
        "--trust-region-fraction",
        type=float,
        default=1.0,
        help=(
            "Limit each optimized Q control to this fraction of its full feasible "
            "range around the operating-point Q reference. Use >=1 to disable."
        ),
    )
    parser.add_argument(
        "--control-deviation-penalty",
        type=float,
        default=0.0,
        help="Optional linear penalty on absolute Q-control movement from the Q reference.",
    )
    parser.add_argument(
        "--no-physical-output-bounds",
        action="store_true",
        help="Disable nonnegative/lower physical bounds on direct surrogate outputs.",
    )
    parser.add_argument(
        "--relu-formulation",
        choices=["big_m", "general"],
        default="big_m",
    )
    parser.add_argument("--big-m-scale", type=float, default=1.0)
    parser.add_argument("--fallback-big-m", type=float, default=1e3)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument(
        "--dual-reductions",
        type=int,
        choices=[0, 1],
        default=0,
        help="Use 0 to distinguish infeasible from unbounded.",
    )
    parser.add_argument(
        "--diagnose-slack",
        action="store_true",
        help="Relax Vworst/WorstI safety constraints with nonnegative slacks.",
    )
    parser.add_argument(
        "--slack-penalty",
        type=float,
        default=1e4,
        help="Penalty on safety slacks when --diagnose-slack is used.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--export-lp", default=None)
    parser.add_argument("--iis", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    x_base = None
    pv_q_base = None
    ess_nodes = np.array([], dtype=int)
    ess_s_rated = np.array([], dtype=float)
    ess_p_base = np.array([], dtype=float)
    ess_q_base = np.array([], dtype=float)
    q_device_nodes = np.array([], dtype=int)
    q_device_min = np.array([], dtype=float)
    q_device_max = np.array([], dtype=float)
    q_device_q_base = np.array([], dtype=float)
    q_device_names = []

    if args.source == "dataset":
        # Read config only for safety limits and auto sample selection. The
        # optimization model itself will load the engine later.
        cfg = {"v_lower": 0.95, "v_upper": 1.05}
        x_base, p_pv, pv_q_base, topo_mask, pv_nodes, s_rated, meta = (
            get_dataset_operating_point(
                args.data,
                cfg,
                sample_index=args.sample_index,
            )
        )
        p_load = np.zeros(33, dtype=float)
        q_load = np.zeros(33, dtype=float)
        p_pv_override = parse_vector_arg(args.p_pv, expected=len(pv_nodes), name="p_pv")
        if p_pv_override is not None:
            p_pv = p_pv_override

        print("Operating point from dataset:")
        print(f"  data: {meta['data_path']}")
        print(
            f"  sample_index: {meta['sample_index']} "
            f"({meta['sample_reason']})"
        )
        if meta["hour"] is not None:
            print(f"  hour={meta['hour']}, mode={meta['mode']}")
        if meta["Pload_sum"] is not None:
            print(f"  Pload sum: {meta['Pload_sum']:.6f} MW")
            print(f"  Qload sum: {meta['Qload_sum']:.6f} MVar")
        print(f"  X net P sum: {meta['X_P_sum']:.6f} MW")
        print(f"  X net Q sum: {meta['X_Q_sum']:.6f} MVar")
        print(f"  PV buses: {pv_nodes.tolist()}")
        print(f"  S_pv: {s_rated.tolist()}")
        print(f"  P_pv: {p_pv.tolist()}")
        print(f"  baseline pv_q: {pv_q_base.tolist()}")
        ess_nodes = meta["ess_nodes"]
        ess_s_rated = meta["S_ess_mva"]
        ess_p_base = meta["ess_p_base"]
        ess_q_base = meta["ess_q_base"]
        q_device_nodes = meta["q_device_nodes"]
        q_device_min = meta["q_device_min"]
        q_device_max = meta["q_device_max"]
        q_device_q_base = meta["q_device_q_base"]
        q_device_names = meta["q_device_names"]
        if len(ess_nodes):
            print(f"  ESS buses: {ess_nodes.tolist()}")
            print(f"  ESS P/Q base: {ess_p_base.tolist()} / {ess_q_base.tolist()}")
        if len(q_device_nodes):
            print(f"  Q-device buses: {q_device_nodes.tolist()}")
            print(f"  Q-device base: {q_device_q_base.tolist()}")
    else:
        p_load, q_load, p_pv, topo_mask, pv_nodes, s_rated = get_default_operating_point(
            pv_pu=args.pv_pu
        )

        p_load_override = parse_vector_arg(args.p_load, expected=33, name="p_load")
        q_load_override = parse_vector_arg(args.q_load, expected=33, name="q_load")
        p_pv_override = parse_vector_arg(args.p_pv, expected=len(pv_nodes), name="p_pv")
        if p_load_override is not None:
            p_load = p_load_override
        if q_load_override is not None:
            q_load = q_load_override
        if p_pv_override is not None:
            p_pv = p_pv_override

        print("Operating point from profile:")
        print(f"  P_load sum: {float(np.sum(p_load)):.6f} MW")
        print(f"  Q_load sum: {float(np.sum(q_load)):.6f} MVar")
        print(f"  PV buses: {pv_nodes.tolist()}")
        print(f"  P_pv: {p_pv.tolist()}")
    print(f"  closed topology branches: {int(np.sum(topo_mask))} / {len(topo_mask)}")

    run_single_step_rpo(
        p_load,
        q_load,
        p_pv,
        topo_mask,
        pv_nodes=pv_nodes,
        s_rated=s_rated,
        engine_path=args.engine,
        relu_formulation=args.relu_formulation,
        big_m_scale=args.big_m_scale,
        fallback_big_m=args.fallback_big_m,
        objective=args.objective,
        loss_objective_model=args.loss_objective_model,
        surrogate_loss_weight=args.surrogate_loss_weight,
        surrogate_loss_consistency_rel=args.surrogate_loss_consistency_rel,
        surrogate_loss_consistency_abs=args.surrogate_loss_consistency_abs,
        loss_weight=args.loss_weight,
        voltage_weight=args.voltage_weight,
        lambda_q=args.lambda_q,
        voltage_margin=args.voltage_margin,
        current_margin=args.current_margin,
        physical_output_bounds=not args.no_physical_output_bounds,
        trust_region_fraction=args.trust_region_fraction,
        control_deviation_penalty=args.control_deviation_penalty,
        base_kv=args.base_kv,
        base_mva=args.base_mva,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        dual_reductions=args.dual_reductions,
        diagnose_slack=args.diagnose_slack,
        slack_penalty=args.slack_penalty,
        output_flag=0 if args.quiet else 1,
        export_lp=args.export_lp,
        iis_path=args.iis,
        x_base=x_base,
        pv_q_base=pv_q_base,
        ess_nodes=ess_nodes,
        ess_s_rated=ess_s_rated,
        ess_p_base=ess_p_base,
        ess_q_base=ess_q_base,
        q_device_nodes=q_device_nodes,
        q_device_min=q_device_min,
        q_device_max=q_device_max,
        q_device_q_base=q_device_q_base,
        q_device_names=q_device_names,
    )


if __name__ == "__main__":
    main()



