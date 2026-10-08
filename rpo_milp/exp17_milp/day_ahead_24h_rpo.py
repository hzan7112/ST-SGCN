"""Day-ahead 24-hour reactive power optimization with the Exp17 SGCN-MILP.

The model uses the nominal 24-hour profiles saved by
``data/generate_ieee33_nodal_pq_pool_50k.py``:

* Pload_24h, Qload_24h: fixed hourly nodal loads.
* Ppv_24h: fixed hourly PV active output.
* PV Q capability: |Q_pv[t,k]| <= sqrt(S_pv[k]^2 - Ppv_24h[t,k]^2).
* ESS active power is optimized across the full horizon with SOC continuity.
* ESS inverter capability couples active and reactive power:
  (P_dis[t,k] - P_ch[t,k])^2 + Q_ess[t,k]^2 <= S_ess[k]^2.

Each hour has its own Exp17 surrogate block and safety constraints. The
objective is the 24-hour sum of learned cumulative voltage deviation, with
optional ESS throughput and Q-movement penalties.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rpo_milp.exp17_milp import single_step_rpo as exp17_rpo
from rpo_milp.disflow import day_ahead_24h_distflow_rpo as disflow_24h


HORIZON = 24


@dataclass
class DayAheadData:
    p_load: np.ndarray
    q_load: np.ndarray
    p_pv: np.ndarray
    pv_nodes: np.ndarray
    s_pv: np.ndarray
    ess_nodes: np.ndarray
    s_ess: np.ndarray
    q_device_nodes: np.ndarray
    q_device_min: np.ndarray
    q_device_max: np.ndarray
    q_device_names: list[str]
    branches: np.ndarray
    topo_mask: np.ndarray
    base_config: dict
    data_path: Path


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def safe_torch_load(path: str | Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_numpy(value, *, dtype=float) -> np.ndarray:
    if value is None:
        return np.array([], dtype=dtype)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def load_day_ahead_data(data_path: str | Path) -> DayAheadData:
    path = resolve_repo_path(data_path)
    data = safe_torch_load(path)
    if not isinstance(data, dict):
        raise TypeError(f"dataset must be a dict: {path}")

    required = [
        "Pload_24h",
        "Qload_24h",
        "Ppv_24h",
        "pv_nodes",
        "S_pv_mva",
        "ess_nodes",
        "S_ess_mva",
        "q_device_nodes",
        "q_device_min",
        "q_device_max",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"dataset missing required 24h keys: {missing}")

    p_load = to_numpy(data["Pload_24h"]).astype(float)
    q_load = to_numpy(data["Qload_24h"]).astype(float)
    p_pv = to_numpy(data["Ppv_24h"]).astype(float)
    if p_load.shape != (33, HORIZON):
        raise ValueError(f"Pload_24h must have shape (33, 24), got {p_load.shape}")
    if q_load.shape != (33, HORIZON):
        raise ValueError(f"Qload_24h must have shape (33, 24), got {q_load.shape}")
    if p_pv.ndim != 2 or p_pv.shape[0] != HORIZON:
        raise ValueError(f"Ppv_24h must have shape (24, n_pv), got {p_pv.shape}")

    pv_nodes = to_numpy(data["pv_nodes"], dtype=int).reshape(-1)
    s_pv = to_numpy(data["S_pv_mva"]).reshape(-1)
    if p_pv.shape[1] != pv_nodes.size or s_pv.size != pv_nodes.size:
        raise ValueError("PV node, rating, and Ppv_24h dimensions do not match")

    ess_nodes = to_numpy(data["ess_nodes"], dtype=int).reshape(-1)
    s_ess = to_numpy(data["S_ess_mva"]).reshape(-1)
    if s_ess.size != ess_nodes.size:
        raise ValueError("ESS node and rating dimensions do not match")

    q_device_nodes = to_numpy(data["q_device_nodes"], dtype=int).reshape(-1)
    q_device_min = to_numpy(data["q_device_min"]).reshape(-1)
    q_device_max = to_numpy(data["q_device_max"]).reshape(-1)
    if not (q_device_nodes.size == q_device_min.size == q_device_max.size):
        raise ValueError("Q-device node/min/max dimensions do not match")
    q_device_names = [str(name) for name in data.get("q_device_names", [])]
    branches = to_numpy(data.get("branch_full")) if "branch_full" in data else disflow_24h.RADIAL_BRANCHES

    if np.any(p_pv < -1e-12):
        raise ValueError("Ppv_24h cannot contain negative active power")
    if np.any(p_pv > s_pv.reshape(1, -1) + 1e-9):
        raise ValueError("Ppv_24h exceeds PV apparent-power ratings")

    return DayAheadData(
        p_load=p_load,
        q_load=q_load,
        p_pv=p_pv,
        pv_nodes=pv_nodes,
        s_pv=s_pv,
        ess_nodes=ess_nodes,
        s_ess=s_ess,
        q_device_nodes=q_device_nodes,
        q_device_min=q_device_min,
        q_device_max=q_device_max,
        q_device_names=q_device_names,
        branches=np.asarray(branches, dtype=float),
        topo_mask=exp17_rpo.get_standard_radial_topology(),
        base_config=dict(data.get("base_config", {})),
        data_path=path,
    )


def q_capacity(p_active: np.ndarray, s_rated: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.asarray(s_rated) ** 2 - np.asarray(p_active) ** 2, 0.0))


def build_zero_q_base_injection(day: DayAheadData, hour: int) -> np.ndarray:
    """Build the fixed hourly net injection used as the Exp17 control baseline.

    ``single_step_rpo.run_single_step_rpo`` only routes ESS and Q-device
    controls into the surrogate input when ``x_base`` is supplied. The day-ahead
    hourly model therefore uses this zero-reactive-control operating point as
    the base and passes zero Q references for every controllable device.
    """
    x_base = np.zeros((33, 2), dtype=float)
    x_base[:, 0] = -day.p_load[:, hour]
    x_base[:, 1] = -day.q_load[:, hour]
    for idx, bus in enumerate(day.pv_nodes):
        x_base[int(bus), 0] += day.p_pv[hour, idx]
    return x_base


def exact_pf_option(args, name: str, fallback: float) -> float:
    value = getattr(args, name, None)
    if value is None:
        return float(fallback)
    return float(value)


def distflow_result_to_dict(metrics) -> dict:
    return {
        "converged": bool(metrics.converged),
        "iterations": int(metrics.iterations),
        "voltage_pu": np.asarray(metrics.voltage_pu, dtype=float),
        "branch_p_mw": np.asarray(metrics.branch_p_mw, dtype=float),
        "branch_q_mvar": np.asarray(metrics.branch_q_mvar, dtype=float),
        "branch_current_ka": np.asarray(metrics.branch_current_ka, dtype=float),
        "branch_loss_mw": np.asarray(metrics.branch_loss_mw, dtype=float),
        "total_loss_mw": float(metrics.total_loss_mw),
        "Vdev_total": float(metrics.vdev_total),
        "Vworst": float(metrics.vworst),
        "WorstI": float(metrics.worst_i_margin),
    }


def exact_pf_outputs_from_metrics(metrics) -> list[dict]:
    return [
        {
            "Vdev_total": float(item.vdev_total),
            "Vworst": float(item.vworst),
            "WorstI": float(item.worst_i_margin),
            "Ploss_total": float(item.total_loss_mw),
            "V_nodes": np.asarray(item.voltage_pu, dtype=float),
        }
        for item in metrics
    ]


def build_exact_pf_dispatch(day: DayAheadData, result: dict) -> disflow_24h.Dispatch | None:
    if result.get("Q_pv_opt") is None or result.get("Q_ess_opt") is None or result.get("Q_device_opt") is None:
        return None

    q_pv = np.asarray(result["Q_pv_opt"], dtype=float).reshape(HORIZON, day.pv_nodes.size)
    q_ess = np.asarray(result["Q_ess_opt"], dtype=float).reshape(HORIZON, day.ess_nodes.size)
    q_device = np.asarray(result["Q_device_opt"], dtype=float).reshape(HORIZON, day.q_device_nodes.size)
    p_ch = (
        np.asarray(result["P_ess_ch_opt"], dtype=float).reshape(HORIZON, day.ess_nodes.size)
        if result.get("P_ess_ch_opt") is not None
        else np.zeros((HORIZON, day.ess_nodes.size), dtype=float)
    )
    p_dis = (
        np.asarray(result["P_ess_dis_opt"], dtype=float).reshape(HORIZON, day.ess_nodes.size)
        if result.get("P_ess_dis_opt") is not None
        else np.zeros((HORIZON, day.ess_nodes.size), dtype=float)
    )
    return disflow_24h.Dispatch(q_pv=q_pv, p_ch=p_ch, p_dis=p_dis, q_ess=q_ess, q_device=q_device)


def attach_exact_power_flow_results(day: DayAheadData, result: dict, args) -> dict:
    dispatch = build_exact_pf_dispatch(day, result)
    if dispatch is None:
        result["exact_pf"] = {
            "available": False,
            "reason": "No complete Q/P dispatch was returned by the optimizer.",
        }
        return result

    base_config = day.base_config
    distflow = disflow_24h.ExactDistFlow(
        branches=day.branches,
        slack_vm_pu=exact_pf_option(args, "exact_slack_vm_pu", base_config.get("slack_vm_pu", 1.03)),
        base_kv=exact_pf_option(args, "exact_base_kv", base_config.get("base_kv", 12.66)),
        base_mva=exact_pf_option(args, "exact_base_mva", base_config.get("base_mva", 1.0)),
        line_max_i_ka=exact_pf_option(args, "exact_line_max_i_ka", base_config.get("line_max_i_ka", 0.20)),
    )
    metrics = disflow_24h.evaluate_24h(
        day,
        distflow,
        dispatch,
        pf_max_iter=int(getattr(args, "exact_pf_max_iter", 200)),
    )
    exact_outputs = exact_pf_outputs_from_metrics(metrics)
    exact_metrics = [distflow_result_to_dict(item) for item in metrics]
    exact_objective = float(sum(item.vdev_total for item in metrics))
    exact_loss = float(sum(item.total_loss_mw for item in metrics))
    max_vworst = float(max(item.vworst for item in metrics))
    max_worst_i = float(max(item.worst_i_margin for item in metrics))
    all_converged = all(bool(item.converged) for item in metrics)

    result["exact_pf"] = {
        "available": True,
        "voltage_source": "exact DistFlow power-flow evaluation of Exp17 optimized dispatch",
        "objective_vdev_total_24h": exact_objective,
        "total_loss_mw_24h": exact_loss,
        "max_Vworst": max_vworst,
        "max_WorstI": max_worst_i,
        "all_converged": bool(all_converged),
        "slack_vm_pu": float(distflow.slack_vm_pu),
        "base_kv": float(distflow.base_kv),
        "base_mva": float(distflow.base_mva),
        "line_max_i_ka": float(distflow.line_max_i_ka),
    }
    result["surrogate_objective"] = result.get("objective")
    result["objective"] = exact_objective
    result["objective_source"] = "exact_pf_vdev_total_24h"
    result["surrogate_outputs"] = result.get("outputs")
    result["outputs"] = exact_outputs
    result["exact_pf_outputs"] = exact_outputs
    result["exact_pf_metrics"] = exact_metrics
    if isinstance(result.get("hourly_results"), list) and len(result["hourly_results"]) == HORIZON:
        for hour, row in enumerate(result["hourly_results"]):
            row["exact_pf_Vdev_total"] = exact_outputs[hour]["Vdev_total"]
            row["exact_pf_Vworst"] = exact_outputs[hour]["Vworst"]
            row["exact_pf_WorstI"] = exact_outputs[hour]["WorstI"]
            row["exact_pf_Ploss_total"] = exact_outputs[hour]["Ploss_total"]
    return result


def control_labels(day: DayAheadData) -> list[str]:
    labels = [f"PV@Bus{int(bus) + 1}" for bus in day.pv_nodes]
    labels.extend(f"ESS@Bus{int(bus) + 1}" for bus in day.ess_nodes)
    for idx, bus in enumerate(day.q_device_nodes):
        name = day.q_device_names[idx] if idx < len(day.q_device_names) else f"QDev{idx + 1}"
        labels.append(f"{name}@Bus{int(bus) + 1}")
    return labels


def add_abs_penalty(model, expr, name: str, weight: float):
    gp_mod, _ = exp17_rpo.require_gurobi()
    if weight <= 0.0:
        return gp_mod.LinExpr(0.0), None
    aux = model.addVar(lb=0.0, name=name)
    model.addConstr(aux >= expr, name=f"{name}_pos")
    model.addConstr(aux >= -expr, name=f"{name}_neg")
    return float(weight) * aux, aux


def build_hourly_injections(
    model,
    day: DayAheadData,
    *,
    hour: int,
    q_pv,
    p_ch,
    p_dis,
    q_ess,
    q_device,
):
    gp_mod, _ = exp17_rpo.require_gurobi()
    pv_to_idx = {int(bus): idx for idx, bus in enumerate(day.pv_nodes)}
    ess_to_idx = {int(bus): idx for idx, bus in enumerate(day.ess_nodes)}
    qdev_to_idx = {int(bus): idx for idx, bus in enumerate(day.q_device_nodes)}

    x_vars = [[None, None] for _ in range(33)]
    for bus in range(33):
        p_expr = gp_mod.LinExpr(-float(day.p_load[bus, hour]))
        q_expr = gp_mod.LinExpr(-float(day.q_load[bus, hour]))
        if bus in pv_to_idx:
            idx = pv_to_idx[bus]
            p_expr += float(day.p_pv[hour, idx])
            q_expr += q_pv[hour, idx]
        if bus in ess_to_idx:
            idx = ess_to_idx[bus]
            p_expr += p_dis[hour, idx] - p_ch[hour, idx]
            q_expr += q_ess[hour, idx]
        if bus in qdev_to_idx:
            idx = qdev_to_idx[bus]
            q_expr += q_device[hour, idx]
        x_vars[bus][0] = p_expr
        x_vars[bus][1] = q_expr
    return x_vars


def add_hourly_device_constraints(
    model,
    day: DayAheadData,
    *,
    q_pv,
    p_ch,
    p_dis,
    q_ess,
    q_device,
    ess_p_max_fraction: float,
    use_charge_binary: bool,
) -> dict:
    _, grb_mod = exp17_rpo.require_gurobi()
    pv_caps = q_capacity(day.p_pv, day.s_pv.reshape(1, -1))
    for t in range(HORIZON):
        for k, bus in enumerate(day.pv_nodes):
            cap = float(pv_caps[t, k])
            q_pv[t, k].LB = -cap
            q_pv[t, k].UB = cap
            model.addQConstr(
                q_pv[t, k] * q_pv[t, k]
                <= max(float(day.s_pv[k]) ** 2 - float(day.p_pv[t, k]) ** 2, 0.0),
                name=f"PV_Cap_t{t:02d}_bus{int(bus)}",
            )

    ess_p_max = float(ess_p_max_fraction) * day.s_ess
    if use_charge_binary and day.ess_nodes.size:
        mode = model.addVars(HORIZON, day.ess_nodes.size, vtype=grb_mod.BINARY, name="ESS_discharge_mode")
    else:
        mode = None

    for t in range(HORIZON):
        for k, bus in enumerate(day.ess_nodes):
            pmax = float(ess_p_max[k])
            p_ch[t, k].LB = 0.0
            p_ch[t, k].UB = pmax
            p_dis[t, k].LB = 0.0
            p_dis[t, k].UB = pmax
            q_ess[t, k].LB = -float(day.s_ess[k])
            q_ess[t, k].UB = float(day.s_ess[k])
            if mode is not None:
                mode[t, k].Start = 0.0
                model.addConstr(p_ch[t, k] <= pmax * (1.0 - mode[t, k]), name=f"ESS_ch_mode_t{t:02d}_bus{int(bus)}")
                model.addConstr(p_dis[t, k] <= pmax * mode[t, k], name=f"ESS_dis_mode_t{t:02d}_bus{int(bus)}")
            p_net = p_dis[t, k] - p_ch[t, k]
            model.addQConstr(
                p_net * p_net + q_ess[t, k] * q_ess[t, k] <= float(day.s_ess[k]) ** 2,
                name=f"ESS_Inverter_Cap_t{t:02d}_bus{int(bus)}",
            )

    for t in range(HORIZON):
        for k, bus in enumerate(day.q_device_nodes):
            q_device[t, k].LB = float(day.q_device_min[k])
            q_device[t, k].UB = float(day.q_device_max[k])

    return {
        "pv_q_caps": pv_caps,
        "ess_p_max": ess_p_max,
        "ess_charge_binary": bool(use_charge_binary),
    }


def set_neutral_start(day: DayAheadData, *, q_pv, p_ch, p_dis, q_ess, soc, q_device, args) -> None:
    """Seed a physically neutral schedule so Gurobi has a first incumbent target."""
    n_pv = day.pv_nodes.size
    n_ess = day.ess_nodes.size
    n_qdev = day.q_device_nodes.size
    q_mid = 0.5 * (day.q_device_min + day.q_device_max) if n_qdev else np.array([], dtype=float)

    for t in range(HORIZON):
        for k in range(n_pv):
            q_pv[t, k].Start = 0.0
        for k in range(n_ess):
            p_ch[t, k].Start = 0.0
            p_dis[t, k].Start = 0.0
            q_ess[t, k].Start = 0.0
        for k in range(n_qdev):
            q_device[t, k].Start = float(q_mid[k])
    for t in range(HORIZON + 1):
        for k in range(n_ess):
            soc[t, k].Start = float(args.soc_initial)


def load_hourly_warm_start(path: str | Path | None) -> dict[int, np.ndarray]:
    if path is None:
        return {}
    start_path = resolve_repo_path(path)
    with start_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    if not isinstance(rows, list):
        raise TypeError(f"hourly warm-start JSON must contain a rows list: {start_path}")

    starts: dict[int, np.ndarray] = {}
    for row in rows:
        if not isinstance(row, dict) or "hour" not in row:
            continue
        q = row.get("Q_control_opt")
        if q is None:
            continue
        starts[int(row["hour"])] = np.asarray(q, dtype=float).reshape(-1)
    if not starts:
        raise ValueError(f"No Q_control_opt entries found in hourly warm-start JSON: {start_path}")
    return starts


def set_hourly_q_warm_start(
    day: DayAheadData,
    starts: dict[int, np.ndarray],
    *,
    q_pv,
    q_ess,
    q_device,
) -> dict:
    n_pv = day.pv_nodes.size
    n_ess = day.ess_nodes.size
    n_qdev = day.q_device_nodes.size
    expected = n_pv + n_ess + n_qdev
    used_hours = []

    for hour, q_control in sorted(starts.items()):
        if hour < 0 or hour >= HORIZON:
            continue
        if q_control.size != expected:
            raise ValueError(
                f"Warm-start hour {hour} Q_control_opt must have length {expected}, got {q_control.size}"
            )
        q_pv_start = q_control[:n_pv]
        q_ess_start = q_control[n_pv : n_pv + n_ess]
        q_device_start = q_control[n_pv + n_ess :]
        pv_caps = q_capacity(day.p_pv[hour], day.s_pv)
        for k in range(n_pv):
            q_pv[hour, k].Start = float(np.clip(q_pv_start[k], -pv_caps[k], pv_caps[k]))
        for k in range(n_ess):
            q_ess[hour, k].Start = float(np.clip(q_ess_start[k], -day.s_ess[k], day.s_ess[k]))
        for k in range(n_qdev):
            q_device[hour, k].Start = float(np.clip(q_device_start[k], day.q_device_min[k], day.q_device_max[k]))
        used_hours.append(int(hour))

    return {
        "source_hours": used_hours,
        "hour_count": len(used_hours),
    }


def add_soc_constraints(
    model,
    day: DayAheadData,
    *,
    p_ch,
    p_dis,
    soc,
    energy_mwh,
    soc_initial,
    soc_min,
    soc_max,
    eta_charge,
    eta_discharge,
    terminal_equal_initial,
    dt_hours,
) -> dict:
    if day.ess_nodes.size == 0:
        return {
            "energy_mwh": [],
            "soc_initial": [],
            "soc_min": [],
            "soc_max": [],
            "terminal_equal_initial": bool(terminal_equal_initial),
        }

    energy = np.asarray(energy_mwh, dtype=float).reshape(-1)
    if energy.size != day.ess_nodes.size:
        raise ValueError(f"energy_mwh must have length {day.ess_nodes.size}, got {energy.size}")
    if np.any(energy <= 0.0):
        raise ValueError("ESS energy capacity must be positive")

    soc0 = np.broadcast_to(np.asarray(soc_initial, dtype=float).reshape(-1), (day.ess_nodes.size,))
    soc_lo = np.broadcast_to(np.asarray(soc_min, dtype=float).reshape(-1), (day.ess_nodes.size,))
    soc_hi = np.broadcast_to(np.asarray(soc_max, dtype=float).reshape(-1), (day.ess_nodes.size,))

    for k, bus in enumerate(day.ess_nodes):
        if not (0.0 <= soc_lo[k] <= soc0[k] <= soc_hi[k] <= 1.0):
            raise ValueError(f"Invalid SOC bounds/initial value for ESS at bus {int(bus)}")
        for t in range(HORIZON + 1):
            soc[t, k].LB = float(soc_lo[k])
            soc[t, k].UB = float(soc_hi[k])
        model.addConstr(soc[0, k] == float(soc0[k]), name=f"ESS_SOC_initial_bus{int(bus)}")
        for t in range(HORIZON):
            delta = (
                float(dt_hours)
                * (float(eta_charge) * p_ch[t, k] - p_dis[t, k] / float(eta_discharge))
                / float(energy[k])
            )
            model.addConstr(soc[t + 1, k] == soc[t, k] + delta, name=f"ESS_SOC_dyn_t{t:02d}_bus{int(bus)}")
        if terminal_equal_initial:
            model.addConstr(soc[HORIZON, k] == float(soc0[k]), name=f"ESS_SOC_terminal_bus{int(bus)}")

    return {
        "energy_mwh": energy.tolist(),
        "soc_initial": soc0.tolist(),
        "soc_min": soc_lo.tolist(),
        "soc_max": soc_hi.tolist(),
        "terminal_equal_initial": bool(terminal_equal_initial),
        "eta_charge": float(eta_charge),
        "eta_discharge": float(eta_discharge),
        "dt_hours": float(dt_hours),
    }


def optimize_24h_hourly_decomposed(day: DayAheadData, args) -> dict:
    """Solve 24 independent Exp17 single-hour MILPs and assemble a 24h schedule.

    The monolithic 24h MILP is useful as a research formulation, but it is very
    hard for Gurobi because it contains 24 embedded ReLU networks. This default
    path keeps the same hourly surrogate constraints and the same PV/Q-device
    limits, fixes ESS active power to zero, and therefore keeps SOC continuous
    and constant over the horizon. It is the reliable day-ahead surrogate
    dispatch path when the goal is to obtain a usable 24h schedule.
    """
    n_pv = day.pv_nodes.size
    n_ess = day.ess_nodes.size
    n_qdev = day.q_device_nodes.size

    q_pv_all = np.zeros((HORIZON, n_pv), dtype=float)
    p_ch_all = np.zeros((HORIZON, n_ess), dtype=float)
    p_dis_all = np.zeros((HORIZON, n_ess), dtype=float)
    q_ess_all = np.zeros((HORIZON, n_ess), dtype=float)
    q_device_all = np.zeros((HORIZON, n_qdev), dtype=float)
    soc_all = np.full((HORIZON + 1, n_ess), float(args.soc_initial), dtype=float)
    outputs_all = []
    hourly_rows = []
    objective = 0.0
    solve_time = 0.0
    all_feasible = True
    binary_count = None

    for hour in range(HORIZON):
        print(f"\nSolving decomposed Exp17 hour {hour:02d} / 23...", flush=True)
        start = time.perf_counter()
        hour_result = exp17_rpo.run_single_step_rpo(
            day.p_load[:, hour],
            day.q_load[:, hour],
            day.p_pv[hour],
            day.topo_mask,
            pv_nodes=day.pv_nodes,
            s_rated=day.s_pv,
            engine_path=args.engine,
            relu_formulation=args.relu_formulation,
            big_m_scale=args.big_m_scale,
            fallback_big_m=args.fallback_big_m,
            objective="vdev",
            loss_objective_model="none",
            voltage_margin=args.voltage_margin,
            current_margin=args.current_margin,
            physical_output_bounds=not args.no_physical_output_bounds,
            trust_region_fraction=1.0,
            control_deviation_penalty=args.q_movement_penalty,
            time_limit=args.hourly_time_limit,
            mip_gap=args.hourly_mip_gap,
            dual_reductions=args.dual_reductions,
            diagnose_slack=args.diagnose_slack,
            slack_penalty=args.slack_penalty,
            output_flag=0 if args.quiet else 1,
            x_base=build_zero_q_base_injection(day, hour),
            pv_q_base=np.zeros(n_pv, dtype=float),
            ess_nodes=day.ess_nodes,
            ess_s_rated=day.s_ess,
            ess_p_base=np.zeros(n_ess, dtype=float),
            ess_q_base=np.zeros(n_ess, dtype=float),
            q_device_nodes=day.q_device_nodes,
            q_device_min=day.q_device_min,
            q_device_max=day.q_device_max,
            q_device_q_base=np.zeros(n_qdev, dtype=float),
            q_device_names=day.q_device_names,
        )
        elapsed = time.perf_counter() - start
        solve_time += elapsed
        if binary_count is None:
            binary_count = hour_result.get("binary_count_estimate")

        if hour_result.get("Q_control_opt") is None:
            all_feasible = False
        else:
            q_control = np.asarray(hour_result["Q_control_opt"], dtype=float).reshape(-1)
            q_pv_all[hour] = q_control[:n_pv]
            q_ess_all[hour] = q_control[n_pv : n_pv + n_ess]
            q_device_all[hour] = q_control[n_pv + n_ess :]
            if hour_result.get("objective") is not None:
                objective += float(hour_result["objective"])

        outputs = hour_result.get("outputs") or {}
        outputs_all.append(
            {
                "Vdev_total": outputs.get("Vdev_total"),
                "Vworst": outputs.get("Vworst"),
                "WorstI": outputs.get("WorstI"),
                "V_nodes": outputs.get("V_nodes"),
            }
        )
        hourly_rows.append(
            {
                "hour": hour,
                "status": hour_result.get("status"),
                "status_name": hour_result.get("status_name"),
                "sol_count": hour_result.get("sol_count"),
                "objective": hour_result.get("objective"),
                "solve_time_sec": elapsed,
                "Vdev_total": outputs.get("Vdev_total"),
                "Vworst": outputs.get("Vworst"),
                "WorstI": outputs.get("WorstI"),
                "safety_slack": hour_result.get("safety_slack"),
            }
        )

    return {
        "status": 2 if all_feasible else 9,
        "status_name": "HOURLY_DECOMPOSED_OPTIMAL" if all_feasible else "HOURLY_DECOMPOSED_PARTIAL",
        "objective": float(objective) if all_feasible else None,
        "solve_time_sec": float(solve_time),
        "data": str(day.data_path),
        "horizon": HORIZON,
        "solve_mode": "hourly_decomposed",
        "device_meta": {
            "pv_q_caps": q_capacity(day.p_pv, day.s_pv.reshape(1, -1)),
            "ess_p_max": np.zeros(n_ess, dtype=float),
            "ess_charge_binary": False,
        },
        "soc_meta": {
            "energy_mwh": (args.ess_energy_hours * day.s_ess).tolist(),
            "soc_initial": [float(args.soc_initial)] * n_ess,
            "soc_min": [float(args.soc_min)] * n_ess,
            "soc_max": [float(args.soc_max)] * n_ess,
            "terminal_equal_initial": True,
            "dt_hours": float(args.dt_hours),
            "note": "ESS active power is fixed to zero in hourly_decomposed mode, so SOC is continuous and constant.",
        },
        "control_labels": control_labels(day),
        "pv_nodes": day.pv_nodes.tolist(),
        "ess_nodes": day.ess_nodes.tolist(),
        "q_device_nodes": day.q_device_nodes.tolist(),
        "q_device_names": day.q_device_names,
        "Pload_sum_24h": day.p_load.sum(axis=0).tolist(),
        "Qload_sum_24h": day.q_load.sum(axis=0).tolist(),
        "Ppv_24h": day.p_pv.tolist(),
        "binary_count_estimate_per_hour": binary_count,
        "binary_count_estimate_total_surrogate": None if binary_count is None else int(HORIZON * int(binary_count)),
        "sol_count": 1 if all_feasible else 0,
        "Q_pv_opt": q_pv_all,
        "P_ess_ch_opt": p_ch_all,
        "P_ess_dis_opt": p_dis_all,
        "P_ess_net_opt": p_dis_all - p_ch_all,
        "Q_ess_opt": q_ess_all,
        "ESS_SOC": soc_all,
        "Q_device_opt": q_device_all,
        "outputs": outputs_all,
        "safety_slack": [row["safety_slack"] for row in hourly_rows],
        "hourly_results": hourly_rows,
        "ess_throughput_mwh": 0.0,
    }


def optimize_24h(day: DayAheadData, args) -> dict:
    gp_mod, grb_mod = exp17_rpo.require_gurobi()
    converter_class = exp17_rpo.get_converter_class()

    model = gp_mod.Model("Exp17_24h_DayAhead_RPO")
    model.Params.TimeLimit = float(args.time_limit)
    model.Params.MIPGap = float(args.mip_gap)
    model.Params.OutputFlag = 0 if args.quiet else 1
    model.Params.NonConvex = 2
    model.Params.DualReductions = int(args.dual_reductions)
    model.Params.MIPFocus = int(args.mip_focus)
    model.Params.Heuristics = float(args.heuristics)
    if float(args.no_rel_heur_time) > 0.0:
        model.Params.NoRelHeurTime = float(args.no_rel_heur_time)
    if int(args.threads) > 0:
        model.Params.Threads = int(args.threads)

    converter = converter_class(
        args.engine,
        relu_formulation=args.relu_formulation,
        fallback_big_m=args.fallback_big_m,
        big_m_scale=args.big_m_scale,
    )

    n_pv = day.pv_nodes.size
    n_ess = day.ess_nodes.size
    n_qdev = day.q_device_nodes.size
    q_pv = model.addVars(HORIZON, n_pv, lb=-grb_mod.INFINITY, ub=grb_mod.INFINITY, name="Q_pv")
    p_ch = model.addVars(HORIZON, n_ess, lb=0.0, ub=grb_mod.INFINITY, name="P_ess_ch")
    p_dis = model.addVars(HORIZON, n_ess, lb=0.0, ub=grb_mod.INFINITY, name="P_ess_dis")
    q_ess = model.addVars(HORIZON, n_ess, lb=-grb_mod.INFINITY, ub=grb_mod.INFINITY, name="Q_ess")
    soc = model.addVars(HORIZON + 1, n_ess, lb=0.0, ub=1.0, name="ESS_SOC")
    q_device = model.addVars(HORIZON, n_qdev, lb=-grb_mod.INFINITY, ub=grb_mod.INFINITY, name="Q_device")

    default_p_fraction = float(day.base_config.get("ess_p_max_fraction", args.ess_p_max_fraction))
    if args.fix_ess_active_zero:
        default_p_fraction = 0.0
    if default_p_fraction < 0.0:
        raise ValueError("ESS active-power max fraction must be nonnegative")
    if not (0.0 < args.eta_charge <= 1.0 and 0.0 < args.eta_discharge <= 1.0):
        raise ValueError("ESS charge/discharge efficiency must be in (0, 1]")
    device_meta = add_hourly_device_constraints(
        model,
        day,
        q_pv=q_pv,
        p_ch=p_ch,
        p_dis=p_dis,
        q_ess=q_ess,
        q_device=q_device,
        ess_p_max_fraction=default_p_fraction,
        use_charge_binary=(not args.allow_simultaneous_charge_discharge and not args.fix_ess_active_zero),
    )

    energy = np.asarray(args.ess_energy_hours * day.s_ess, dtype=float)
    soc_meta = add_soc_constraints(
        model,
        day,
        p_ch=p_ch,
        p_dis=p_dis,
        soc=soc,
        energy_mwh=energy,
        soc_initial=np.full(n_ess, args.soc_initial, dtype=float),
        soc_min=np.full(n_ess, args.soc_min, dtype=float),
        soc_max=np.full(n_ess, args.soc_max, dtype=float),
        eta_charge=args.eta_charge,
        eta_discharge=args.eta_discharge,
        terminal_equal_initial=not args.no_terminal_soc,
        dt_hours=args.dt_hours,
    )
    set_neutral_start(
        day,
        q_pv=q_pv,
        p_ch=p_ch,
        p_dis=p_dis,
        q_ess=q_ess,
        soc=soc,
        q_device=q_device,
        args=args,
    )
    hourly_warm_start_meta = None
    if args.hourly_warm_start_json:
        hourly_warm_start = load_hourly_warm_start(args.hourly_warm_start_json)
        hourly_warm_start_meta = set_hourly_q_warm_start(
            day,
            hourly_warm_start,
            q_pv=q_pv,
            q_ess=q_ess,
            q_device=q_device,
        )

    objective = gp_mod.LinExpr(0.0)
    hourly_outputs = []
    safety_slacks = []
    q_abs_aux = []
    for t in range(HORIZON):
        x_vars = build_hourly_injections(
            model,
            day,
            hour=t,
            q_pv=q_pv,
            p_ch=p_ch,
            p_dis=p_dis,
            q_ess=q_ess,
            q_device=q_device,
        )
        outputs = converter.embed_sgcn_constraints(
            model,
            x_vars,
            topo_mask=day.topo_mask,
            name_prefix=f"exp17_24h_t{t:02d}",
        )
        if not args.no_physical_output_bounds:
            model.addConstr(outputs.Vdev_total >= 0.0, name=f"Physical_Vdev_t{t:02d}")
            model.addConstr(outputs.WorstI >= -1.0, name=f"Physical_WorstI_t{t:02d}")
            model.addConstr(outputs.Vworst >= -0.05, name=f"Physical_Vworst_t{t:02d}")

        objective += outputs.Vdev_total
        if args.diagnose_slack:
            s_v = model.addVar(lb=0.0, name=f"slack_Vworst_t{t:02d}")
            s_i = model.addVar(lb=0.0, name=f"slack_WorstI_t{t:02d}")
            model.addConstr(outputs.Vworst <= float(args.voltage_margin) + s_v, name=f"Surrogate_Vworst_safe_t{t:02d}")
            model.addConstr(outputs.WorstI <= float(args.current_margin) + s_i, name=f"Surrogate_WorstI_safe_t{t:02d}")
            objective += float(args.slack_penalty) * (s_v + s_i)
            safety_slacks.append((s_v, s_i))
        else:
            model.addConstr(outputs.Vworst <= float(args.voltage_margin), name=f"Surrogate_Vworst_safe_t{t:02d}")
            model.addConstr(outputs.WorstI <= float(args.current_margin), name=f"Surrogate_WorstI_safe_t{t:02d}")
        hourly_outputs.append(outputs)

        for k in range(n_pv):
            penalty, aux = add_abs_penalty(model, q_pv[t, k], f"Abs_Q_pv_t{t:02d}_{k}", args.q_movement_penalty)
            objective += penalty
            if aux is not None:
                q_abs_aux.append(aux)
        for k in range(n_ess):
            penalty, aux = add_abs_penalty(model, q_ess[t, k], f"Abs_Q_ess_t{t:02d}_{k}", args.q_movement_penalty)
            objective += penalty
            if aux is not None:
                q_abs_aux.append(aux)
        for k in range(n_qdev):
            penalty, aux = add_abs_penalty(model, q_device[t, k], f"Abs_Q_device_t{t:02d}_{k}", args.q_movement_penalty)
            objective += penalty
            if aux is not None:
                q_abs_aux.append(aux)

    throughput = gp_mod.quicksum(p_ch[t, k] + p_dis[t, k] for t in range(HORIZON) for k in range(n_ess))
    objective += float(args.ess_throughput_penalty) * throughput
    model.setObjective(objective, grb_mod.MINIMIZE)

    if args.export_lp:
        export_path = resolve_repo_path(args.export_lp)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        model.write(str(export_path))

    start = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - start

    result = {
        "status": int(model.status),
        "status_name": exp17_rpo.status_name(model.status),
        "objective": None,
        "solve_time_sec": float(solve_time),
        "data": str(day.data_path),
        "horizon": HORIZON,
        "device_meta": device_meta,
        "soc_meta": soc_meta,
        "control_labels": control_labels(day),
        "pv_nodes": day.pv_nodes.tolist(),
        "ess_nodes": day.ess_nodes.tolist(),
        "q_device_nodes": day.q_device_nodes.tolist(),
        "q_device_names": day.q_device_names,
        "Pload_sum_24h": day.p_load.sum(axis=0).tolist(),
        "Qload_sum_24h": day.q_load.sum(axis=0).tolist(),
        "Ppv_24h": day.p_pv.tolist(),
        "binary_count_estimate_per_hour": converter.binary_count,
        "binary_count_estimate_total_surrogate": int(HORIZON * converter.binary_count),
        "sol_count": int(model.SolCount),
        "gurobi_search": {
            "MIPFocus": int(args.mip_focus),
            "Heuristics": float(args.heuristics),
            "NoRelHeurTime": float(args.no_rel_heur_time),
            "Threads": int(args.threads),
            "warm_start": "neutral zero-P/Q ESS and PV-Q, midpoint Q-device, flat initial SOC",
            "hourly_q_warm_start": hourly_warm_start_meta,
        },
        "fix_ess_active_zero": bool(args.fix_ess_active_zero),
    }

    if model.SolCount > 0:
        result.update(
            {
                "objective": float(model.ObjVal),
                "Q_pv_opt": np.array([[q_pv[t, k].X for k in range(n_pv)] for t in range(HORIZON)], dtype=float),
                "P_ess_ch_opt": np.array([[p_ch[t, k].X for k in range(n_ess)] for t in range(HORIZON)], dtype=float),
                "P_ess_dis_opt": np.array([[p_dis[t, k].X for k in range(n_ess)] for t in range(HORIZON)], dtype=float),
                "P_ess_net_opt": np.array([[p_dis[t, k].X - p_ch[t, k].X for k in range(n_ess)] for t in range(HORIZON)], dtype=float),
                "Q_ess_opt": np.array([[q_ess[t, k].X for k in range(n_ess)] for t in range(HORIZON)], dtype=float),
                "ESS_SOC": np.array([[soc[t, k].X for k in range(n_ess)] for t in range(HORIZON + 1)], dtype=float),
                "Q_device_opt": np.array([[q_device[t, k].X for k in range(n_qdev)] for t in range(HORIZON)], dtype=float),
                "outputs": [
                    {
                        "Vdev_total": float(out.Vdev_total.X),
                        "Vworst": float(out.Vworst.X),
                        "WorstI": float(out.WorstI.X),
                        "V_nodes": np.array([var.X for var in out.V_nodes], dtype=float),
                    }
                    for out in hourly_outputs
                ],
                "safety_slack": (
                    [
                        {"Vworst": float(s_v.X), "WorstI": float(s_i.X)}
                        for s_v, s_i in safety_slacks
                    ]
                    if safety_slacks
                    else None
                ),
                "ess_throughput_mwh": float(sum((p_ch[t, k].X + p_dis[t, k].X) * args.dt_hours for t in range(HORIZON) for k in range(n_ess))),
            }
        )
    elif model.status == grb_mod.INFEASIBLE and args.iis:
        iis_path = resolve_repo_path(args.iis)
        iis_path.parent.mkdir(parents=True, exist_ok=True)
        model.computeIIS()
        model.write(str(iis_path))
        result["iis"] = str(iis_path)

    return result


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="24-hour Exp17 SGCN-MILP reactive power optimization.")
    parser.add_argument("--data", default="data/ieee33_nodal_pq_correlated_raw_pool_50k.pt")
    parser.add_argument("--engine", default=str(exp17_rpo.DEFAULT_ENGINE_PATH))
    parser.add_argument("--out", default="rpo_milp/exp17_milp/results/day_ahead_24h_result.json")
    parser.add_argument(
        "--solve-mode",
        choices=["hourly", "monolithic"],
        default="hourly",
        help=(
            "hourly solves 24 independent single-hour Exp17 MILPs and assembles "
            "a 24h schedule; monolithic builds the original full 24h MIP."
        ),
    )
    parser.add_argument(
        "--hourly-time-limit",
        type=float,
        default=120.0,
        help="Time limit for each single-hour MILP in solve-mode=hourly.",
    )
    parser.add_argument(
        "--hourly-mip-gap",
        type=float,
        default=0.02,
        help="MIP gap for each single-hour MILP in solve-mode=hourly.",
    )
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--mip-gap", type=float, default=0.02)
    parser.add_argument("--dual-reductions", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--relu-formulation",
        choices=["big_m", "general"],
        default="general",
        help=(
            "Use Gurobi general MAX constraints by default for the 24h model. "
            "Use big_m to reproduce the explicit binary ReLU formulation."
        ),
    )
    parser.add_argument("--big-m-scale", type=float, default=1.0)
    parser.add_argument("--fallback-big-m", type=float, default=1e3)
    parser.add_argument("--voltage-margin", type=float, default=0.0)
    parser.add_argument("--current-margin", type=float, default=0.0)
    parser.add_argument("--diagnose-slack", action="store_true")
    parser.add_argument("--slack-penalty", type=float, default=1e4)
    parser.add_argument("--no-physical-output-bounds", action="store_true")
    parser.add_argument("--ess-energy-hours", type=float, default=2.0)
    parser.add_argument("--ess-p-max-fraction", type=float, default=0.6)
    parser.add_argument("--soc-initial", type=float, default=0.5)
    parser.add_argument("--soc-min", type=float, default=0.1)
    parser.add_argument("--soc-max", type=float, default=0.9)
    parser.add_argument("--eta-charge", type=float, default=0.95)
    parser.add_argument("--eta-discharge", type=float, default=0.95)
    parser.add_argument("--dt-hours", type=float, default=1.0)
    parser.add_argument("--no-terminal-soc", action="store_true")
    parser.add_argument("--allow-simultaneous-charge-discharge", action="store_true")
    parser.add_argument(
        "--fix-ess-active-zero",
        action="store_true",
        default=True,
        help="Diagnostic mode: keep ESS active charge/discharge at zero while still optimizing Q_ess.",
    )
    parser.add_argument(
        "--optimize-ess-active",
        action="store_false",
        dest="fix_ess_active_zero",
        help="Enable ESS charge/discharge and SOC-coupled active-power optimization in monolithic mode.",
    )
    parser.add_argument(
        "--hourly-warm-start-json",
        default=None,
        help=(
            "JSON written by scan_24h_hourly_feasibility.py. Its hourly "
            "Q_control_opt values are used as a stronger 24h MIP start."
        ),
    )
    parser.add_argument("--ess-throughput-penalty", type=float, default=1e-3)
    parser.add_argument("--q-movement-penalty", type=float, default=0.0)
    parser.add_argument(
        "--exact-pf-max-iter",
        type=int,
        default=200,
        help="Maximum iterations for exact DistFlow power-flow evaluation after Exp17 optimization.",
    )
    parser.add_argument(
        "--exact-slack-vm-pu",
        type=float,
        default=None,
        help="Slack voltage used by exact DistFlow post-evaluation; defaults to dataset base_config or 1.03.",
    )
    parser.add_argument(
        "--exact-base-kv",
        type=float,
        default=None,
        help="Base kV used by exact DistFlow post-evaluation; defaults to dataset base_config or 12.66.",
    )
    parser.add_argument(
        "--exact-base-mva",
        type=float,
        default=None,
        help="Base MVA used by exact DistFlow post-evaluation; defaults to dataset base_config or 1.0.",
    )
    parser.add_argument(
        "--exact-line-max-i-ka",
        type=float,
        default=None,
        help="Line current limit used by exact DistFlow post-evaluation; defaults to dataset base_config or 0.20.",
    )
    parser.add_argument(
        "--mip-focus",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Gurobi MIPFocus; 1 emphasizes finding feasible incumbents.",
    )
    parser.add_argument("--heuristics", type=float, default=0.5)
    parser.add_argument(
        "--no-rel-heur-time",
        type=float,
        default=120.0,
        help="Seconds spent in Gurobi NoRel heuristic before the root relaxation.",
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--export-lp", default=None)
    parser.add_argument("--iis", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    day = load_day_ahead_data(args.data)
    print("Loaded 24h profile data:")
    print(f"  data: {day.data_path}")
    print(f"  PV buses: {day.pv_nodes.tolist()}")
    print(f"  ESS buses: {day.ess_nodes.tolist()}")
    print(f"  Q devices: {control_labels(day)[day.pv_nodes.size + day.ess_nodes.size:]}")
    if args.solve_mode == "hourly":
        result = optimize_24h_hourly_decomposed(day, args)
    else:
        result = optimize_24h(day, args)
    result = attach_exact_power_flow_results(day, result, args)

    out_path = resolve_repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(result), f, indent=2)

    print()
    print("24h Exp17 SGCN-MILP RPO complete.")
    print(f"Status: {result['status_name']} ({result['status']})")
    print(f"Solve time: {result['solve_time_sec']:.3f} s")
    if result.get("surrogate_objective") is not None:
        print(f"Surrogate objective: {result['surrogate_objective']:.8f}")
        surrogate_outputs = result.get("surrogate_outputs") or []
        if surrogate_outputs:
            print(f"Max surrogate Vworst: {max(x['Vworst'] for x in surrogate_outputs):.8f}")
            print(f"Max surrogate WorstI: {max(x['WorstI'] for x in surrogate_outputs):.8f}")
    exact_pf = result.get("exact_pf") or {}
    if exact_pf.get("available"):
        print(f"Objective: {result['objective']:.8f} ({result['objective_source']})")
        print(f"Exact PF 24h Vdev_total: {exact_pf['objective_vdev_total_24h']:.8f}")
        print(f"Exact PF 24h total loss: {exact_pf['total_loss_mw_24h']:.8f} MW")
        print(f"Max exact PF Vworst: {exact_pf['max_Vworst']:.8f}")
        print(f"Max exact PF WorstI: {exact_pf['max_WorstI']:.8f}")
    print(f"Result JSON: {out_path}")


if __name__ == "__main__":
    main()
