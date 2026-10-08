"""Exact DistFlow 24-hour continuous reactive power optimization.

This is a physics-based day-ahead baseline for the same 24-hour profiles used
by ``rpo_milp/exp17_milp/day_ahead_24h_rpo.py``. It follows the default Exp17
day-ahead setting: ESS active charge/discharge is fixed to zero, and only
reactive controls are optimized over the full horizon.

Important time-coupled constraints:

* PV Q limits are recomputed every hour from the fixed PV active-power profile.
* ESS active power is fixed: P_ch = P_dis = 0.
* ESS SOC is therefore continuous and constant over the horizon.
* ESS inverter capability reduces to |Q_ess[t,k]| <= S_ess[k].
"""

from __future__ import annotations

import argparse
import csv
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

from rpo_milp.disflow.single_time_opf.compare_exp17_single_step import ExactDistFlow, RADIAL_BRANCHES


HORIZON = 24

gp = None
GRB = None


def get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def require_gurobi():
    global gp, GRB
    if gp is not None and GRB is not None:
        return gp, GRB
    try:
        import gurobipy as gp_mod
        from gurobipy import GRB as grb_mod
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gurobipy is required for the DistFlow Gurobi QCP optimizer."
        ) from exc
    gp = gp_mod
    GRB = grb_mod
    return gp, GRB


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
    base_config: dict
    data_path: Path


@dataclass
class StorageParams:
    energy_mwh: np.ndarray
    p_max: np.ndarray
    soc_initial: np.ndarray
    soc_min: np.ndarray
    soc_max: np.ndarray
    eta_charge: float
    eta_discharge: float
    dt_hours: float
    terminal_equal_initial: bool


@dataclass
class Dispatch:
    q_pv: np.ndarray
    p_ch: np.ndarray
    p_dis: np.ndarray
    q_ess: np.ndarray
    q_device: np.ndarray


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
    ess_nodes = to_numpy(data["ess_nodes"], dtype=int).reshape(-1)
    s_ess = to_numpy(data["S_ess_mva"]).reshape(-1)
    q_device_nodes = to_numpy(data["q_device_nodes"], dtype=int).reshape(-1)
    q_device_min = to_numpy(data["q_device_min"]).reshape(-1)
    q_device_max = to_numpy(data["q_device_max"]).reshape(-1)
    if p_pv.shape[1] != pv_nodes.size or s_pv.size != pv_nodes.size:
        raise ValueError("PV node, rating, and Ppv_24h dimensions do not match")
    if ess_nodes.size != s_ess.size:
        raise ValueError("ESS node and rating dimensions do not match")
    if not (q_device_nodes.size == q_device_min.size == q_device_max.size):
        raise ValueError("Q-device node/min/max dimensions do not match")
    if np.any(p_pv < -1e-12):
        raise ValueError("Ppv_24h cannot contain negative active power")
    if np.any(p_pv > s_pv.reshape(1, -1) + 1e-9):
        raise ValueError("Ppv_24h exceeds PV apparent-power ratings")

    branches = to_numpy(data.get("branch_full")) if "branch_full" in data else RADIAL_BRANCHES
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
        q_device_names=[str(name) for name in data.get("q_device_names", [])],
        branches=np.asarray(branches, dtype=float),
        base_config=dict(data.get("base_config", {})),
        data_path=path,
    )


def make_storage_params(day: DayAheadData, args) -> StorageParams:
    n_ess = day.ess_nodes.size
    p_fraction = 0.0
    energy = np.asarray(args.ess_energy_hours * day.s_ess, dtype=float)
    p_max = p_fraction * day.s_ess
    if n_ess and np.any(energy <= 0.0):
        raise ValueError("ESS energy capacity must be positive")
    if not (0.0 < args.eta_charge <= 1.0 and 0.0 < args.eta_discharge <= 1.0):
        raise ValueError("ESS charge/discharge efficiency must be in (0, 1]")
    if not (0.0 <= args.soc_min <= args.soc_initial <= args.soc_max <= 1.0):
        raise ValueError("SOC must satisfy 0 <= soc_min <= soc_initial <= soc_max <= 1")
    if p_fraction < 0.0:
        raise ValueError("ESS active-power max fraction must be nonnegative")
    return StorageParams(
        energy_mwh=energy,
        p_max=p_max,
        soc_initial=np.full(n_ess, args.soc_initial, dtype=float),
        soc_min=np.full(n_ess, args.soc_min, dtype=float),
        soc_max=np.full(n_ess, args.soc_max, dtype=float),
        eta_charge=float(args.eta_charge),
        eta_discharge=float(args.eta_discharge),
        dt_hours=float(args.dt_hours),
        terminal_equal_initial=not args.no_terminal_soc,
    )


def q_pv_capacity(day: DayAheadData) -> np.ndarray:
    return np.sqrt(np.maximum(day.s_pv.reshape(1, -1) ** 2 - day.p_pv**2, 0.0))


def variable_sizes(day: DayAheadData) -> dict[str, int]:
    return {
        "q_pv": HORIZON * day.pv_nodes.size,
        "p_ch": 0,
        "p_dis": 0,
        "q_ess": HORIZON * day.ess_nodes.size,
        "q_device": HORIZON * day.q_device_nodes.size,
    }


def pack_dispatch(dispatch: Dispatch) -> np.ndarray:
    return np.concatenate(
        [
            dispatch.q_pv.reshape(-1),
            dispatch.q_ess.reshape(-1),
            dispatch.q_device.reshape(-1),
        ]
    )


def unpack_dispatch(day: DayAheadData, x: np.ndarray) -> Dispatch:
    x = np.asarray(x, dtype=float).reshape(-1)
    n_pv = day.pv_nodes.size
    n_ess = day.ess_nodes.size
    n_qdev = day.q_device_nodes.size
    sizes = variable_sizes(day)
    expected = sum(sizes.values())
    if x.size != expected:
        raise ValueError(f"dispatch vector must have length {expected}, got {x.size}")

    cursor = 0
    q_pv = x[cursor : cursor + sizes["q_pv"]].reshape(HORIZON, n_pv)
    cursor += sizes["q_pv"]
    p_ch = np.zeros((HORIZON, n_ess), dtype=float)
    p_dis = np.zeros((HORIZON, n_ess), dtype=float)
    q_ess = x[cursor : cursor + sizes["q_ess"]].reshape(HORIZON, n_ess)
    cursor += sizes["q_ess"]
    q_device = x[cursor : cursor + sizes["q_device"]].reshape(HORIZON, n_qdev)
    return Dispatch(q_pv=q_pv, p_ch=p_ch, p_dis=p_dis, q_ess=q_ess, q_device=q_device)


def baseline_dispatch(day: DayAheadData) -> Dispatch:
    return Dispatch(
        q_pv=np.zeros((HORIZON, day.pv_nodes.size), dtype=float),
        p_ch=np.zeros((HORIZON, day.ess_nodes.size), dtype=float),
        p_dis=np.zeros((HORIZON, day.ess_nodes.size), dtype=float),
        q_ess=np.zeros((HORIZON, day.ess_nodes.size), dtype=float),
        q_device=np.zeros((HORIZON, day.q_device_nodes.size), dtype=float),
    )


def load_hourly_q_warm_start(path: str | Path | None) -> dict[int, np.ndarray]:
    if path is None:
        return {}
    start_path = resolve_repo_path(path)
    if not start_path.is_file():
        print(f"Hourly Q warm-start JSON not found; continuing without it: {start_path}", flush=True)
        return {}
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


def dispatch_from_hourly_q_start(day: DayAheadData, starts: dict[int, np.ndarray]) -> Dispatch:
    dispatch = baseline_dispatch(day)
    n_pv = day.pv_nodes.size
    n_ess = day.ess_nodes.size
    n_qdev = day.q_device_nodes.size
    expected = n_pv + n_ess + n_qdev
    pv_caps = q_pv_capacity(day)

    for hour, q_control in sorted(starts.items()):
        if hour < 0 or hour >= HORIZON:
            continue
        if q_control.size != expected:
            raise ValueError(
                f"Warm-start hour {hour} Q_control_opt must have length {expected}, got {q_control.size}"
            )
        dispatch.q_pv[hour] = np.clip(q_control[:n_pv], -pv_caps[hour], pv_caps[hour])
        dispatch.q_ess[hour] = np.clip(q_control[n_pv : n_pv + n_ess], -day.s_ess, day.s_ess)
        dispatch.q_device[hour] = np.clip(q_control[n_pv + n_ess :], day.q_device_min, day.q_device_max)
    return dispatch


def bounds(day: DayAheadData, storage: StorageParams) -> list[tuple[float, float]]:
    pv_caps = q_pv_capacity(day)
    result: list[tuple[float, float]] = []
    result.extend((float(-cap), float(cap)) for cap in pv_caps.reshape(-1))
    result.extend((float(-s), float(s)) for _ in range(HORIZON) for s in day.s_ess)
    result.extend(
        (float(lo), float(hi))
        for _ in range(HORIZON)
        for lo, hi in zip(day.q_device_min, day.q_device_max)
    )
    return result


def soc_trajectory(storage: StorageParams, dispatch: Dispatch) -> np.ndarray:
    n_ess = dispatch.p_ch.shape[1]
    soc = np.zeros((HORIZON + 1, n_ess), dtype=float)
    if n_ess == 0:
        return soc
    soc[:] = storage.soc_initial.reshape(1, -1)
    return soc


def build_net_injection(day: DayAheadData, dispatch: Dispatch, hour: int) -> np.ndarray:
    x_net = np.zeros((33, 2), dtype=float)
    x_net[:, 0] = -day.p_load[:, hour]
    x_net[:, 1] = -day.q_load[:, hour]
    for k, bus in enumerate(day.pv_nodes):
        x_net[int(bus), 0] += day.p_pv[hour, k]
        x_net[int(bus), 1] += dispatch.q_pv[hour, k]
    for k, bus in enumerate(day.ess_nodes):
        x_net[int(bus), 0] += dispatch.p_dis[hour, k] - dispatch.p_ch[hour, k]
        x_net[int(bus), 1] += dispatch.q_ess[hour, k]
    for k, bus in enumerate(day.q_device_nodes):
        x_net[int(bus), 1] += dispatch.q_device[hour, k]
    return x_net


def evaluate_24h(day: DayAheadData, distflow: ExactDistFlow, dispatch: Dispatch, *, pf_max_iter: int):
    metrics = [
        distflow.solve(build_net_injection(day, dispatch, t), max_iter=pf_max_iter)
        for t in range(HORIZON)
    ]
    return metrics


def hourly_control_bounds(day: DayAheadData, hour: int) -> tuple[np.ndarray, np.ndarray]:
    pv_caps = q_pv_capacity(day)[hour]
    lower = np.concatenate([-pv_caps, -day.s_ess, day.q_device_min])
    upper = np.concatenate([pv_caps, day.s_ess, day.q_device_max])
    return lower.astype(float), upper.astype(float)


def split_hour_control(day: DayAheadData, q_control: np.ndarray):
    q = np.asarray(q_control, dtype=float).reshape(-1)
    n_pv = day.pv_nodes.size
    n_ess = day.ess_nodes.size
    n_qdev = day.q_device_nodes.size
    expected = n_pv + n_ess + n_qdev
    if q.size != expected:
        raise ValueError(f"hourly q_control must have length {expected}, got {q.size}")
    return q[:n_pv], q[n_pv : n_pv + n_ess], q[n_pv + n_ess :]


def build_hour_net_injection(day: DayAheadData, hour: int, q_control: np.ndarray) -> np.ndarray:
    q_pv, q_ess, q_device = split_hour_control(day, q_control)
    x_net = np.zeros((33, 2), dtype=float)
    x_net[:, 0] = -day.p_load[:, hour]
    x_net[:, 1] = -day.q_load[:, hour]
    for k, bus in enumerate(day.pv_nodes):
        x_net[int(bus), 0] += day.p_pv[hour, k]
        x_net[int(bus), 1] += q_pv[k]
    for k, bus in enumerate(day.ess_nodes):
        x_net[int(bus), 1] += q_ess[k]
    for k, bus in enumerate(day.q_device_nodes):
        x_net[int(bus), 1] += q_device[k]
    return x_net


def solve_hour_reactive_rpo_gurobi(
    day: DayAheadData,
    distflow: ExactDistFlow,
    hour: int,
    args,
    *,
    q_start: np.ndarray | None,
) -> dict:
    """Solve one hourly exact DistFlow RPO as a continuous Gurobi QCP."""
    gp_mod, grb_mod = require_gurobi()
    model = gp_mod.Model(f"Exact_DistFlow_RPO_h{hour:02d}")
    model.Params.OutputFlag = 0 if args.quiet_gurobi else 1
    model.Params.NonConvex = 2
    model.Params.TimeLimit = float(args.gurobi_time_limit)
    model.Params.MIPGap = float(args.gurobi_mip_gap)

    lower, upper = hourly_control_bounds(day, hour)
    n_pv = day.pv_nodes.size
    n_ess = day.ess_nodes.size
    n_qdev = day.q_device_nodes.size
    n_ctrl = lower.size
    q_ctrl = model.addVars(n_ctrl, lb=-grb_mod.INFINITY, ub=grb_mod.INFINITY, name="Q_ctrl")
    for idx in range(n_ctrl):
        q_ctrl[idx].LB = float(lower[idx])
        q_ctrl[idx].UB = float(upper[idx])
        if q_start is not None:
            q_ctrl[idx].Start = float(np.clip(np.asarray(q_start, dtype=float).reshape(-1)[idx], lower[idx], upper[idx]))
        else:
            q_ctrl[idx].Start = 0.0

    branches = np.asarray(day.branches, dtype=float)
    branch_count = branches.shape[0]
    z_base_ohm = float(args.base_kv) ** 2 / float(args.base_mva)
    i_base_ka = float(args.base_mva) / (np.sqrt(3.0) * float(args.base_kv))
    r_pu = branches[:, 2] / z_base_ohm
    x_pu = branches[:, 3] / z_base_ohm

    v_sq_lb = float(args.v_lower) ** 2
    v_sq_ub = float(args.v_upper) ** 2
    v_sq = model.addVars(33, lb=v_sq_lb, ub=v_sq_ub, name="v_sq")
    p_flow = model.addVars(branch_count, lb=-grb_mod.INFINITY, ub=grb_mod.INFINITY, name="P_flow")
    q_flow = model.addVars(branch_count, lb=-grb_mod.INFINITY, ub=grb_mod.INFINITY, name="Q_flow")
    ell = model.addVars(branch_count, lb=0.0, ub=grb_mod.INFINITY, name="ell")
    ell_limit = (float(args.line_max_i_ka) / i_base_ka) ** 2
    if not args.skip_current_constraints:
        for branch_idx in range(branch_count):
            ell[branch_idx].UB = float(ell_limit)

    parent_branch = np.full(33, -1, dtype=int)
    children = [[] for _ in range(33)]
    for branch_idx, (fr, to, _, _) in enumerate(branches):
        fr_i = int(fr)
        to_i = int(to)
        parent_branch[to_i] = branch_idx
        children[fr_i].append(to_i)

    pv_to_idx = {int(bus): idx for idx, bus in enumerate(day.pv_nodes)}
    ess_to_idx = {int(bus): idx for idx, bus in enumerate(day.ess_nodes)}
    qdev_to_idx = {int(bus): idx for idx, bus in enumerate(day.q_device_nodes)}

    model.addConstr(v_sq[0] == float(args.slack_vm_pu) ** 2, name="slack_v_sq")
    for node in range(32, 0, -1):
        branch_idx = int(parent_branch[node])
        child_branches = [int(parent_branch[c]) for c in children[node]]
        p_demand = (float(day.p_load[node, hour]) - (float(day.p_pv[hour, pv_to_idx[node]]) if node in pv_to_idx else 0.0)) / float(args.base_mva)
        q_demand = float(day.q_load[node, hour]) / float(args.base_mva)
        if node in pv_to_idx:
            q_demand += -q_ctrl[pv_to_idx[node]] / float(args.base_mva)
        if node in ess_to_idx:
            q_demand += -q_ctrl[n_pv + ess_to_idx[node]] / float(args.base_mva)
        if node in qdev_to_idx:
            q_demand += -q_ctrl[n_pv + n_ess + qdev_to_idx[node]] / float(args.base_mva)

        model.addConstr(
            p_flow[branch_idx]
            == p_demand
            + gp_mod.quicksum(p_flow[idx] for idx in child_branches)
            + float(r_pu[branch_idx]) * ell[branch_idx],
            name=f"P_balance_{node}",
        )
        model.addConstr(
            q_flow[branch_idx]
            == q_demand
            + gp_mod.quicksum(q_flow[idx] for idx in child_branches)
            + float(x_pu[branch_idx]) * ell[branch_idx],
            name=f"Q_balance_{node}",
        )

    for branch_idx, (fr, to, _, _) in enumerate(branches):
        fr_i = int(fr)
        to_i = int(to)
        model.addConstr(
            v_sq[to_i]
            == v_sq[fr_i]
            - 2.0 * (float(r_pu[branch_idx]) * p_flow[branch_idx] + float(x_pu[branch_idx]) * q_flow[branch_idx])
            + (float(r_pu[branch_idx]) ** 2 + float(x_pu[branch_idx]) ** 2) * ell[branch_idx],
            name=f"V_drop_{branch_idx}",
        )
        model.addQConstr(
            p_flow[branch_idx] * p_flow[branch_idx] + q_flow[branch_idx] * q_flow[branch_idx]
            == v_sq[fr_i] * ell[branch_idx],
            name=f"Branch_flow_soc_eq_{branch_idx}",
        )

    v_mag = model.addVars(33, lb=float(args.v_lower), ub=float(args.v_upper), name="v_mag")
    v_dev = model.addVars(32, lb=0.0, name="v_abs_dev")
    x_pts = np.linspace(float(args.v_lower) ** 2, float(args.v_upper) ** 2, int(args.pwl_points))
    y_pts = np.sqrt(x_pts)
    model.addConstr(v_mag[0] == float(args.slack_vm_pu), name="slack_v_mag")
    for node in range(1, 33):
        model.addGenConstrPWL(
            v_sq[node],
            v_mag[node],
            x_pts.tolist(),
            y_pts.tolist(),
            name=f"sqrt_v_pwl_{node}",
        )
        model.addConstr(v_dev[node - 1] >= v_mag[node] - 1.0, name=f"vdev_pos_{node}")
        model.addConstr(v_dev[node - 1] >= 1.0 - v_mag[node], name=f"vdev_neg_{node}")

    model.setObjective(
        float(args.voltage_weight) * gp_mod.quicksum(v_dev[node] for node in range(32))
        + float(args.loss_weight)
        * gp_mod.quicksum(3.0 * (i_base_ka**2) * float(branches[idx, 2]) * ell[idx] for idx in range(branch_count)),
        grb_mod.MINIMIZE,
    )
    model.optimize()

    if model.SolCount == 0:
        return {
            "success": False,
            "message": f"Gurobi status {int(model.status)}",
            "objective": None,
            "max_constraint_violation": float("inf"),
            "q_control": np.zeros(n_ctrl, dtype=float),
            "metrics": distflow.solve(build_hour_net_injection(day, hour, np.zeros(n_ctrl, dtype=float))),
            "nit": int(model.NodeCount),
            "nfev": 0,
            "gurobi_status": int(model.status),
        }

    q_solution = np.array([q_ctrl[idx].X for idx in range(n_ctrl)], dtype=float)
    metrics = distflow.solve(build_hour_net_injection(day, hour, q_solution), max_iter=args.distflow_pf_max_iter)
    violation = max(
        float(args.v_lower) - float(np.min(metrics.voltage_pu[1:])),
        float(np.max(metrics.voltage_pu[1:])) - float(args.v_upper),
        0.0,
    )
    if not args.skip_current_constraints:
        violation = max(violation, float(np.max(metrics.branch_current_ka)) - float(args.line_max_i_ka))

    return {
        "success": bool(violation <= 1e-5),
        "message": f"Gurobi status {int(model.status)}",
        "objective": float(
            objective_from_metrics(
                [metrics],
                loss_weight=args.loss_weight,
                voltage_weight=args.voltage_weight,
            )
        ),
        "max_constraint_violation": float(max(violation, 0.0)),
        "q_control": q_solution,
        "metrics": metrics,
        "nit": int(model.NodeCount),
        "nfev": 0,
        "gurobi_status": int(model.status),
        "gurobi_objective": float(model.ObjVal),
    }


def solve_24h_hourly_decomposed(day: DayAheadData, storage: StorageParams, distflow: ExactDistFlow, args) -> dict:
    """Solve the 24h pure reactive DistFlow problem as 24 independent hours."""
    n_pv = day.pv_nodes.size
    n_ess = day.ess_nodes.size
    n_qdev = day.q_device_nodes.size
    dispatch = baseline_dispatch(day)
    warm_starts = load_hourly_q_warm_start(args.hourly_warm_start_json) if args.hourly_warm_start_json else {}
    metrics = []
    hourly_results = []
    objective = 0.0
    max_violation = 0.0
    success = True
    total_nit = 0
    total_nfev = 0

    for hour in range(HORIZON):
        q_start = warm_starts.get(hour)
        result = solve_hour_reactive_rpo_gurobi(day, distflow, hour, args, q_start=q_start)
        q_pv, q_ess, q_device = split_hour_control(day, result["q_control"])
        dispatch.q_pv[hour] = q_pv
        dispatch.q_ess[hour] = q_ess
        dispatch.q_device[hour] = q_device
        metrics.append(result["metrics"])
        objective += float(result["objective"])
        max_violation = max(max_violation, float(result["max_constraint_violation"]))
        success = success and bool(result["success"])
        total_nit += max(int(result["nit"]), 0)
        total_nfev += max(int(result["nfev"]), 0)
        hourly_results.append(
            {
                "hour": hour,
                "success": result["success"],
                "message": result["message"],
                "objective": result["objective"],
                "max_constraint_violation": result["max_constraint_violation"],
                "nit": result["nit"],
                "nfev": result["nfev"],
            }
        )
        print(
            f"hour {hour:02d}: success={result['success']} "
            f"obj={result['objective']:.6f} "
            f"viol={result['max_constraint_violation']:.3e} "
            f"nit={result['nit']} nfev={result['nfev']}",
            flush=True,
        )

    return {
        "success": bool(success),
        "message": "hourly decomposed pure reactive DistFlow optimization",
        "raw_optimizer_success": bool(success),
        "objective": float(objective),
        "max_constraint_violation": float(max_violation),
        "dispatch": dispatch,
        "metrics": metrics,
        "soc": soc_trajectory(storage, dispatch),
        "nit": int(total_nit),
        "nfev": int(total_nfev),
        "hourly_results": hourly_results,
        "solve_mode": "hourly",
    }


def objective_from_metrics(metrics, *, loss_weight: float, voltage_weight: float) -> float:
    vdev = sum(float(m.vdev_total) for m in metrics)
    loss = sum(float(m.total_loss_mw) for m in metrics)
    return float(voltage_weight) * vdev + float(loss_weight) * loss


def constraint_violation(metrics, day: DayAheadData, storage: StorageParams, dispatch: Dispatch, args) -> float:
    violation = 0.0
    for m in metrics:
        violation = max(violation, float(args.v_lower) - float(np.min(m.voltage_pu[1:])))
        violation = max(violation, float(np.max(m.voltage_pu[1:])) - float(args.v_upper))
        if not args.skip_current_constraints:
            violation = max(violation, float(np.max(m.branch_current_ka)) - float(args.line_max_i_ka))
    soc = soc_trajectory(storage, dispatch)
    if day.ess_nodes.size:
        inv_margin = day.s_ess.reshape(1, -1) ** 2 - dispatch.q_ess**2
        violation = max(violation, float(np.max(-inv_margin)))
    return max(violation, 0.0)


def control_labels(day: DayAheadData) -> list[str]:
    labels = [f"PV@Bus{int(bus) + 1}" for bus in day.pv_nodes]
    labels.extend(f"ESS@Bus{int(bus) + 1}" for bus in day.ess_nodes)
    for idx, bus in enumerate(day.q_device_nodes):
        name = day.q_device_names[idx] if idx < len(day.q_device_names) else f"QDev{idx + 1}"
        labels.append(f"{name}@Bus{int(bus) + 1}")
    return labels


def summarize_hourly(metrics) -> list[dict]:
    rows = []
    for t, m in enumerate(metrics):
        rows.append(
            {
                "hour": t,
                "v_min": float(np.min(m.voltage_pu[1:])),
                "v_max": float(np.max(m.voltage_pu[1:])),
                "vdev_total": float(m.vdev_total),
                "vworst": float(m.vworst),
                "worst_i_margin": float(m.worst_i_margin),
                "total_loss_mw": float(m.total_loss_mw),
                "converged": bool(m.converged),
                "iterations": int(m.iterations),
            }
        )
    return rows


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(out_dir: Path, metrics, soc: np.ndarray, dispatch: Dispatch) -> None:
    plt = get_pyplot()
    out_dir.mkdir(parents=True, exist_ok=True)
    hours = np.arange(HORIZON)
    v_min = np.array([np.min(m.voltage_pu[1:]) for m in metrics], dtype=float)
    v_max = np.array([np.max(m.voltage_pu[1:]) for m in metrics], dtype=float)
    vdev = np.array([m.vdev_total for m in metrics], dtype=float)
    loss = np.array([m.total_loss_mw for m in metrics], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), dpi=160, sharex=True)
    axes[0].plot(hours, v_min, marker="o", label="V min")
    axes[0].plot(hours, v_max, marker="s", label="V max")
    axes[0].axhline(0.95, color="tab:red", linestyle="--", linewidth=1.0)
    axes[0].axhline(1.05, color="tab:red", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Voltage (p.u.)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(hours, vdev, marker="o", color="tab:green")
    axes[1].set_ylabel("Vdev")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(hours, loss, marker="o", color="tab:purple")
    axes[2].set_xlabel("Hour")
    axes[2].set_ylabel("Loss (MW)")
    axes[2].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "hourly_voltage_loss.png")
    plt.close(fig)

    if soc.size:
        fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
        for k in range(soc.shape[1]):
            ax.step(np.arange(HORIZON + 1), soc[:, k], where="post", label=f"ESS{k + 1}")
        ax.set_xlabel("Hour")
        ax.set_ylabel("SOC")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "ess_soc.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
        p_net = dispatch.p_dis - dispatch.p_ch
        for k in range(p_net.shape[1]):
            ax.step(hours, p_net[:, k], where="mid", label=f"ESS{k + 1} P")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Hour")
        ax.set_ylabel("P discharge - charge (MW)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "ess_active_power.png")
        plt.close(fig)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Dispatch):
        return {
            "q_pv": value.q_pv.tolist(),
            "p_ch": value.p_ch.tolist(),
            "p_dis": value.p_dis.tolist(),
            "p_net": (value.p_dis - value.p_ch).tolist(),
            "q_ess": value.q_ess.tolist(),
            "q_device": value.q_device.tolist(),
        }
    if hasattr(value, "voltage_pu") and hasattr(value, "branch_current_ka"):
        return {
            "converged": value.converged,
            "iterations": value.iterations,
            "voltage_pu": value.voltage_pu.tolist(),
            "branch_p_mw": value.branch_p_mw.tolist(),
            "branch_q_mvar": value.branch_q_mvar.tolist(),
            "branch_current_ka": value.branch_current_ka.tolist(),
            "branch_loss_mw": value.branch_loss_mw.tolist(),
            "total_loss_mw": value.total_loss_mw,
            "vdev_total": value.vdev_total,
            "vworst": value.vworst,
            "worst_i_margin": value.worst_i_margin,
        }
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact DistFlow 24-hour continuous RPO.")
    parser.add_argument("--data", default="data/ieee33_nodal_pq_correlated_raw_pool_50k.pt")
    parser.add_argument("--out-dir", default="rpo_milp/disflow/results/day_ahead_24h")
    parser.add_argument(
        "--solve-mode",
        choices=["hourly"],
        default="hourly",
        help="Solve 24 independent 11-variable Gurobi DistFlow reactive optimizations.",
    )
    parser.add_argument("--v-lower", type=float, default=0.95)
    parser.add_argument("--v-upper", type=float, default=1.05)
    parser.add_argument("--slack-vm-pu", type=float, default=1.03)
    parser.add_argument("--base-kv", type=float, default=12.66)
    parser.add_argument("--base-mva", type=float, default=1.0)
    parser.add_argument("--line-max-i-ka", type=float, default=0.20)
    parser.add_argument("--skip-current-constraints", action="store_true")
    parser.add_argument("--distflow-pf-max-iter", type=int, default=200)
    parser.add_argument("--gurobi-time-limit", type=float, default=120.0)
    parser.add_argument("--gurobi-mip-gap", type=float, default=1e-4)
    parser.add_argument("--quiet-gurobi", action="store_true")
    parser.add_argument("--pwl-points", type=int, default=41)
    parser.add_argument("--voltage-weight", type=float, default=1.0)
    parser.add_argument("--loss-weight", type=float, default=0.0)
    parser.add_argument("--ess-energy-hours", type=float, default=2.0)
    parser.add_argument(
        "--ess-p-max-fraction",
        type=float,
        default=0.0,
        help="Kept for compatibility; ESS active power is fixed to zero in this script.",
    )
    parser.add_argument("--soc-initial", type=float, default=0.5)
    parser.add_argument("--soc-min", type=float, default=0.1)
    parser.add_argument("--soc-max", type=float, default=0.9)
    parser.add_argument("--eta-charge", type=float, default=0.95)
    parser.add_argument("--eta-discharge", type=float, default=0.95)
    parser.add_argument("--dt-hours", type=float, default=1.0)
    parser.add_argument(
        "--no-terminal-soc",
        action="store_true",
        help="Kept for compatibility; SOC is constant because ESS active power is fixed to zero.",
    )
    parser.add_argument(
        "--hourly-warm-start-json",
        default=None,
        help=(
            "Optional diagnostic warm start. Leave unset for an independent "
            "DistFlow baseline that starts from all-zero reactive controls."
        ),
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only evaluate the zero-dispatch 24h DistFlow baseline and write outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    day = load_day_ahead_data(args.data)
    storage = make_storage_params(day, args)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    distflow = ExactDistFlow(
        branches=day.branches,
        slack_vm_pu=args.slack_vm_pu,
        base_kv=args.base_kv,
        base_mva=args.base_mva,
        line_max_i_ka=args.line_max_i_ka,
    )

    print("Loaded 24h DistFlow RPO case:")
    print(f"  data: {day.data_path}")
    print(f"  controls: {control_labels(day)}")
    print(f"  variables: {sum(variable_sizes(day).values())}")
    print(
        "  optimizer: hourly Gurobi QCP pure reactive dispatch; "
        "ESS active power is fixed at zero.",
        flush=True,
    )

    if args.baseline_only:
        dispatch = baseline_dispatch(day)
        metrics = evaluate_24h(day, distflow, dispatch, pf_max_iter=args.distflow_pf_max_iter)
        result = {
            "success": True,
            "message": "baseline-only",
            "raw_optimizer_success": True,
            "objective": objective_from_metrics(
                metrics,
                loss_weight=args.loss_weight,
                voltage_weight=args.voltage_weight,
            ),
            "max_constraint_violation": constraint_violation(metrics, day, storage, dispatch, args),
            "solve_time_sec": 0.0,
            "nit": 0,
            "nfev": 1,
            "dispatch": dispatch,
            "metrics": metrics,
            "soc": soc_trajectory(storage, dispatch),
        }
        solve_time = 0.0
    else:
        start = time.perf_counter()
        result = solve_24h_hourly_decomposed(day, storage, distflow, args)
        solve_time = time.perf_counter() - start

    rows = summarize_hourly(result["metrics"])
    write_summary_csv(out_dir / "hourly_summary.csv", rows)
    make_plots(out_dir, result["metrics"], result["soc"], result["dispatch"])

    payload = {
        "success": result["success"],
        "message": result["message"],
        "raw_optimizer_success": result["raw_optimizer_success"],
        "objective": result["objective"],
        "max_constraint_violation": result["max_constraint_violation"],
        "solve_time_sec": float(solve_time),
        "nit": result["nit"],
        "nfev": result["nfev"],
        "solve_mode": result.get("solve_mode", args.solve_mode),
        "hourly_results": result.get("hourly_results"),
        "data": str(day.data_path),
        "control_labels": control_labels(day),
        "pv_nodes": day.pv_nodes.tolist(),
        "s_pv": day.s_pv.tolist(),
        "Ppv_24h": day.p_pv.tolist(),
        "ess_nodes": day.ess_nodes.tolist(),
        "s_ess": day.s_ess.tolist(),
        "storage": {
            "energy_mwh": storage.energy_mwh.tolist(),
            "p_max": storage.p_max.tolist(),
            "soc_initial": storage.soc_initial.tolist(),
            "soc_min": storage.soc_min.tolist(),
            "soc_max": storage.soc_max.tolist(),
            "eta_charge": storage.eta_charge,
            "eta_discharge": storage.eta_discharge,
            "terminal_equal_initial": storage.terminal_equal_initial,
            "active_power_note": "P_ess_ch and P_ess_dis are fixed to zero; ESS only provides reactive power.",
        },
        "q_device_nodes": day.q_device_nodes.tolist(),
        "q_device_names": day.q_device_names,
        "q_device_min": day.q_device_min.tolist(),
        "q_device_max": day.q_device_max.tolist(),
        "dispatch": result["dispatch"],
        "soc": result["soc"],
        "hourly_summary": rows,
        "metrics": result["metrics"],
    }
    with (out_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2)

    print()
    print("24h exact DistFlow RPO complete.")
    print(f"Success: {result['success']} ({result['message']})")
    print(f"Objective: {result['objective']:.8f}")
    print(f"Max constraint violation: {result['max_constraint_violation']:.8e}")
    print(f"Solve time: {solve_time:.3f} s")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
