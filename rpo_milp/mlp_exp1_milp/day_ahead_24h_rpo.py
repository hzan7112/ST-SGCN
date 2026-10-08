"""Day-ahead 24-hour RPO with the ``model/mlp_exp1`` MILP surrogate.

This follows the Exp17 day-ahead device/SOC/constraint setting, but replaces
the ST-SGCN block with a pure raw-PQ MLP. The MLP consumes only
``[P_net, Q_net]`` and predicts Exp17-style ``V_nodes`` plus ``YI_worst``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rpo_milp.exp17_milp import day_ahead_24h_rpo as base
from rpo_milp.exp17_milp import single_step_rpo as exp17_rpo
from rpo_milp.mlp_exp1_milp.mlp_milp_converter import DEFAULT_ENGINE_PATH, MLPExp1MILPConverter


HORIZON = 24
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "ieee33_nodal_pq_correlated_raw_pool_50k.pt"
DEFAULT_RESULT_PATH = REPO_ROOT / "rpo_milp" / "mlp_exp1_milp" / "results" / "day_ahead_24h_result.json"


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


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


def add_physical_output_bounds(model, outputs, hour: int) -> None:
    model.addConstr(outputs.Vdev_total >= 0.0, name=f"Physical_Vdev_t{hour:02d}")
    model.addConstr(outputs.WorstI >= -1.0, name=f"Physical_WorstI_t{hour:02d}")
    model.addConstr(outputs.Vworst >= -0.05, name=f"Physical_Vworst_t{hour:02d}")


def optimize_hour(day, args, hour: int, converter: MLPExp1MILPConverter) -> dict:
    gp_mod, grb_mod = exp17_rpo.require_gurobi()
    model = gp_mod.Model(f"MLP_Exp1_RPO_Hour_{hour:02d}")
    model.Params.TimeLimit = float(args.hourly_time_limit)
    model.Params.MIPGap = float(args.hourly_mip_gap)
    model.Params.OutputFlag = 0 if args.quiet else 1
    model.Params.NonConvex = 2
    model.Params.DualReductions = int(args.dual_reductions)

    n_pv = day.pv_nodes.size
    n_ess = day.ess_nodes.size
    n_qdev = day.q_device_nodes.size
    q_pv = model.addVars(n_pv, lb=-grb_mod.INFINITY, ub=grb_mod.INFINITY, name="Q_pv")
    q_ess = model.addVars(n_ess, lb=-grb_mod.INFINITY, ub=grb_mod.INFINITY, name="Q_ess")
    q_device = model.addVars(n_qdev, lb=-grb_mod.INFINITY, ub=grb_mod.INFINITY, name="Q_device")

    pv_caps = base.q_capacity(day.p_pv[hour], day.s_pv)
    for k, bus in enumerate(day.pv_nodes):
        q_cap = float(pv_caps[k])
        q_pv[k].LB = -q_cap
        q_pv[k].UB = q_cap
        q_pv[k].Start = 0.0
        model.addQConstr(
            q_pv[k] * q_pv[k] <= max(float(day.s_pv[k]) ** 2 - float(day.p_pv[hour, k]) ** 2, 0.0),
            name=f"PV_Cap_bus{int(bus)}",
        )

    for k, bus in enumerate(day.ess_nodes):
        q_ess[k].LB = -float(day.s_ess[k])
        q_ess[k].UB = float(day.s_ess[k])
        q_ess[k].Start = 0.0
        model.addQConstr(
            q_ess[k] * q_ess[k] <= float(day.s_ess[k]) ** 2,
            name=f"ESS_Q_Cap_bus{int(bus)}",
        )

    q_mid = 0.5 * (day.q_device_min + day.q_device_max) if n_qdev else np.array([], dtype=float)
    for k, bus in enumerate(day.q_device_nodes):
        q_device[k].LB = float(day.q_device_min[k])
        q_device[k].UB = float(day.q_device_max[k])
        q_device[k].Start = float(q_mid[k])

    q_pv_24 = {(hour, k): q_pv[k] for k in range(n_pv)}
    q_ess_24 = {(hour, k): q_ess[k] for k in range(n_ess)}
    q_device_24 = {(hour, k): q_device[k] for k in range(n_qdev)}
    zero_p = {(hour, k): gp_mod.LinExpr(0.0) for k in range(n_ess)}
    x_vars = base.build_hourly_injections(
        model,
        day,
        hour=hour,
        q_pv=q_pv_24,
        p_ch=zero_p,
        p_dis=zero_p,
        q_ess=q_ess_24,
        q_device=q_device_24,
    )

    outputs = converter.embed_mlp_constraints(
        model,
        x_vars,
        topo_mask=day.topo_mask,
        name_prefix=f"mlp_exp1_h{hour:02d}",
    )
    if not args.no_physical_output_bounds:
        add_physical_output_bounds(model, outputs, hour)

    objective = gp_mod.LinExpr(0.0)
    objective += outputs.Vdev_total
    slack_vars = None
    if args.diagnose_slack:
        s_v = model.addVar(lb=0.0, name="slack_Vworst")
        s_i = model.addVar(lb=0.0, name="slack_WorstI")
        model.addConstr(outputs.Vworst <= float(args.voltage_margin) + s_v, name="Surrogate_Vworst_safe")
        model.addConstr(outputs.WorstI <= float(args.current_margin) + s_i, name="Surrogate_WorstI_safe")
        objective += float(args.slack_penalty) * (s_v + s_i)
        slack_vars = (s_v, s_i)
    else:
        model.addConstr(outputs.Vworst <= float(args.voltage_margin), name="Surrogate_Vworst_safe")
        model.addConstr(outputs.WorstI <= float(args.current_margin), name="Surrogate_WorstI_safe")

    for k in range(n_pv):
        penalty, _ = base.add_abs_penalty(model, q_pv[k], f"Abs_Q_pv_{k}", args.q_movement_penalty)
        objective += penalty
    for k in range(n_ess):
        penalty, _ = base.add_abs_penalty(model, q_ess[k], f"Abs_Q_ess_{k}", args.q_movement_penalty)
        objective += penalty
    for k in range(n_qdev):
        penalty, _ = base.add_abs_penalty(model, q_device[k], f"Abs_Q_device_{k}", args.q_movement_penalty)
        objective += penalty

    if args.export_lp:
        export_base = resolve_repo_path(args.export_lp)
        export_path = export_base.with_name(f"{export_base.stem}_h{hour:02d}{export_base.suffix or '.lp'}")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        model.write(str(export_path))

    model.setObjective(objective, grb_mod.MINIMIZE)
    start = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - start

    result = {
        "hour": int(hour),
        "status": int(model.status),
        "status_name": exp17_rpo.status_name(model.status),
        "objective": None,
        "solve_time_sec": float(solve_time),
        "sol_count": int(model.SolCount),
        "Q_pv_opt": None,
        "Q_ess_opt": None,
        "Q_device_opt": None,
        "outputs": None,
        "safety_slack": None,
    }
    if model.SolCount > 0:
        result.update(
            {
                "objective": float(model.ObjVal),
                "Q_pv_opt": np.array([q_pv[k].X for k in range(n_pv)], dtype=float),
                "Q_ess_opt": np.array([q_ess[k].X for k in range(n_ess)], dtype=float),
                "Q_device_opt": np.array([q_device[k].X for k in range(n_qdev)], dtype=float),
                "outputs": {
                    "Vdev_total": float(outputs.Vdev_total.X),
                    "Vworst": float(outputs.Vworst.X),
                    "WorstI": float(outputs.WorstI.X),
                    "Ploss_total": float(outputs.Ploss_total.X),
                    "V_nodes": np.array([var.X for var in outputs.V_nodes], dtype=float),
                },
            }
        )
        if slack_vars is not None:
            s_v, s_i = slack_vars
            result["safety_slack"] = {"Vworst": float(s_v.X), "WorstI": float(s_i.X)}
    elif model.status == grb_mod.INFEASIBLE and args.iis:
        iis_base = resolve_repo_path(args.iis)
        iis_path = iis_base.with_name(f"{iis_base.stem}_h{hour:02d}{iis_base.suffix or '.ilp'}")
        iis_path.parent.mkdir(parents=True, exist_ok=True)
        model.computeIIS()
        model.write(str(iis_path))
        result["iis"] = str(iis_path)

    return result


def optimize_24h_hourly_decomposed(day, args) -> dict:
    converter = MLPExp1MILPConverter(
        args.engine,
        relu_formulation=args.relu_formulation,
        fallback_big_m=args.fallback_big_m,
        big_m_scale=args.big_m_scale,
    )
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

    for hour in range(HORIZON):
        print(f"\nSolving decomposed MLP Exp1 hour {hour:02d} / 23...", flush=True)
        hour_result = optimize_hour(day, args, hour, converter)
        solve_time += float(hour_result["solve_time_sec"])
        hourly_rows.append(hour_result)
        if hour_result.get("Q_pv_opt") is None:
            all_feasible = False
            outputs_all.append(None)
            continue
        q_pv_all[hour] = np.asarray(hour_result["Q_pv_opt"], dtype=float)
        q_ess_all[hour] = np.asarray(hour_result["Q_ess_opt"], dtype=float)
        q_device_all[hour] = np.asarray(hour_result["Q_device_opt"], dtype=float)
        outputs_all.append(hour_result["outputs"])
        objective += float(hour_result["objective"])

    return {
        "status": 2 if all_feasible else 9,
        "status_name": "HOURLY_DECOMPOSED_OPTIMAL" if all_feasible else "HOURLY_DECOMPOSED_PARTIAL",
        "objective": float(objective) if all_feasible else None,
        "solve_time_sec": float(solve_time),
        "data": str(day.data_path),
        "engine": str(resolve_repo_path(args.engine)),
        "horizon": HORIZON,
        "solve_mode": "hourly_decomposed",
        "surrogate": "model/mlp_exp1 MLPExp17Outputs raw-PQ",
        "device_meta": {
            "pv_q_caps": base.q_capacity(day.p_pv, day.s_pv.reshape(1, -1)),
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
        "control_labels": base.control_labels(day),
        "pv_nodes": day.pv_nodes.tolist(),
        "ess_nodes": day.ess_nodes.tolist(),
        "q_device_nodes": day.q_device_nodes.tolist(),
        "q_device_names": day.q_device_names,
        "Pload_sum_24h": day.p_load.sum(axis=0).tolist(),
        "Qload_sum_24h": day.q_load.sum(axis=0).tolist(),
        "Ppv_24h": day.p_pv.tolist(),
        "binary_count_estimate_per_hour": converter.binary_count,
        "binary_count_estimate_total_surrogate": int(HORIZON * converter.binary_count),
        "sol_count": 1 if all_feasible else 0,
        "Q_pv_opt": q_pv_all,
        "P_ess_ch_opt": p_ch_all,
        "P_ess_dis_opt": p_dis_all,
        "P_ess_net_opt": p_dis_all - p_ch_all,
        "Q_ess_opt": q_ess_all,
        "ESS_SOC": soc_all,
        "Q_device_opt": q_device_all,
        "outputs": outputs_all,
        "safety_slack": [row.get("safety_slack") for row in hourly_rows],
        "hourly_results": hourly_rows,
        "ess_throughput_mwh": 0.0,
    }


def optimize_24h(day, args) -> dict:
    gp_mod, grb_mod = exp17_rpo.require_gurobi()
    model = gp_mod.Model("MLP_Exp1_24h_DayAhead_RPO")
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

    converter = MLPExp1MILPConverter(
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

    ess_p_fraction = float(day.base_config.get("ess_p_max_fraction", args.ess_p_max_fraction))
    if args.fix_ess_active_zero:
        ess_p_fraction = 0.0
    if ess_p_fraction < 0.0:
        raise ValueError("ESS active-power max fraction must be nonnegative")
    if not (0.0 < args.eta_charge <= 1.0 and 0.0 < args.eta_discharge <= 1.0):
        raise ValueError("ESS charge/discharge efficiency must be in (0, 1]")

    device_meta = base.add_hourly_device_constraints(
        model,
        day,
        q_pv=q_pv,
        p_ch=p_ch,
        p_dis=p_dis,
        q_ess=q_ess,
        q_device=q_device,
        ess_p_max_fraction=ess_p_fraction,
        use_charge_binary=(not args.allow_simultaneous_charge_discharge and not args.fix_ess_active_zero),
    )
    soc_meta = base.add_soc_constraints(
        model,
        day,
        p_ch=p_ch,
        p_dis=p_dis,
        soc=soc,
        energy_mwh=np.asarray(args.ess_energy_hours * day.s_ess, dtype=float),
        soc_initial=np.full(n_ess, args.soc_initial, dtype=float),
        soc_min=np.full(n_ess, args.soc_min, dtype=float),
        soc_max=np.full(n_ess, args.soc_max, dtype=float),
        eta_charge=args.eta_charge,
        eta_discharge=args.eta_discharge,
        terminal_equal_initial=not args.no_terminal_soc,
        dt_hours=args.dt_hours,
    )
    base.set_neutral_start(
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
        hourly_warm_start = base.load_hourly_warm_start(args.hourly_warm_start_json)
        hourly_warm_start_meta = base.set_hourly_q_warm_start(
            day,
            hourly_warm_start,
            q_pv=q_pv,
            q_ess=q_ess,
            q_device=q_device,
        )

    objective = gp_mod.LinExpr(0.0)
    hourly_outputs = []
    safety_slacks = []
    for hour in range(HORIZON):
        x_vars = base.build_hourly_injections(
            model,
            day,
            hour=hour,
            q_pv=q_pv,
            p_ch=p_ch,
            p_dis=p_dis,
            q_ess=q_ess,
            q_device=q_device,
        )
        outputs = converter.embed_mlp_constraints(
            model,
            x_vars,
            topo_mask=day.topo_mask,
            name_prefix=f"mlp_exp1_24h_t{hour:02d}",
        )
        if not args.no_physical_output_bounds:
            add_physical_output_bounds(model, outputs, hour)

        objective += outputs.Vdev_total
        if args.diagnose_slack:
            s_v = model.addVar(lb=0.0, name=f"slack_Vworst_t{hour:02d}")
            s_i = model.addVar(lb=0.0, name=f"slack_WorstI_t{hour:02d}")
            model.addConstr(outputs.Vworst <= float(args.voltage_margin) + s_v, name=f"Surrogate_Vworst_safe_t{hour:02d}")
            model.addConstr(outputs.WorstI <= float(args.current_margin) + s_i, name=f"Surrogate_WorstI_safe_t{hour:02d}")
            objective += float(args.slack_penalty) * (s_v + s_i)
            safety_slacks.append((s_v, s_i))
        else:
            model.addConstr(outputs.Vworst <= float(args.voltage_margin), name=f"Surrogate_Vworst_safe_t{hour:02d}")
            model.addConstr(outputs.WorstI <= float(args.current_margin), name=f"Surrogate_WorstI_safe_t{hour:02d}")
        hourly_outputs.append(outputs)

        for k in range(n_pv):
            penalty, _ = base.add_abs_penalty(model, q_pv[hour, k], f"Abs_Q_pv_t{hour:02d}_{k}", args.q_movement_penalty)
            objective += penalty
        for k in range(n_ess):
            penalty, _ = base.add_abs_penalty(model, q_ess[hour, k], f"Abs_Q_ess_t{hour:02d}_{k}", args.q_movement_penalty)
            objective += penalty
        for k in range(n_qdev):
            penalty, _ = base.add_abs_penalty(model, q_device[hour, k], f"Abs_Q_device_t{hour:02d}_{k}", args.q_movement_penalty)
            objective += penalty

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
        "engine": str(resolve_repo_path(args.engine)),
        "horizon": HORIZON,
        "solve_mode": "monolithic",
        "surrogate": "model/mlp_exp1 MLPExp17Outputs raw-PQ",
        "device_meta": device_meta,
        "soc_meta": soc_meta,
        "control_labels": base.control_labels(day),
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
                        "Ploss_total": float(out.Ploss_total.X),
                        "V_nodes": np.array([var.X for var in out.V_nodes], dtype=float),
                    }
                    for out in hourly_outputs
                ],
                "safety_slack": (
                    [{"Vworst": float(s_v.X), "WorstI": float(s_i.X)} for s_v, s_i in safety_slacks]
                    if safety_slacks
                    else None
                ),
                "ess_throughput_mwh": float(
                    sum(
                        (p_ch[t, k].X + p_dis[t, k].X) * args.dt_hours
                        for t in range(HORIZON)
                        for k in range(n_ess)
                    )
                ),
            }
        )
    elif model.status == grb_mod.INFEASIBLE and args.iis:
        iis_path = resolve_repo_path(args.iis)
        iis_path.parent.mkdir(parents=True, exist_ok=True)
        model.computeIIS()
        model.write(str(iis_path))
        result["iis"] = str(iis_path)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="24-hour MLP Exp1-MILP reactive power optimization.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--engine", default=str(DEFAULT_ENGINE_PATH))
    parser.add_argument("--out", default=str(DEFAULT_RESULT_PATH))
    parser.add_argument(
        "--solve-mode",
        choices=["hourly", "monolithic"],
        default="hourly",
        help=(
            "hourly solves 24 independent single-hour MLP Exp1 MILPs and matches "
            "the Exp17 default workflow; monolithic builds the full 24h MIP."
        ),
    )
    parser.add_argument("--hourly-time-limit", type=float, default=120.0)
    parser.add_argument("--hourly-mip-gap", type=float, default=0.02)
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--mip-gap", type=float, default=0.02)
    parser.add_argument("--dual-reductions", type=int, choices=[0, 1], default=0)
    parser.add_argument("--relu-formulation", choices=["big_m", "general"], default="general")
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
    parser.add_argument("--fix-ess-active-zero", action="store_true", default=True)
    parser.add_argument("--optimize-ess-active", action="store_false", dest="fix_ess_active_zero")
    parser.add_argument("--hourly-warm-start-json", default=None)
    parser.add_argument("--ess-throughput-penalty", type=float, default=1e-3)
    parser.add_argument("--q-movement-penalty", type=float, default=0.0)
    parser.add_argument("--exact-pf-max-iter", type=int, default=200)
    parser.add_argument("--exact-slack-vm-pu", type=float, default=None)
    parser.add_argument("--exact-base-kv", type=float, default=None)
    parser.add_argument("--exact-base-mva", type=float, default=None)
    parser.add_argument("--exact-line-max-i-ka", type=float, default=None)
    parser.add_argument("--mip-focus", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--heuristics", type=float, default=0.5)
    parser.add_argument("--no-rel-heur-time", type=float, default=120.0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--export-lp", default=None)
    parser.add_argument("--iis", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    day = base.load_day_ahead_data(args.data)
    print("Loaded 24h profile data:")
    print(f"  data: {day.data_path}")
    print(f"  PV buses: {day.pv_nodes.tolist()}")
    print(f"  ESS buses: {day.ess_nodes.tolist()}")
    print(f"  Q devices: {base.control_labels(day)[day.pv_nodes.size + day.ess_nodes.size:]}")

    if args.solve_mode == "hourly":
        result = optimize_24h_hourly_decomposed(day, args)
    else:
        result = optimize_24h(day, args)
    result = base.attach_exact_power_flow_results(day, result, args)
    exact_pf = result.get("exact_pf") or {}
    if exact_pf.get("available"):
        exact_pf["voltage_source"] = "exact DistFlow power-flow evaluation of MLP Exp1 optimized dispatch"

    out_path = resolve_repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(result), f, indent=2)

    print()
    print("24h MLP Exp1-MILP RPO complete.")
    print(f"Status: {result['status_name']} ({result['status']})")
    print(f"Solve time: {result['solve_time_sec']:.3f} s")
    if result.get("surrogate_objective") is not None:
        print(f"Surrogate objective: {result['surrogate_objective']:.8f}")
    surrogate_outputs = [
        item for item in (result.get("surrogate_outputs") or result.get("outputs") or [])
        if isinstance(item, dict)
    ]
    if surrogate_outputs:
        print(f"Max surrogate Vworst: {max(x['Vworst'] for x in surrogate_outputs):.8f}")
        print(f"Max surrogate WorstI: {max(x['WorstI'] for x in surrogate_outputs):.8f}")
    if exact_pf.get("available"):
        print(f"Objective: {result['objective']:.8f} ({result['objective_source']})")
        print(f"Exact PF 24h Vdev_total: {exact_pf['objective_vdev_total_24h']:.8f}")
        print(f"Exact PF 24h total loss: {exact_pf['total_loss_mw_24h']:.8f} MW")
        print(f"Max exact PF Vworst: {exact_pf['max_Vworst']:.8f}")
        print(f"Max exact PF WorstI: {exact_pf['max_WorstI']:.8f}")
    print(f"Result JSON: {out_path}")


if __name__ == "__main__":
    main()
