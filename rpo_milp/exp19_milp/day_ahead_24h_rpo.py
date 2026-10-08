"""Day-ahead 24-hour reactive power optimization with the Exp19 SGCN-MILP.

The data and device settings follow the Exp17 24-hour RPO script. The solve
path uses 24 independent Exp19 single-step MILPs, which keeps the embedded
ReLU models tractable while preserving the same hourly PV, ESS-Q, and
reactive-device limits.

Exp19 has no learned network-loss head. The default objective follows the
single-step Exp19 setting:

    voltage_weight * Vdev_total + q_weight * sum_k (Q_k / Q_k,max)^2

After the surrogate dispatch is assembled, the script evaluates the 24-hour
schedule with exact DistFlow and reports physical voltage and loss metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rpo_milp.disflow import day_ahead_24h_distflow_rpo as disflow_24h
from rpo_milp.exp17_milp.day_ahead_24h_rpo import (
    DayAheadData,
    build_zero_q_base_injection,
    control_labels,
    load_day_ahead_data,
    q_capacity,
)
from rpo_milp.exp19_milp import single_step_rpo as exp19_rpo


HORIZON = 24


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


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
    exact_vdev = float(sum(item.vdev_total for item in metrics))
    exact_loss = float(sum(item.total_loss_mw for item in metrics))
    max_vworst = float(max(item.vworst for item in metrics))
    max_worst_i = float(max(item.worst_i_margin for item in metrics))
    all_converged = all(bool(item.converged) for item in metrics)

    result["exact_pf"] = {
        "available": True,
        "voltage_source": "exact DistFlow power-flow evaluation of Exp19 optimized dispatch",
        "objective_vdev_total_24h": exact_vdev,
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
    result["objective"] = exact_vdev
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


def optimize_24h_hourly_decomposed(day: DayAheadData, args) -> dict:
    """Solve 24 independent Exp19 single-hour MILPs and assemble one schedule."""
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
    surrogate_objective = 0.0
    solve_time = 0.0
    all_feasible = True
    binary_count = None

    for hour in range(HORIZON):
        print(f"\nSolving decomposed Exp19 hour {hour:02d} / 23...", flush=True)
        start = time.perf_counter()
        hour_result = exp19_rpo.run_single_step_rpo(
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
            objective=args.objective,
            loss_objective_model="none",
            surrogate_loss_weight=args.surrogate_loss_weight,
            surrogate_loss_consistency_rel=args.surrogate_loss_consistency_rel,
            surrogate_loss_consistency_abs=args.surrogate_loss_consistency_abs,
            loss_weight=args.loss_weight,
            voltage_weight=args.voltage_weight,
            q_weight=args.q_weight,
            voltage_margin=args.voltage_margin,
            current_margin=args.current_margin,
            physical_output_bounds=not args.no_physical_output_bounds,
            trust_region_fraction=args.trust_region_fraction,
            control_deviation_penalty=args.control_deviation_penalty,
            base_kv=args.base_kv,
            base_mva=args.base_mva,
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
                surrogate_objective += float(hour_result["objective"])

        outputs = hour_result.get("outputs") or {}
        outputs_all.append(
            {
                "Vdev_total": outputs.get("Vdev_total"),
                "Vworst": outputs.get("Vworst"),
                "WorstI": outputs.get("WorstI"),
            }
        )
        hourly_rows.append(
            {
                "hour": hour,
                "status": hour_result.get("status"),
                "status_name": hour_result.get("status_name"),
                "sol_count": hour_result.get("sol_count"),
                "objective": hour_result.get("objective"),
                "objective_components": hour_result.get("objective_components"),
                "solve_time_sec": elapsed,
                "Vdev_total": outputs.get("Vdev_total"),
                "Vworst": outputs.get("Vworst"),
                "WorstI": outputs.get("WorstI"),
                "safety_slack": hour_result.get("safety_slack"),
                "q_usage_regularization": hour_result.get("q_usage_regularization"),
                "q_usage_penalty": hour_result.get("q_usage_penalty"),
            }
        )

    return {
        "status": 2 if all_feasible else 9,
        "status_name": "HOURLY_DECOMPOSED_OPTIMAL" if all_feasible else "HOURLY_DECOMPOSED_PARTIAL",
        "objective": float(surrogate_objective) if all_feasible else None,
        "objective_mode": args.objective,
        "objective_source": "surrogate_exp19_hourly_sum_before_exact_pf",
        "loss_objective_model": "none",
        "surrogate_loss_weight": float(args.surrogate_loss_weight),
        "voltage_weight": float(args.voltage_weight),
        "q_weight": float(args.q_weight),
        "solve_time_sec": float(solve_time),
        "data": str(day.data_path),
        "engine": str(resolve_repo_path(args.engine)),
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
        "gurobi_search": {
            "hourly_time_limit": float(args.hourly_time_limit),
            "hourly_mip_gap": float(args.hourly_mip_gap),
            "warm_start": "zero-Q PV/ESS, zero ESS active power, zero-Q reactive-device baseline",
        },
    }


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
    parser = argparse.ArgumentParser(description="24-hour Exp19 SGCN-MILP reactive power optimization.")
    parser.add_argument("--data", default="data/ieee33_nodal_pq_correlated_raw_pool_50k.pt")
    parser.add_argument("--engine", default=str(exp19_rpo.DEFAULT_ENGINE_PATH))
    parser.add_argument("--out", default="rpo_milp/exp19_milp/results/day_ahead_24h_result.json")
    parser.add_argument("--solve-mode", choices=["hourly"], default="hourly")
    parser.add_argument("--hourly-time-limit", type=float, default=120.0)
    parser.add_argument("--hourly-mip-gap", type=float, default=0.02)
    parser.add_argument("--objective", choices=["vdev_q", "vdev"], default="vdev_q")
    parser.add_argument("--surrogate-loss-weight", type=float, default=0.5)
    parser.add_argument("--surrogate-loss-consistency-rel", type=float, default=0.5)
    parser.add_argument("--surrogate-loss-consistency-abs", type=float, default=0.002)
    parser.add_argument("--loss-weight", type=float, default=0.0)
    parser.add_argument("--voltage-weight", type=float, default=1.0)
    parser.add_argument("--q-weight", type=float, default=1e-3)
    parser.add_argument("--trust-region-fraction", type=float, default=1.0)
    parser.add_argument(
        "--control-deviation-penalty",
        "--q-movement-penalty",
        dest="control_deviation_penalty",
        type=float,
        default=0.0,
    )
    parser.add_argument("--voltage-margin", type=float, default=0.0)
    parser.add_argument("--current-margin", type=float, default=0.0)
    parser.add_argument("--base-kv", type=float, default=12.66)
    parser.add_argument("--base-mva", type=float, default=1.0)
    parser.add_argument("--relu-formulation", choices=["big_m", "general"], default="big_m")
    parser.add_argument("--big-m-scale", type=float, default=1.0)
    parser.add_argument("--fallback-big-m", type=float, default=1e3)
    parser.add_argument("--dual-reductions", type=int, choices=[0, 1], default=0)
    parser.add_argument("--diagnose-slack", action="store_true")
    parser.add_argument("--slack-penalty", type=float, default=1e4)
    parser.add_argument("--no-physical-output-bounds", action="store_true")
    parser.add_argument("--ess-energy-hours", type=float, default=2.0)
    parser.add_argument("--soc-initial", type=float, default=0.5)
    parser.add_argument("--soc-min", type=float, default=0.1)
    parser.add_argument("--soc-max", type=float, default=0.9)
    parser.add_argument("--dt-hours", type=float, default=1.0)
    parser.add_argument("--exact-pf-max-iter", type=int, default=200)
    parser.add_argument("--exact-slack-vm-pu", type=float, default=None)
    parser.add_argument("--exact-base-kv", type=float, default=None)
    parser.add_argument("--exact-base-mva", type=float, default=None)
    parser.add_argument("--exact-line-max-i-ka", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    day = load_day_ahead_data(args.data)
    print("Loaded 24h profile data:")
    print(f"  data: {day.data_path}")
    print(f"  PV buses: {day.pv_nodes.tolist()}")
    print(f"  ESS buses: {day.ess_nodes.tolist()}")
    print(f"  Q devices: {control_labels(day)[day.pv_nodes.size + day.ess_nodes.size:]}")
    print(f"  Exp19 objective: {args.objective}, q_weight={args.q_weight}")

    result = optimize_24h_hourly_decomposed(day, args)
    result = attach_exact_power_flow_results(day, result, args)

    out_path = resolve_repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(result), f, indent=2)

    print()
    print("24h Exp19 SGCN-MILP RPO complete.")
    print(f"Status: {result['status_name']} ({result['status']})")
    print(f"Solve time: {result['solve_time_sec']:.3f} s")
    if result.get("surrogate_objective") is not None:
        print(f"Surrogate objective: {result['surrogate_objective']:.8f}")
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
