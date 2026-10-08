"""Compare model/mlp MLP-MILP voltage-deviation RPO with DistFlow.

The MLP MILP optimizes controllable reactive injections through the surrogate
exported by ``model/mlp/train_mlp.py``. The surrogate safety constraints are
``Vworst <= voltage_margin`` and ``WorstI <= current_margin``. This script
reuses the single-step DistFlow evaluation and plotting utilities from the
Exp17 comparison workflow, while replacing the surrogate optimizer with
``rpo_milp.mlp_milp.single_step_rpo``.
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


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rpo_milp.mlp_milp import single_step_rpo as mlp_rpo


DEFAULT_OUT_DIR = REPO_ROOT / "rpo_milp" / "disflow" / "results" / "mlp_single_step"


def get_base_compare():
    """Load the Exp17 comparison helpers only when a run needs them."""

    try:
        from rpo_milp.disflow.single_time_opf import compare_exp17_single_step as base_compare
    except ImportError as exc:
        raise ImportError(
            "compare_mlp_single_step.py reuses plotting/DistFlow helpers from "
            "compare_exp17_single_step.py. Importing those helpers failed; check "
            "the matplotlib/numpy environment used for the existing compare scripts."
        ) from exc

    # Route helper-level RPO calls through the MLP module and make relative
    # paths resolve against the real repository root.
    base_compare.exp17_rpo = mlp_rpo
    base_compare.REPO_ROOT = REPO_ROOT
    return base_compare


def run_milp(op, args) -> dict:
    return mlp_rpo.run_single_step_rpo(
        op.p_load,
        op.q_load,
        op.p_pv,
        op.topo_mask,
        pv_nodes=op.pv_nodes,
        s_rated=op.s_rated,
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
        output_flag=0 if args.quiet_milp else 1,
        export_lp=args.export_lp,
        iis_path=args.iis,
        x_base=op.x_base,
        pv_q_base=op.pv_q_base if op.x_base is not None else None,
        ess_nodes=op.ess_nodes,
        ess_s_rated=op.ess_s_rated,
        ess_p_base=op.ess_p_base,
        ess_q_base=op.ess_q_base,
        q_device_nodes=op.q_device_nodes,
        q_device_min=op.q_device_min,
        q_device_max=op.q_device_max,
        q_device_q_base=op.q_device_q_base,
        q_device_names=op.q_device_names,
    )


def _set_parser_default(parser: argparse.ArgumentParser, dest: str, value) -> None:
    parser.set_defaults(**{dest: value})
    for action in parser._actions:
        if action.dest == dest:
            action.default = value
            return


def _set_parser_help(parser: argparse.ArgumentParser, dest: str, help_text: str) -> None:
    for action in parser._actions:
        if action.dest == dest:
            action.help = help_text
            return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run MLP-MILP RPO, solve exact DistFlow RPO, and write exact "
            "voltage-deviation comparison plots."
        )
    )
    parser.add_argument("--engine", default=str(mlp_rpo.DEFAULT_ENGINE_PATH))
    parser.add_argument(
        "--source",
        choices=["dataset", "profile"],
        default="dataset",
    )
    parser.add_argument("--data", default=str(mlp_rpo.DEFAULT_DATA_PATH))
    parser.add_argument("--sample-index", type=int, default=-1)
    parser.add_argument("--pv-pu", type=float, default=0.8)
    parser.add_argument("--p-load", default=None)
    parser.add_argument("--q-load", default=None)
    parser.add_argument("--p-pv", default=None)
    parser.add_argument(
        "--objective",
        choices=["vdev"],
        default="vdev",
        help="Compare the Exp17-style Vdev objective; exact DistFlow baseline supports this objective.",
    )
    parser.add_argument("--loss-objective-model", choices=["none"], default="none")
    parser.add_argument("--surrogate-loss-weight", type=float, default=0.5)
    parser.add_argument("--surrogate-loss-consistency-rel", type=float, default=0.5)
    parser.add_argument("--surrogate-loss-consistency-abs", type=float, default=0.002)
    parser.add_argument("--loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--voltage-weight",
        type=float,
        default=1e-3,
        help="Unused by the default MLP MILP vdev objective.",
    )
    parser.add_argument(
        "--lambda-q",
        type=float,
        default=0.0,
        help="Unused for the MLP voltage-only objective; must remain 0.",
    )
    parser.add_argument("--v-lower", type=float, default=0.95)
    parser.add_argument("--v-upper", type=float, default=1.05)
    parser.add_argument("--slack-vm-pu", type=float, default=1.03)
    parser.add_argument("--base-kv", type=float, default=12.66)
    parser.add_argument("--base-mva", type=float, default=1.0)
    parser.add_argument("--line-max-i-ka", type=float, default=None)
    parser.add_argument("--default-line-max-i-ka", type=float, default=0.20)
    parser.add_argument(
        "--skip-current-constraints",
        action="store_true",
        help="Do not constrain branch current in the exact DistFlow optimizer.",
    )
    parser.add_argument("--distflow-max-iter", type=int, default=250)
    parser.add_argument("--distflow-pf-max-iter", type=int, default=200)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--skip-milp",
        action="store_true",
        help="Only run baseline and exact DistFlow optimization.",
    )
    parser.add_argument(
        "--milp-q",
        default=None,
        help=(
            "JSON list or JSON file containing a precomputed MLP-MILP full "
            "Q-control dispatch: PV, ESS, then Q devices."
        ),
    )
    parser.add_argument("--voltage-margin", type=float, default=0.0)
    parser.add_argument("--current-margin", type=float, default=0.0)
    parser.add_argument("--trust-region-fraction", type=float, default=1.0)
    parser.add_argument("--control-deviation-penalty", type=float, default=0.0)
    parser.add_argument("--no-physical-output-bounds", action="store_true")
    parser.add_argument(
        "--relu-formulation",
        choices=["big_m", "general"],
        default="big_m",
    )
    parser.add_argument("--big-m-scale", type=float, default=1.0)
    parser.add_argument("--fallback-big-m", type=float, default=1e3)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--dual-reductions", type=int, choices=[0, 1], default=0)
    parser.add_argument("--diagnose-slack", action="store_true")
    parser.add_argument("--slack-penalty", type=float, default=1e4)
    parser.add_argument("--quiet-milp", action="store_true")
    parser.add_argument("--export-lp", default=None)
    parser.add_argument("--iis", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_compare = get_base_compare()
    resolve_repo_path = base_compare.resolve_repo_path
    get_line_max_i_ka = base_compare.get_line_max_i_ka
    load_operating_point = base_compare.load_operating_point
    ExactDistFlow = base_compare.ExactDistFlow
    RADIAL_BRANCHES = base_compare.RADIAL_BRANCHES
    q_capacity = base_compare.q_capacity
    baseline_control = base_compare.baseline_control
    build_net_injection = base_compare.build_net_injection
    control_labels = base_compare.control_labels
    solve_exact_distflow_rpo = base_compare.solve_exact_distflow_rpo
    make_plots = base_compare.make_plots
    summarize_metrics = base_compare.summarize_metrics
    save_csv = base_compare.save_csv
    jsonable = base_compare.jsonable

    op = load_operating_point(args)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    line_max_i_ka = (
        get_line_max_i_ka(args.data, args.default_line_max_i_ka)
        if args.line_max_i_ka is None
        else float(args.line_max_i_ka)
    )
    distflow = ExactDistFlow(
        branches=op.branches if op.branches is not None else RADIAL_BRANCHES,
        slack_vm_pu=args.slack_vm_pu,
        base_kv=args.base_kv,
        base_mva=args.base_mva,
        line_max_i_ka=line_max_i_ka,
    )

    q_caps = q_capacity(op.p_pv, op.s_rated)
    baseline_q = baseline_control(op)
    cases = [
        (
            "Baseline",
            baseline_q,
            distflow.solve(
                build_net_injection(op, baseline_q),
                max_iter=args.distflow_pf_max_iter,
            ),
        )
    ]

    milp_result = None
    milp_solve_time_sec = None
    milp_q = mlp_rpo.parse_vector_arg(
        args.milp_q,
        expected=len(control_labels(op)),
        name="milp_q",
    )
    if milp_q is None and not args.skip_milp:
        milp_start = time.perf_counter()
        milp_result = run_milp(op, args)
        milp_solve_time_sec = time.perf_counter() - milp_start
        if milp_result.get("Q_control_opt") is None:
            raise RuntimeError(
                "MLP-MILP did not return a feasible Q_control_opt; rerun with "
                "--diagnose-slack or pass --milp-q with a precomputed dispatch."
            )
        milp_q = np.asarray(milp_result["Q_control_opt"], dtype=float)

    if milp_q is not None:
        cases.append(
            (
                "MLP-MILP",
                milp_q,
                distflow.solve(
                    build_net_injection(op, milp_q),
                    max_iter=args.distflow_pf_max_iter,
                ),
            )
        )

    distflow_start = time.perf_counter()
    exact = solve_exact_distflow_rpo(
        op,
        distflow,
        objective=args.objective,
        loss_weight=args.loss_weight,
        voltage_weight=args.voltage_weight,
        v_lower=args.v_lower,
        v_upper=args.v_upper,
        enforce_current=not args.skip_current_constraints,
        max_iter=args.distflow_max_iter,
    )
    distflow_solve_time_sec = time.perf_counter() - distflow_start
    cases.append(("Exact DistFlow", exact["Q_control_opt"], exact["metrics"]))

    make_plots(out_dir, cases, op=op, q_caps=q_caps)
    rows = [summarize_metrics(name, q, metrics, op) for name, q, metrics in cases]
    save_csv(out_dir / "comparison_summary.csv", rows)

    payload = {
        "source": args.source,
        "data": str(resolve_repo_path(args.data)) if args.source == "dataset" else None,
        "sample_index": op.meta.get("sample_index"),
        "sample_reason": op.meta.get("sample_reason"),
        "hour": op.meta.get("hour"),
        "mode": op.meta.get("mode"),
        "objective": args.objective,
        "loss_weight": float(args.loss_weight),
        "voltage_weight": float(args.voltage_weight),
        "line_max_i_ka": float(line_max_i_ka),
        "solve_time_sec": {
            "mlp_milp": milp_solve_time_sec,
            "exact_distflow_rpo": distflow_solve_time_sec,
        },
        "pv_nodes": op.pv_nodes.tolist(),
        "p_pv": op.p_pv.tolist(),
        "s_rated": op.s_rated.tolist(),
        "control_labels": control_labels(op),
        "baseline_definition": (
            "PV reactive output is set to zero before optimization; ESS and "
            "Q devices start from their operating-point values."
        ),
        "ess_nodes": op.ess_nodes.tolist(),
        "ess_s_rated": op.ess_s_rated.tolist(),
        "ess_p_base": op.ess_p_base.tolist(),
        "ess_q_base": op.ess_q_base.tolist(),
        "q_device_nodes": op.q_device_nodes.tolist(),
        "q_device_names": op.q_device_names,
        "q_device_min": op.q_device_min.tolist(),
        "q_device_max": op.q_device_max.tolist(),
        "q_device_q_base": op.q_device_q_base.tolist(),
        "milp_result": milp_result,
        "exact_distflow_solver": {
            "success": exact["success"],
            "message": exact["message"],
            "objective": exact["objective"],
            "max_constraint_violation": exact["max_constraint_violation"],
            "optimizer": exact.get("optimizer", "gurobi"),
        },
        "summary": rows,
    }
    with (out_dir / "comparison_result.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2)

    print()
    print("Exact voltage-deviation comparison complete.")
    print(f"Output directory: {out_dir}")
    print(f"Voltage plot: {out_dir / 'voltage_comparison.png'}")
    print(f"Q dispatch plot: {out_dir / 'q_dispatch_comparison.png'}")
    print(f"Summary CSV: {out_dir / 'comparison_summary.csv'}")
    print("Solve time:")
    if milp_solve_time_sec is None:
        print("  MLP-MILP: skipped/precomputed")
    else:
        print(f"  MLP-MILP: {milp_solve_time_sec:.3f} s")
    print(f"  Exact DistFlow RPO: {distflow_solve_time_sec:.3f} s")
    print()
    for row in rows:
        print(
            f"{row['name']}: "
            f"Vmin={row['v_min']:.6f}, Vmax={row['v_max']:.6f}, "
            f"Vdev={row['vdev_total']:.6f}, "
            f"WorstI={row['worst_i_margin']:.6f}"
        )


if __name__ == "__main__":
    main()
