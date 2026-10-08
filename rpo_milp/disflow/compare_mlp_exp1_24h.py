"""Compare saved 24-hour model/mlp_exp1 MLP-MILP and DistFlow RPO JSON files."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rpo_milp.disflow import compare_mlp_exp2_24h as base


HORIZON = 24
DEFAULT_MLP_RESULT = REPO_ROOT / "rpo_milp" / "mlp_exp1_milp" / "results" / "day_ahead_24h_result.json"
DEFAULT_DISFLOW_RESULT = REPO_ROOT / "rpo_milp" / "disflow" / "results" / "day_ahead_24h" / "result.json"
DEFAULT_OUT_DIR = REPO_ROOT / "rpo_milp" / "disflow" / "results" / "mlp_exp1_24h_compare"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot saved 24h model/mlp_exp1 MLP-MILP RPO vs saved exact DistFlow RPO results."
    )
    parser.add_argument("--mlp-result", default=str(DEFAULT_MLP_RESULT))
    parser.add_argument("--disflow-result", default=str(DEFAULT_DISFLOW_RESULT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--slack-vm-pu", type=float, default=1.03)
    parser.add_argument("--base-kv", type=float, default=12.66)
    parser.add_argument("--base-mva", type=float, default=1.0)
    parser.add_argument("--line-max-i-ka", type=float, default=0.20)
    parser.add_argument("--distflow-pf-max-iter", type=int, default=200)
    parser.add_argument("--selected-hours", default="0,6,12,18,23")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = base.resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mlp_payload, mlp_path = base.load_json(args.mlp_result, label="MLP Exp1 24h result")
    disflow_payload_raw, disflow_path = base.load_json(args.disflow_result, label="DistFlow 24h result")
    disflow_payload = base.unwrap_disflow_payload(disflow_payload_raw)

    mlp_dispatch = base.extract_mlp_dispatch(mlp_payload, horizon=HORIZON)
    disflow_dispatch = base.extract_disflow_dispatch(disflow_payload, horizon=HORIZON)
    mlp_voltage, mlp_loss = base.evaluate_dispatch_voltage(
        mlp_payload,
        disflow_payload,
        mlp_dispatch,
        slack_vm_pu=args.slack_vm_pu,
        base_kv=args.base_kv,
        base_mva=args.base_mva,
        line_max_i_ka=args.line_max_i_ka,
        pf_max_iter=args.distflow_pf_max_iter,
    )
    disflow_voltage = base.extract_disflow_voltage(disflow_payload, horizon=HORIZON)
    disflow_loss = base.extract_disflow_loss(disflow_payload, horizon=HORIZON)
    labels = base.control_labels(mlp_payload, disflow_payload, mlp_dispatch)

    rows = base.summarize_hourly(mlp_voltage, disflow_voltage, mlp_loss, disflow_loss)
    base.write_csv(out_dir / "comparison_24h_summary.csv", rows)

    paths = {
        "voltage_envelope": base.plot_voltage_envelope(out_dir, mlp_voltage, disflow_voltage),
        "vdev_loss": base.plot_vdev_loss(out_dir, mlp_voltage, disflow_voltage, mlp_loss, disflow_loss),
        "q_dispatch": base.plot_q_dispatch(out_dir, labels, mlp_dispatch, disflow_dispatch),
        "selected_voltage_profiles": base.plot_selected_voltage_profiles(
            out_dir,
            mlp_voltage,
            disflow_voltage,
            base.parse_hours(args.selected_hours),
        ),
    }

    payload = {
        "mlp_result_source": str(mlp_path),
        "disflow_result_source": str(disflow_path),
        "out_dir": str(out_dir),
        "control_labels": labels,
        "solve_time_sec": {
            "mlp_exp1_recorded_in_result": mlp_payload.get("solve_time_sec"),
            "disflow_recorded_in_result": disflow_payload.get("solve_time_sec"),
        },
        "voltage_source": {
            "mlp_exp1": "exact DistFlow power-flow evaluation of Q_pv_opt/Q_ess_opt/Q_device_opt",
            "disflow": "saved exact DistFlow optimization metrics[*].voltage_pu",
        },
        "mlp_exp1_dispatch": mlp_dispatch,
        "disflow_dispatch": disflow_dispatch,
        "mlp_exp1_exact_distflow_eval": {
            "voltage_pu": mlp_voltage,
            "total_loss_mw": mlp_loss,
        },
        "summary": rows,
        "plot_paths": {key: str(value) for key, value in paths.items()},
    }
    base.save_json(out_dir / "comparison_24h_result.json", payload)

    print()
    print("24h saved MLP Exp1-MILP vs saved DistFlow comparison complete.")
    print(f"MLP Exp1 result: {mlp_path}")
    print(f"DistFlow result: {disflow_path}")
    print(f"Output directory: {out_dir}")
    print("Recorded optimization solve time:")
    mlp_time = mlp_payload.get("solve_time_sec")
    disflow_time = disflow_payload.get("solve_time_sec")
    print(
        f"  model/mlp_exp1 MLP-MILP 24h: {float(mlp_time):.3f} s"
        if mlp_time is not None
        else "  model/mlp_exp1 MLP-MILP 24h: unavailable"
    )
    print(
        f"  Exact DistFlow Gurobi QCP 24h: {float(disflow_time):.3f} s"
        if disflow_time is not None
        else "  Exact DistFlow Gurobi QCP 24h: unavailable"
    )
    for key, path in paths.items():
        print(f"  {key}: {path}")
    print(f"  summary: {out_dir / 'comparison_24h_summary.csv'}")
    print(f"  json: {out_dir / 'comparison_24h_result.json'}")


if __name__ == "__main__":
    main()
