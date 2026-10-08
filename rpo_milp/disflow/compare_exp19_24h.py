"""Plot a 24-hour comparison from saved Exp19 and DistFlow RPO JSON files.

This script does not run either optimizer. It reads:

* rpo_milp/exp19_milp/results/day_ahead_24h_result.json
* rpo_milp/disflow/results/day_ahead_24h/result.json

The Exp19 day-ahead script saves exact PF outputs after optimization. This
comparison uses those saved physical voltage/loss values when present, and
falls back to recomputing them from the saved Exp19 Q dispatch with the same
exact DistFlow power-flow routine used by the DistFlow script.

Comparison plots and summaries are written under
``rpo_milp/disflow/results/exp19_24h_compare``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rpo_milp.disflow import compare_exp17_24h as base


HORIZON = 24
DEFAULT_EXP19_RESULT = REPO_ROOT / "rpo_milp" / "exp19_milp" / "results" / "day_ahead_24h_result.json"
DEFAULT_DISFLOW_RESULT = REPO_ROOT / "rpo_milp" / "disflow" / "results" / "day_ahead_24h" / "result.json"
DEFAULT_OUT_DIR = REPO_ROOT / "rpo_milp" / "disflow" / "results" / "exp19_24h_compare"


def get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def extract_exp19_dispatch(payload: dict, *, horizon: int) -> dict[str, np.ndarray]:
    q_ess = base.as_2d(payload.get("Q_ess_opt"), horizon=horizon, name="Q_ess_opt")
    p_ch = (
        base.as_2d(payload.get("P_ess_ch_opt"), horizon=horizon, name="P_ess_ch_opt")
        if payload.get("P_ess_ch_opt") is not None
        else np.zeros_like(q_ess)
    )
    p_dis = (
        base.as_2d(payload.get("P_ess_dis_opt"), horizon=horizon, name="P_ess_dis_opt")
        if payload.get("P_ess_dis_opt") is not None
        else np.zeros_like(q_ess)
    )
    return {
        "q_pv": base.as_2d(payload.get("Q_pv_opt"), horizon=horizon, name="Q_pv_opt"),
        "q_ess": q_ess,
        "q_device": base.as_2d(payload.get("Q_device_opt"), horizon=horizon, name="Q_device_opt"),
        "p_ch": p_ch,
        "p_dis": p_dis,
    }


def summarize_hourly(
    exp19_voltage: np.ndarray,
    disflow_voltage: np.ndarray,
    exp19_loss: np.ndarray,
    disflow_loss: np.ndarray,
) -> list[dict]:
    exp19 = base.voltage_summary(exp19_voltage)
    disflow = base.voltage_summary(disflow_voltage)
    rows = []
    for hour in range(HORIZON):
        rows.append(
            {
                "hour": hour,
                "exp19_v_min": float(exp19["v_min"][hour]),
                "exp19_v_max": float(exp19["v_max"][hour]),
                "exp19_v_mean": float(exp19["v_mean"][hour]),
                "exp19_vdev_total": float(exp19["vdev_total"][hour]),
                "exp19_total_loss_mw": float(exp19_loss[hour]),
                "disflow_v_min": float(disflow["v_min"][hour]),
                "disflow_v_max": float(disflow["v_max"][hour]),
                "disflow_v_mean": float(disflow["v_mean"][hour]),
                "disflow_vdev_total": float(disflow["vdev_total"][hour]),
                "disflow_total_loss_mw": float(disflow_loss[hour]),
                "delta_vdev_total": float(exp19["vdev_total"][hour] - disflow["vdev_total"][hour]),
                "delta_total_loss_mw": float(exp19_loss[hour] - disflow_loss[hour]),
            }
        )
    return rows


def extract_saved_exp19_exact_pf(payload: dict, *, horizon: int) -> tuple[np.ndarray, np.ndarray] | None:
    outputs = payload.get("exact_pf_outputs")
    if isinstance(outputs, list) and len(outputs) == horizon:
        voltage_rows = []
        loss_values = []
        for item in outputs:
            if not isinstance(item, dict) or item.get("V_nodes") is None or item.get("Ploss_total") is None:
                return None
            voltage = np.asarray(item["V_nodes"], dtype=float).reshape(-1)
            if voltage.size != 33:
                return None
            voltage_rows.append(voltage)
            loss_values.append(float(item["Ploss_total"]))
        return np.vstack(voltage_rows), np.asarray(loss_values, dtype=float)

    metrics = payload.get("exact_pf_metrics")
    if isinstance(metrics, list) and len(metrics) == horizon:
        voltage_rows = []
        loss_values = []
        for item in metrics:
            if not isinstance(item, dict) or item.get("voltage_pu") is None or item.get("total_loss_mw") is None:
                return None
            voltage = np.asarray(item["voltage_pu"], dtype=float).reshape(-1)
            if voltage.size != 33:
                return None
            voltage_rows.append(voltage)
            loss_values.append(float(item["total_loss_mw"]))
        return np.vstack(voltage_rows), np.asarray(loss_values, dtype=float)

    return None


def plot_voltage_envelope(out_dir: Path, exp19_voltage: np.ndarray, disflow_voltage: np.ndarray) -> Path:
    plt = get_pyplot()
    hours = np.arange(HORIZON)
    exp19 = base.voltage_summary(exp19_voltage)
    disflow = base.voltage_summary(disflow_voltage)

    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=160)
    ax.fill_between(hours, exp19["v_min"], exp19["v_max"], color="#4C78A8", alpha=0.12, label="Exp19 Q exact PF range")
    ax.fill_between(hours, disflow["v_min"], disflow["v_max"], color="#F58518", alpha=0.12, label="DistFlow range")
    ax.plot(hours, exp19["v_min"], color="#4C78A8", marker="o", linewidth=1.7, label="Exp19 Q exact PF min")
    ax.plot(hours, exp19["v_max"], color="#4C78A8", marker="s", linewidth=1.7, linestyle="--", label="Exp19 Q exact PF max")
    ax.plot(hours, exp19["v_mean"], color="#4C78A8", marker="^", linewidth=1.5, linestyle=":", label="Exp19 Q exact PF mean")
    ax.plot(hours, disflow["v_min"], color="#F58518", marker="o", linewidth=1.7, label="DistFlow min")
    ax.plot(hours, disflow["v_max"], color="#F58518", marker="s", linewidth=1.7, linestyle="--", label="DistFlow max")
    ax.plot(hours, disflow["v_mean"], color="#F58518", marker="^", linewidth=1.5, linestyle=":", label="DistFlow mean")
    ax.axhline(0.95, color="tab:red", linestyle="--", linewidth=1.0)
    ax.axhline(1.05, color="tab:red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title("24h voltage envelope: Exp19 Q exact PF vs exact DistFlow RPO")
    ax.set_xticks(hours)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    path = out_dir / "compare_24h_voltage_envelope.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_vdev_loss(
    out_dir: Path,
    exp19_voltage: np.ndarray,
    disflow_voltage: np.ndarray,
    exp19_loss: np.ndarray,
    disflow_loss: np.ndarray,
) -> Path:
    plt = get_pyplot()
    hours = np.arange(HORIZON)
    exp19_vdev = base.voltage_summary(exp19_voltage)["vdev_total"]
    disflow_vdev = base.voltage_summary(disflow_voltage)["vdev_total"]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), dpi=160, sharex=True)
    axes[0].plot(hours, exp19_vdev, marker="o", linewidth=1.8, label="Exp19 Q exact PF")
    axes[0].plot(hours, disflow_vdev, marker="s", linewidth=1.8, label="Exact DistFlow saved result")
    axes[0].set_ylabel("Vdev total")
    axes[0].set_title("Hourly voltage-deviation comparison")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(hours, exp19_loss, marker="o", linewidth=1.8, color="#4C78A8", label="Exp19 Q exact PF loss")
    axes[1].plot(hours, disflow_loss, marker="s", linewidth=1.8, color="#F58518", label="Exact DistFlow loss")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Loss (MW)")
    axes[1].set_title("Hourly exact DistFlow loss comparison")
    axes[1].set_xticks(hours)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")
    fig.tight_layout()
    path = out_dir / "compare_24h_vdev_loss.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_q_dispatch(
    out_dir: Path,
    labels: list[str],
    exp19_dispatch: dict[str, np.ndarray],
    disflow_dispatch: dict[str, np.ndarray],
) -> Path:
    plt = get_pyplot()
    hours = np.arange(HORIZON)
    exp19_q = base.dispatch_matrix(exp19_dispatch)
    disflow_q = base.dispatch_matrix(disflow_dispatch)
    if exp19_q.shape != disflow_q.shape:
        raise ValueError(f"Q dispatch shapes differ: Exp19 {exp19_q.shape}, DistFlow {disflow_q.shape}")

    n_pv = exp19_dispatch["q_pv"].shape[1]
    n_ess = exp19_dispatch["q_ess"].shape[1]
    groups = [
        ("PV Q", 0, n_pv),
        ("ESS Q", n_pv, n_pv + n_ess),
        ("Q-device Q", n_pv + n_ess, exp19_q.shape[1]),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.0), dpi=160, sharex=True)
    color_cycle = plt.cm.tab10(np.linspace(0.0, 1.0, max(exp19_q.shape[1], 10)))
    for ax, (title, start, end) in zip(axes, groups):
        if start == end:
            ax.text(0.5, 0.5, "No devices", transform=ax.transAxes, ha="center", va="center")
        for idx in range(start, end):
            color = color_cycle[idx % len(color_cycle)]
            ax.plot(hours, exp19_q[:, idx], color=color, linewidth=1.7, marker="o", markersize=3, label=f"Exp19 {labels[idx]}")
            ax.plot(
                hours,
                disflow_q[:, idx],
                color=color,
                linewidth=1.7,
                linestyle="--",
                marker="s",
                markersize=3,
                label=f"DistFlow {labels[idx]}",
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Q (MVar)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", ncol=3, fontsize=7)
    axes[-1].set_xlabel("Hour")
    axes[-1].set_xticks(hours)
    fig.tight_layout()
    path = out_dir / "compare_24h_q_dispatch.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_selected_voltage_profiles(
    out_dir: Path,
    exp19_voltage: np.ndarray,
    disflow_voltage: np.ndarray,
    selected_hours: list[int],
) -> Path:
    plt = get_pyplot()
    bus = np.arange(2, 34)
    selected = [hour for hour in selected_hours if 0 <= hour < HORIZON]
    if not selected:
        selected = [0, 6, 12, 18, 23]

    fig, axes = plt.subplots(len(selected), 1, figsize=(11.5, 2.3 * len(selected)), dpi=160, sharex=True)
    if len(selected) == 1:
        axes = [axes]
    for ax, hour in zip(axes, selected):
        ax.plot(bus, exp19_voltage[hour, 1:], marker="o", linewidth=1.6, label="Exp19 Q exact PF")
        ax.plot(bus, disflow_voltage[hour, 1:], marker="s", linewidth=1.6, linestyle="--", label="DistFlow saved result")
        ax.axhline(0.95, color="tab:red", linestyle="--", linewidth=1.0)
        ax.axhline(1.05, color="tab:red", linestyle="--", linewidth=1.0)
        ax.set_ylabel("V (p.u.)")
        ax.set_title(f"Hour {hour:02d} non-slack nodal voltage")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Bus")
    axes[-1].set_xticks(np.arange(2, 34, 2))
    fig.tight_layout()
    path = out_dir / "compare_24h_selected_voltage_profiles.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot saved 24h Exp19 RPO vs saved exact DistFlow RPO results.")
    parser.add_argument("--exp19-result", default=str(DEFAULT_EXP19_RESULT))
    parser.add_argument("--disflow-result", default=str(DEFAULT_DISFLOW_RESULT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--slack-vm-pu", type=float, default=1.03)
    parser.add_argument("--base-kv", type=float, default=12.66)
    parser.add_argument("--base-mva", type=float, default=1.0)
    parser.add_argument("--line-max-i-ka", type=float, default=0.20)
    parser.add_argument("--distflow-pf-max-iter", type=int, default=200)
    parser.add_argument("--selected-hours", default="0,6,12,18,23")
    parser.add_argument(
        "--recompute-exp19-pf",
        action="store_true",
        help="Ignore saved exact_pf_outputs and recompute Exp19 dispatch with exact DistFlow.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = base.resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp19_payload, exp19_path = base.load_json(args.exp19_result, label="Exp19 24h result")
    disflow_payload_raw, disflow_path = base.load_json(args.disflow_result, label="DistFlow 24h result")
    disflow_payload = base.unwrap_disflow_payload(disflow_payload_raw)

    exp19_dispatch = extract_exp19_dispatch(exp19_payload, horizon=HORIZON)
    disflow_dispatch = base.extract_disflow_dispatch(disflow_payload, horizon=HORIZON)
    saved_exp19_eval = None if args.recompute_exp19_pf else extract_saved_exp19_exact_pf(exp19_payload, horizon=HORIZON)
    exp19_eval_source = "saved Exp19 exact_pf_outputs/exact_pf_metrics"
    if saved_exp19_eval is None:
        exp19_voltage, exp19_loss = base.evaluate_dispatch_voltage(
            exp19_payload,
            disflow_payload,
            exp19_dispatch,
            slack_vm_pu=args.slack_vm_pu,
            base_kv=args.base_kv,
            base_mva=args.base_mva,
            line_max_i_ka=args.line_max_i_ka,
            pf_max_iter=args.distflow_pf_max_iter,
        )
        exp19_eval_source = "recomputed exact DistFlow power-flow evaluation of Q_pv_opt/Q_ess_opt/Q_device_opt"
    else:
        exp19_voltage, exp19_loss = saved_exp19_eval
    disflow_voltage = base.extract_disflow_voltage(disflow_payload, horizon=HORIZON)
    disflow_loss = base.extract_disflow_loss(disflow_payload, horizon=HORIZON)
    labels = base.control_labels(exp19_payload, disflow_payload, exp19_dispatch)

    rows = summarize_hourly(exp19_voltage, disflow_voltage, exp19_loss, disflow_loss)
    base.write_csv(out_dir / "comparison_24h_summary.csv", rows)

    paths = {
        "voltage_envelope": plot_voltage_envelope(out_dir, exp19_voltage, disflow_voltage),
        "vdev_loss": plot_vdev_loss(out_dir, exp19_voltage, disflow_voltage, exp19_loss, disflow_loss),
        "q_dispatch": plot_q_dispatch(out_dir, labels, exp19_dispatch, disflow_dispatch),
        "selected_voltage_profiles": plot_selected_voltage_profiles(
            out_dir,
            exp19_voltage,
            disflow_voltage,
            base.parse_hours(args.selected_hours),
        ),
    }

    payload = {
        "exp19_result_source": str(exp19_path),
        "disflow_result_source": str(disflow_path),
        "out_dir": str(out_dir),
        "control_labels": labels,
        "solve_time_sec": {
            "exp19_recorded_in_result": exp19_payload.get("solve_time_sec"),
            "disflow_recorded_in_result": disflow_payload.get("solve_time_sec"),
        },
        "objective": {
            "exp19_recorded_objective": exp19_payload.get("objective"),
            "exp19_objective_source": exp19_payload.get("objective_source"),
            "exp19_surrogate_objective": exp19_payload.get("surrogate_objective"),
            "disflow_recorded_objective": disflow_payload.get("objective"),
        },
        "voltage_source": {
            "exp19": exp19_eval_source,
            "disflow": "saved exact DistFlow optimization metrics[*].voltage_pu",
        },
        "exp19_dispatch": exp19_dispatch,
        "disflow_dispatch": disflow_dispatch,
        "exp19_exact_distflow_eval": {
            "voltage_pu": exp19_voltage,
            "total_loss_mw": exp19_loss,
            "total_loss_mw_24h": float(np.sum(exp19_loss)),
        },
        "disflow_saved_eval": {
            "total_loss_mw": disflow_loss,
            "total_loss_mw_24h": float(np.nansum(disflow_loss)),
        },
        "summary": rows,
        "plot_paths": {key: str(value) for key, value in paths.items()},
    }
    base.save_json(out_dir / "comparison_24h_result.json", payload)

    print()
    print("24h saved Exp19 vs saved DistFlow comparison complete.")
    print(f"Exp19 result: {exp19_path}")
    print(f"DistFlow result: {disflow_path}")
    print(f"Output directory: {out_dir}")
    print(f"Exp19 physical evaluation source: {exp19_eval_source}")
    print("Recorded optimization solve time:")
    exp19_time = exp19_payload.get("solve_time_sec")
    disflow_time = disflow_payload.get("solve_time_sec")
    print(
        f"  Exp19 SGCN-MILP 24h: {float(exp19_time):.3f} s"
        if exp19_time is not None
        else "  Exp19 SGCN-MILP 24h: unavailable"
    )
    print(
        f"  Exact DistFlow Gurobi QCP 24h: {float(disflow_time):.3f} s"
        if disflow_time is not None
        else "  Exact DistFlow Gurobi QCP 24h: unavailable"
    )
    print(f"Exact PF total loss from saved Exp19 Q: {float(np.sum(exp19_loss)):.8f} MW")
    print(f"Exact DistFlow saved total loss: {float(np.nansum(disflow_loss)):.8f} MW")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    print(f"  summary: {out_dir / 'comparison_24h_summary.csv'}")
    print(f"  json: {out_dir / 'comparison_24h_result.json'}")


if __name__ == "__main__":
    main()
