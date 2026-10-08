"""Plot a 24-hour comparison from saved Exp17 and DistFlow RPO JSON files.

This script does not run either optimizer. It reads:

* rpo_milp/exp17_milp/results/day_ahead_24h_result.json
* rpo_milp/disflow/results/day_ahead_24h/result.json

Exp17 voltages are not taken from surrogate ``outputs.V_nodes``. The saved
Exp17 Q dispatch is evaluated with the same exact DistFlow power-flow routine
used by the DistFlow script, so both voltage curves use the same physical
calculation.

and writes comparison plots under
``rpo_milp/disflow/results/exp17_24h_compare``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rpo_milp.disflow import day_ahead_24h_distflow_rpo as disflow_24h


HORIZON = 24
DEFAULT_EXP17_RESULT = REPO_ROOT / "rpo_milp" / "exp17_milp" / "results" / "day_ahead_24h_result.json"
DEFAULT_DISFLOW_RESULT = REPO_ROOT / "rpo_milp" / "disflow" / "results" / "day_ahead_24h" / "result.json"
DEFAULT_OUT_DIR = REPO_ROOT / "rpo_milp" / "disflow" / "results" / "exp17_24h_compare"


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def load_json(path: str | Path, *, label: str) -> tuple[dict, Path]:
    json_path = resolve_repo_path(path)
    if not json_path.is_file():
        raise FileNotFoundError(
            f"{label} JSON not found: {json_path}\n"
            "Run the corresponding 24h optimization script first, or pass the correct path."
        )
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"{label} JSON must contain an object: {json_path}")
    return payload, json_path


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2)


def as_2d(value, *, horizon: int, name: str) -> np.ndarray:
    if value is None:
        return np.zeros((horizon, 0), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != horizon:
        raise ValueError(f"{name} must have shape ({horizon}, n), got {arr.shape}")
    return arr


def unwrap_disflow_payload(payload: dict) -> dict:
    nested = payload.get("disflow_result")
    return nested if isinstance(nested, dict) else payload


def extract_exp17_dispatch(payload: dict, *, horizon: int) -> dict[str, np.ndarray]:
    q_ess = as_2d(payload.get("Q_ess_opt"), horizon=horizon, name="Q_ess_opt")
    return {
        "q_pv": as_2d(payload.get("Q_pv_opt"), horizon=horizon, name="Q_pv_opt"),
        "q_ess": q_ess,
        "q_device": as_2d(payload.get("Q_device_opt"), horizon=horizon, name="Q_device_opt"),
        "p_ch": as_2d(payload.get("P_ess_ch_opt"), horizon=horizon, name="P_ess_ch_opt")
        if payload.get("P_ess_ch_opt") is not None
        else np.zeros_like(q_ess),
        "p_dis": as_2d(payload.get("P_ess_dis_opt"), horizon=horizon, name="P_ess_dis_opt")
        if payload.get("P_ess_dis_opt") is not None
        else np.zeros_like(q_ess),
    }


def extract_disflow_dispatch(payload: dict, *, horizon: int) -> dict[str, np.ndarray]:
    dispatch = payload.get("dispatch")
    if not isinstance(dispatch, dict):
        raise ValueError("DistFlow JSON must contain dispatch as an object")
    return {
        "q_pv": as_2d(dispatch.get("q_pv"), horizon=horizon, name="dispatch.q_pv"),
        "q_ess": as_2d(dispatch.get("q_ess"), horizon=horizon, name="dispatch.q_ess"),
        "q_device": as_2d(dispatch.get("q_device"), horizon=horizon, name="dispatch.q_device"),
    }


def dispatch_dict_to_disflow(day, dispatch: dict[str, np.ndarray]) -> disflow_24h.Dispatch:
    q_pv = dispatch["q_pv"]
    q_ess = dispatch["q_ess"]
    q_device = dispatch["q_device"]
    p_ch = dispatch.get("p_ch")
    p_dis = dispatch.get("p_dis")
    if p_ch is None:
        p_ch = np.zeros((HORIZON, day.ess_nodes.size), dtype=float)
    if p_dis is None:
        p_dis = np.zeros((HORIZON, day.ess_nodes.size), dtype=float)
    if q_pv.shape != (HORIZON, day.pv_nodes.size):
        raise ValueError(f"Exp17 q_pv shape {q_pv.shape} does not match dataset PV size {day.pv_nodes.size}")
    if q_ess.shape != (HORIZON, day.ess_nodes.size):
        raise ValueError(f"Exp17 q_ess shape {q_ess.shape} does not match dataset ESS size {day.ess_nodes.size}")
    if q_device.shape != (HORIZON, day.q_device_nodes.size):
        raise ValueError(
            f"Exp17 q_device shape {q_device.shape} does not match dataset Q-device size {day.q_device_nodes.size}"
        )
    return disflow_24h.Dispatch(
        q_pv=q_pv,
        p_ch=p_ch,
        p_dis=p_dis,
        q_ess=q_ess,
        q_device=q_device,
    )


def evaluate_dispatch_voltage(
    exp17_payload: dict,
    disflow_payload: dict,
    exp17_dispatch: dict[str, np.ndarray],
    *,
    slack_vm_pu: float,
    base_kv: float,
    base_mva: float,
    line_max_i_ka: float,
    pf_max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    data_path = exp17_payload.get("data") or disflow_payload.get("data")
    if not data_path:
        raise ValueError("Exp17 or DistFlow JSON must contain a data path to evaluate Exp17 Q by exact DistFlow")

    day = disflow_24h.load_day_ahead_data(data_path)
    distflow = disflow_24h.ExactDistFlow(
        branches=day.branches,
        slack_vm_pu=float(slack_vm_pu),
        base_kv=float(base_kv),
        base_mva=float(base_mva),
        line_max_i_ka=float(line_max_i_ka),
    )
    dispatch = dispatch_dict_to_disflow(day, exp17_dispatch)
    metrics = disflow_24h.evaluate_24h(day, distflow, dispatch, pf_max_iter=pf_max_iter)
    voltage = np.vstack([np.asarray(item.voltage_pu, dtype=float).reshape(-1) for item in metrics])
    loss = np.asarray([float(item.total_loss_mw) for item in metrics], dtype=float)
    return voltage, loss


def extract_disflow_voltage(payload: dict, *, horizon: int) -> np.ndarray:
    metrics = payload.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != horizon:
        raise ValueError(f"DistFlow JSON must contain metrics as a {horizon}-element list")

    voltage_rows = []
    missing = []
    for hour, item in enumerate(metrics):
        row = item.get("voltage_pu") if isinstance(item, dict) else None
        if row is None:
            missing.append(hour)
            continue
        voltage = np.asarray(row, dtype=float).reshape(-1)
        if voltage.size != 33:
            raise ValueError(f"metrics[{hour}].voltage_pu must have length 33, got {voltage.size}")
        voltage_rows.append(voltage)
    if missing:
        raise ValueError(f"DistFlow JSON is missing metrics[*].voltage_pu for hours: {missing}")
    return np.vstack(voltage_rows)


def extract_disflow_loss(payload: dict, *, horizon: int) -> np.ndarray:
    metrics = payload.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != horizon:
        return np.full(horizon, np.nan, dtype=float)
    values = []
    for item in metrics:
        if isinstance(item, dict) and item.get("total_loss_mw") is not None:
            values.append(float(item["total_loss_mw"]))
        else:
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def dispatch_matrix(dispatch: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([dispatch["q_pv"], dispatch["q_ess"], dispatch["q_device"]], axis=1)


def control_labels(exp17_payload: dict, disflow_payload: dict, exp17_dispatch: dict[str, np.ndarray]) -> list[str]:
    expected = dispatch_matrix(exp17_dispatch).shape[1]
    for payload in (exp17_payload, disflow_payload):
        labels = payload.get("control_labels")
        if isinstance(labels, list) and len(labels) == expected:
            return [str(item) for item in labels]

    labels = []
    labels.extend(f"PV{i + 1}" for i in range(exp17_dispatch["q_pv"].shape[1]))
    labels.extend(f"ESS{i + 1}" for i in range(exp17_dispatch["q_ess"].shape[1]))
    labels.extend(f"QDev{i + 1}" for i in range(exp17_dispatch["q_device"].shape[1]))
    return labels


def voltage_summary(voltage: np.ndarray) -> dict[str, np.ndarray]:
    non_slack = voltage[:, 1:]
    return {
        "v_min": non_slack.min(axis=1),
        "v_max": non_slack.max(axis=1),
        "v_mean": non_slack.mean(axis=1),
        "vdev_total": np.abs(non_slack - 1.0).sum(axis=1),
    }


def summarize_hourly(
    exp17_voltage: np.ndarray,
    disflow_voltage: np.ndarray,
    exp17_loss: np.ndarray,
    disflow_loss: np.ndarray,
) -> list[dict]:
    exp17 = voltage_summary(exp17_voltage)
    disflow = voltage_summary(disflow_voltage)
    rows = []
    for hour in range(HORIZON):
        rows.append(
            {
                "hour": hour,
                "exp17_v_min": float(exp17["v_min"][hour]),
                "exp17_v_max": float(exp17["v_max"][hour]),
                "exp17_v_mean": float(exp17["v_mean"][hour]),
                "exp17_vdev_total": float(exp17["vdev_total"][hour]),
                "exp17_total_loss_mw": float(exp17_loss[hour]),
                "disflow_v_min": float(disflow["v_min"][hour]),
                "disflow_v_max": float(disflow["v_max"][hour]),
                "disflow_v_mean": float(disflow["v_mean"][hour]),
                "disflow_vdev_total": float(disflow["vdev_total"][hour]),
                "disflow_total_loss_mw": float(disflow_loss[hour]),
                "delta_vdev_total": float(exp17["vdev_total"][hour] - disflow["vdev_total"][hour]),
                "delta_total_loss_mw": float(exp17_loss[hour] - disflow_loss[hour]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_voltage_envelope(out_dir: Path, exp17_voltage: np.ndarray, disflow_voltage: np.ndarray) -> Path:
    hours = np.arange(HORIZON)
    exp17 = voltage_summary(exp17_voltage)
    disflow = voltage_summary(disflow_voltage)

    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=160)
    ax.fill_between(hours, exp17["v_min"], exp17["v_max"], color="#4C78A8", alpha=0.12, label="Exp17 Q exact PF range")
    ax.fill_between(hours, disflow["v_min"], disflow["v_max"], color="#F58518", alpha=0.12, label="DistFlow range")
    ax.plot(hours, exp17["v_min"], color="#4C78A8", marker="o", linewidth=1.7, label="Exp17 Q exact PF min")
    ax.plot(hours, exp17["v_max"], color="#4C78A8", marker="s", linewidth=1.7, linestyle="--", label="Exp17 Q exact PF max")
    ax.plot(hours, exp17["v_mean"], color="#4C78A8", marker="^", linewidth=1.5, linestyle=":", label="Exp17 Q exact PF mean")
    ax.plot(hours, disflow["v_min"], color="#F58518", marker="o", linewidth=1.7, label="DistFlow min")
    ax.plot(hours, disflow["v_max"], color="#F58518", marker="s", linewidth=1.7, linestyle="--", label="DistFlow max")
    ax.plot(hours, disflow["v_mean"], color="#F58518", marker="^", linewidth=1.5, linestyle=":", label="DistFlow mean")
    ax.axhline(0.95, color="tab:red", linestyle="--", linewidth=1.0)
    ax.axhline(1.05, color="tab:red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title("24h voltage envelope: Exp17 Q exact PF vs exact DistFlow RPO")
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
    exp17_voltage: np.ndarray,
    disflow_voltage: np.ndarray,
    exp17_loss: np.ndarray,
    disflow_loss: np.ndarray,
) -> Path:
    hours = np.arange(HORIZON)
    exp17_vdev = voltage_summary(exp17_voltage)["vdev_total"]
    disflow_vdev = voltage_summary(disflow_voltage)["vdev_total"]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), dpi=160, sharex=True)
    axes[0].plot(hours, exp17_vdev, marker="o", linewidth=1.8, label="Exp17 Q exact PF")
    axes[0].plot(hours, disflow_vdev, marker="s", linewidth=1.8, label="Exact DistFlow saved result")
    axes[0].set_ylabel("Vdev total")
    axes[0].set_title("Hourly voltage-deviation comparison")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(hours, exp17_loss, marker="o", linewidth=1.8, color="#4C78A8", label="Exp17 Q exact PF loss")
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
    exp17_dispatch: dict[str, np.ndarray],
    disflow_dispatch: dict[str, np.ndarray],
) -> Path:
    hours = np.arange(HORIZON)
    exp17_q = dispatch_matrix(exp17_dispatch)
    disflow_q = dispatch_matrix(disflow_dispatch)
    if exp17_q.shape != disflow_q.shape:
        raise ValueError(f"Q dispatch shapes differ: Exp17 {exp17_q.shape}, DistFlow {disflow_q.shape}")

    n_pv = exp17_dispatch["q_pv"].shape[1]
    n_ess = exp17_dispatch["q_ess"].shape[1]
    groups = [
        ("PV Q", 0, n_pv),
        ("ESS Q", n_pv, n_pv + n_ess),
        ("Q-device Q", n_pv + n_ess, exp17_q.shape[1]),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.0), dpi=160, sharex=True)
    color_cycle = plt.cm.tab10(np.linspace(0.0, 1.0, max(exp17_q.shape[1], 10)))
    for ax, (title, start, end) in zip(axes, groups):
        if start == end:
            ax.text(0.5, 0.5, "No devices", transform=ax.transAxes, ha="center", va="center")
        for idx in range(start, end):
            color = color_cycle[idx % len(color_cycle)]
            ax.plot(hours, exp17_q[:, idx], color=color, linewidth=1.7, marker="o", markersize=3, label=f"Exp17 {labels[idx]}")
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
    exp17_voltage: np.ndarray,
    disflow_voltage: np.ndarray,
    selected_hours: list[int],
) -> Path:
    bus = np.arange(2, 34)
    selected = [hour for hour in selected_hours if 0 <= hour < HORIZON]
    if not selected:
        selected = [0, 6, 12, 18, 23]

    fig, axes = plt.subplots(len(selected), 1, figsize=(11.5, 2.3 * len(selected)), dpi=160, sharex=True)
    if len(selected) == 1:
        axes = [axes]
    for ax, hour in zip(axes, selected):
        ax.plot(bus, exp17_voltage[hour, 1:], marker="o", linewidth=1.6, label="Exp17 Q exact PF")
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


def parse_hours(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


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
    parser = argparse.ArgumentParser(description="Plot saved 24h Exp17 RPO vs saved exact DistFlow RPO results.")
    parser.add_argument("--exp17-result", default=str(DEFAULT_EXP17_RESULT))
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
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp17_payload, exp17_path = load_json(args.exp17_result, label="Exp17 24h result")
    disflow_payload_raw, disflow_path = load_json(args.disflow_result, label="DistFlow 24h result")
    disflow_payload = unwrap_disflow_payload(disflow_payload_raw)

    exp17_dispatch = extract_exp17_dispatch(exp17_payload, horizon=HORIZON)
    disflow_dispatch = extract_disflow_dispatch(disflow_payload, horizon=HORIZON)
    exp17_voltage, exp17_loss = evaluate_dispatch_voltage(
        exp17_payload,
        disflow_payload,
        exp17_dispatch,
        slack_vm_pu=args.slack_vm_pu,
        base_kv=args.base_kv,
        base_mva=args.base_mva,
        line_max_i_ka=args.line_max_i_ka,
        pf_max_iter=args.distflow_pf_max_iter,
    )
    disflow_voltage = extract_disflow_voltage(disflow_payload, horizon=HORIZON)
    disflow_loss = extract_disflow_loss(disflow_payload, horizon=HORIZON)
    labels = control_labels(exp17_payload, disflow_payload, exp17_dispatch)

    rows = summarize_hourly(exp17_voltage, disflow_voltage, exp17_loss, disflow_loss)
    write_csv(out_dir / "comparison_24h_summary.csv", rows)

    paths = {
        "voltage_envelope": plot_voltage_envelope(out_dir, exp17_voltage, disflow_voltage),
        "vdev_loss": plot_vdev_loss(out_dir, exp17_voltage, disflow_voltage, exp17_loss, disflow_loss),
        "q_dispatch": plot_q_dispatch(out_dir, labels, exp17_dispatch, disflow_dispatch),
        "selected_voltage_profiles": plot_selected_voltage_profiles(
            out_dir,
            exp17_voltage,
            disflow_voltage,
            parse_hours(args.selected_hours),
        ),
    }

    payload = {
        "exp17_result_source": str(exp17_path),
        "disflow_result_source": str(disflow_path),
        "out_dir": str(out_dir),
        "control_labels": labels,
        "solve_time_sec": {
            "exp17_recorded_in_result": exp17_payload.get("solve_time_sec"),
            "disflow_recorded_in_result": disflow_payload.get("solve_time_sec"),
        },
        "voltage_source": {
            "exp17": "exact DistFlow power-flow evaluation of Q_pv_opt/Q_ess_opt/Q_device_opt",
            "disflow": "saved exact DistFlow optimization metrics[*].voltage_pu",
        },
        "exp17_dispatch": exp17_dispatch,
        "disflow_dispatch": disflow_dispatch,
        "exp17_exact_distflow_eval": {
            "voltage_pu": exp17_voltage,
            "total_loss_mw": exp17_loss,
        },
        "summary": rows,
        "plot_paths": {key: str(value) for key, value in paths.items()},
    }
    save_json(out_dir / "comparison_24h_result.json", payload)

    print()
    print("24h saved Exp17 vs saved DistFlow comparison complete.")
    print(f"Exp17 result: {exp17_path}")
    print(f"DistFlow result: {disflow_path}")
    print(f"Output directory: {out_dir}")
    print("Recorded optimization solve time:")
    exp17_time = exp17_payload.get("solve_time_sec")
    disflow_time = disflow_payload.get("solve_time_sec")
    print(
        f"  Exp17 SGCN-MILP 24h: {float(exp17_time):.3f} s"
        if exp17_time is not None
        else "  Exp17 SGCN-MILP 24h: unavailable"
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
