"""Plot 24-hour exact DistFlow RPO dispatch and nodal voltages.

The default input is the JSON written by
``rpo_milp/disflow/day_ahead_24h_distflow_rpo.py``. Figures are saved under
``rpo_milp/disflow/results/day_ahead_24h`` by default:

* 24h_reactive_dispatch.png
* 24h_voltage_heatmap.png
* 24h_voltage_envelope.png
"""

from __future__ import annotations

import argparse
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
DEFAULT_RESULT = REPO_ROOT / "rpo_milp" / "disflow" / "results" / "day_ahead_24h" / "result.json"
DEFAULT_OUT_DIR = REPO_ROOT / "rpo_milp" / "disflow" / "results" / "day_ahead_24h"


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def load_result(path: str | Path) -> dict:
    result_path = resolve_repo_path(path)
    with result_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Result JSON must be an object: {result_path}")
    return payload


def unwrap_disflow_result(payload: dict) -> dict:
    nested = payload.get("disflow_result")
    if isinstance(nested, dict):
        return nested
    return payload


def infer_horizon(result_payload: dict) -> int:
    metrics = result_payload.get("metrics")
    if isinstance(metrics, list) and metrics:
        return len(metrics)
    dispatch = result_payload.get("dispatch")
    if isinstance(dispatch, dict):
        for key in ("q_pv", "q_ess", "q_device"):
            value = dispatch.get(key)
            if value is not None:
                arr = np.asarray(value, dtype=float)
                if arr.ndim == 2:
                    return arr.shape[0]
    return HORIZON


def as_2d_array(value, *, horizon: int, name: str) -> np.ndarray:
    if value is None:
        return np.zeros((horizon, 0), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != horizon:
        raise ValueError(f"{name} must have shape ({horizon}, n), got {arr.shape}")
    return arr


def extract_dispatch(result_payload: dict, *, horizon: int) -> dict[str, np.ndarray]:
    dispatch = result_payload.get("dispatch")
    if not isinstance(dispatch, dict):
        raise ValueError("Result JSON must contain dispatch as an object")
    return {
        "q_pv": as_2d_array(dispatch.get("q_pv"), horizon=horizon, name="dispatch.q_pv"),
        "q_ess": as_2d_array(dispatch.get("q_ess"), horizon=horizon, name="dispatch.q_ess"),
        "q_device": as_2d_array(dispatch.get("q_device"), horizon=horizon, name="dispatch.q_device"),
        "p_ch": as_2d_array(dispatch.get("p_ch"), horizon=horizon, name="dispatch.p_ch"),
        "p_dis": as_2d_array(dispatch.get("p_dis"), horizon=horizon, name="dispatch.p_dis"),
    }


def extract_voltage(result_payload: dict, *, horizon: int) -> np.ndarray:
    metrics = result_payload.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != horizon:
        raise ValueError(f"Result JSON must contain metrics as a {horizon}-element list")

    voltage_rows = []
    missing = []
    for hour, item in enumerate(metrics):
        voltage = item.get("voltage_pu") if isinstance(item, dict) else None
        if voltage is None:
            missing.append(hour)
            continue
        row = np.asarray(voltage, dtype=float).reshape(-1)
        if row.size != 33:
            raise ValueError(f"metrics[{hour}].voltage_pu must have length 33, got {row.size}")
        voltage_rows.append(row)

    if missing:
        raise ValueError(f"Result JSON does not contain voltage_pu for hours: {missing}")
    return np.vstack(voltage_rows)


def extract_zero_control_voltage(payload: dict, *, horizon: int, pf_max_iter: int) -> np.ndarray | None:
    data_path = payload.get("data")
    if not data_path:
        return None

    path = resolve_repo_path(data_path)
    if not path.is_file():
        return None

    day = disflow_24h.load_day_ahead_data(path)
    base_config = day.base_config
    distflow = disflow_24h.ExactDistFlow(
        branches=day.branches,
        slack_vm_pu=float(base_config.get("slack_vm_pu", 1.03)),
        base_kv=float(base_config.get("base_kv", 12.66)),
        base_mva=float(base_config.get("base_mva", 1.0)),
        line_max_i_ka=float(base_config.get("line_max_i_ka", 0.20)),
    )
    dispatch = disflow_24h.baseline_dispatch(day)
    metrics = disflow_24h.evaluate_24h(day, distflow, dispatch, pf_max_iter=pf_max_iter)
    voltage = np.vstack([np.asarray(m.voltage_pu, dtype=float).reshape(-1) for m in metrics])
    if voltage.shape != (horizon, 33):
        return None
    return voltage


def control_labels(payload: dict, result_payload: dict, dispatch: dict[str, np.ndarray]) -> list[str]:
    labels = payload.get("control_labels", result_payload.get("control_labels"))
    expected = dispatch["q_pv"].shape[1] + dispatch["q_ess"].shape[1] + dispatch["q_device"].shape[1]
    if isinstance(labels, list) and len(labels) == expected:
        return [str(item) for item in labels]

    fallback = []
    fallback.extend(f"PV{i + 1}" for i in range(dispatch["q_pv"].shape[1]))
    fallback.extend(f"ESS{i + 1}" for i in range(dispatch["q_ess"].shape[1]))
    fallback.extend(f"QDev{i + 1}" for i in range(dispatch["q_device"].shape[1]))
    return fallback


def plot_reactive_dispatch(out_dir: Path, payload: dict, result_payload: dict, dispatch: dict[str, np.ndarray]) -> Path:
    q_pv = dispatch["q_pv"]
    q_ess = dispatch["q_ess"]
    q_device = dispatch["q_device"]
    hours = np.arange(q_pv.shape[0], dtype=int)
    q_all = np.concatenate([q_pv, q_ess, q_device], axis=1)
    if q_all.shape[1] == 0:
        raise ValueError("No reactive dispatch arrays found in result JSON")

    labels = control_labels(payload, result_payload, dispatch)
    groups = [
        ("PV inverter Q output", q_pv, labels[: q_pv.shape[1]]),
        (
            "ESS inverter Q output",
            q_ess,
            labels[q_pv.shape[1] : q_pv.shape[1] + q_ess.shape[1]],
        ),
        (
            "Reactive device Q output",
            q_device,
            labels[q_pv.shape[1] + q_ess.shape[1] :],
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.0), dpi=160, sharex=True)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(q_all.shape[1], 10)))
    color_idx = 0
    for ax, (title, values, group_labels) in zip(axes, groups):
        if values.shape[1] == 0:
            ax.text(0.5, 0.5, "No devices", transform=ax.transAxes, ha="center", va="center")
        for idx, label in enumerate(group_labels):
            ax.step(
                hours,
                values[:, idx],
                where="mid",
                linewidth=1.8,
                marker="o",
                markersize=3.2,
                color=colors[color_idx % len(colors)],
                label=label,
            )
            color_idx += 1
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Q (MVar)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", ncol=3, fontsize=8)

    axes[-1].set_xlabel("Hour")
    axes[-1].set_xticks(hours)
    fig.tight_layout()
    out_path = out_dir / "24h_reactive_dispatch.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_voltage_heatmap(out_dir: Path, voltage: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(12.0, 6.4), dpi=160)
    im = ax.imshow(
        voltage.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0.95,
        vmax=1.05,
        extent=(-0.5, voltage.shape[0] - 0.5, 0.5, 33.5),
    )
    ax.set_xlabel("Hour")
    ax.set_ylabel("Bus")
    ax.set_title("24-hour exact DistFlow nodal voltage")
    ax.set_xticks(np.arange(voltage.shape[0]))
    ax.set_yticks(np.arange(1, 34, 2))
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Voltage (p.u.)")
    fig.tight_layout()
    out_path = out_dir / "24h_voltage_heatmap.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_voltage_envelope(out_dir: Path, voltage: np.ndarray, pre_voltage: np.ndarray | None) -> Path:
    hours = np.arange(voltage.shape[0], dtype=int)
    v_min = voltage[:, 1:].min(axis=1)
    v_max = voltage[:, 1:].max(axis=1)
    v_mean = voltage[:, 1:].mean(axis=1)

    fig, ax = plt.subplots(figsize=(11.0, 5.2), dpi=160)
    if pre_voltage is not None:
        pre_min = pre_voltage[:, 1:].min(axis=1)
        pre_max = pre_voltage[:, 1:].max(axis=1)
        pre_mean = pre_voltage[:, 1:].mean(axis=1)
        ax.fill_between(
            hours,
            pre_min,
            pre_max,
            color="#E45756",
            alpha=0.12,
            label="Before range, all controls = 0",
        )
        ax.plot(hours, pre_min, linestyle="--", linewidth=1.5, color="#E45756", label="Before min")
        ax.plot(hours, pre_max, linestyle="--", linewidth=1.5, color="#B279A2", label="Before max")
        ax.plot(hours, pre_mean, linestyle=":", linewidth=1.5, color="#9D755D", label="Before mean")

    ax.fill_between(hours, v_min, v_max, color="#4C78A8", alpha=0.18, label="Optimized range")
    ax.plot(hours, v_min, marker="o", linewidth=1.8, color="#4C78A8", label="Optimized min")
    ax.plot(hours, v_max, marker="s", linewidth=1.8, color="#F58518", label="Optimized max")
    ax.plot(hours, v_mean, marker="^", linewidth=1.6, color="#54A24B", label="Optimized mean")
    ax.axhline(0.95, color="tab:red", linestyle="--", linewidth=1.0)
    ax.axhline(1.05, color="tab:red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title("24-hour exact DistFlow voltage envelope")
    ax.set_xticks(hours)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "24h_voltage_envelope.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot exact DistFlow 24h RPO dispatch and voltage results.")
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--distflow-pf-max-iter", type=int, default=200)
    parser.add_argument(
        "--no-before-voltage",
        action="store_true",
        help="Skip the zero-control voltage envelope in the before/after plot.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = load_result(args.result)
    result_payload = unwrap_disflow_result(payload)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    horizon = infer_horizon(result_payload)
    if horizon != HORIZON:
        raise ValueError(f"This plotting script expects a 24-hour result, got horizon={horizon}")

    dispatch = extract_dispatch(result_payload, horizon=horizon)
    voltage = extract_voltage(result_payload, horizon=horizon)
    pre_voltage = None
    if not args.no_before_voltage:
        pre_voltage = extract_zero_control_voltage(payload, horizon=horizon, pf_max_iter=args.distflow_pf_max_iter)

    dispatch_path = plot_reactive_dispatch(out_dir, payload, result_payload, dispatch)
    heatmap_path = plot_voltage_heatmap(out_dir, voltage)
    envelope_path = plot_voltage_envelope(out_dir, voltage, pre_voltage)

    print("24h exact DistFlow RPO plots written:")
    print(f"  {dispatch_path}")
    print(f"  {heatmap_path}")
    print(f"  {envelope_path}")


if __name__ == "__main__":
    main()
