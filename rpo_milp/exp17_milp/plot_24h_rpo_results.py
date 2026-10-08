"""Plot 24-hour Exp17 reactive-power dispatch and nodal voltages.

The default input is the JSON written by ``day_ahead_24h_rpo.py``. Figures are
saved under ``rpo_milp/exp17_milp/results`` by default:

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

from rpo_milp.disflow.single_time_opf.compare_exp17_single_step import ExactDistFlow, RADIAL_BRANCHES


DEFAULT_RESULT = REPO_ROOT / "rpo_milp" / "exp17_milp" / "results" / "day_ahead_24h_result.json"
DEFAULT_OUT_DIR = REPO_ROOT / "rpo_milp" / "exp17_milp" / "results"


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


def as_2d_array(payload: dict, key: str, *, horizon: int) -> np.ndarray:
    value = payload.get(key)
    if value is None:
        return np.zeros((horizon, 0), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != horizon:
        raise ValueError(f"{key} must have shape (24, n), got {arr.shape}")
    return arr


def extract_voltage(payload: dict, *, horizon: int) -> np.ndarray:
    outputs = payload.get("exact_pf_outputs")
    source = "exact_pf_outputs"
    if not isinstance(outputs, list):
        outputs = payload.get("outputs")
        source = "outputs"
    if not isinstance(outputs, list) or len(outputs) != horizon:
        raise ValueError("Result JSON must contain exact_pf_outputs or outputs as a 24-element list")
    if source == "outputs":
        print("Warning: exact_pf_outputs not found; plotting surrogate outputs.V_nodes instead.")

    rows = []
    missing = []
    for hour, item in enumerate(outputs):
        v_nodes = item.get("V_nodes") if isinstance(item, dict) else None
        if v_nodes is None:
            missing.append(hour)
            continue
        row = np.asarray(v_nodes, dtype=float).reshape(-1)
        if row.size != 33:
            raise ValueError(f"{source}[{hour}].V_nodes must have length 33, got {row.size}")
        rows.append(row)

    if missing:
        raise ValueError(
            "Result JSON does not contain nodal voltages for every hour. "
            "Rerun day_ahead_24h_rpo.py after the exact PF output update. "
            f"Missing hours: {missing}"
        )
    return np.vstack(rows)


def extract_zero_control_voltage(payload: dict, *, horizon: int) -> np.ndarray | None:
    """Compute voltage with every decision variable set to zero.

    This is the "before optimization" case used in the plots:
    PV active power follows Ppv_24h, loads follow Pload_24h/Qload_24h, and all
    controllable injections are zero:

        Q_pv = 0, Q_ess = 0, Q_device = 0, P_ess_ch = 0, P_ess_dis = 0
    """
    data_path = payload.get("data")
    if not data_path:
        return None

    path = resolve_repo_path(data_path)
    if not path.is_file():
        return None

    data = safe_torch_load(path)
    if not isinstance(data, dict):
        return None
    required = ["Pload_24h", "Qload_24h", "Ppv_24h", "pv_nodes"]
    if any(key not in data for key in required):
        return None

    p_load = to_numpy(data["Pload_24h"])
    q_load = to_numpy(data["Qload_24h"])
    p_pv = to_numpy(data["Ppv_24h"])
    pv_nodes = to_numpy(data["pv_nodes"], dtype=int).reshape(-1)
    branches = to_numpy(data.get("branch_full")) if "branch_full" in data else RADIAL_BRANCHES
    base_config = dict(data.get("base_config", {}))
    if p_load.shape != (33, horizon) or q_load.shape != (33, horizon):
        return None
    if p_pv.ndim != 2 or p_pv.shape[0] != horizon or p_pv.shape[1] != pv_nodes.size:
        return None

    distflow = ExactDistFlow(
        branches=np.asarray(branches, dtype=float),
        slack_vm_pu=float(base_config.get("slack_vm_pu", 1.03)),
        line_max_i_ka=float(base_config.get("line_max_i_ka", 0.20)),
    )

    voltage_rows = []
    for hour in range(horizon):
        x_net = np.zeros((33, 2), dtype=float)
        x_net[:, 0] = -p_load[:, hour]
        x_net[:, 1] = -q_load[:, hour]
        for idx, bus in enumerate(pv_nodes):
            x_net[int(bus), 0] += p_pv[hour, idx]
        metrics = distflow.solve(x_net)
        voltage_rows.append(metrics.voltage_pu)
    return np.vstack(voltage_rows)


def control_labels(payload: dict, q_pv: np.ndarray, q_ess: np.ndarray, q_device: np.ndarray) -> list[str]:
    labels = payload.get("control_labels")
    expected = q_pv.shape[1] + q_ess.shape[1] + q_device.shape[1]
    if isinstance(labels, list) and len(labels) == expected:
        return [str(x) for x in labels]

    fallback = []
    fallback.extend(f"PV{i + 1}" for i in range(q_pv.shape[1]))
    fallback.extend(f"ESS{i + 1}" for i in range(q_ess.shape[1]))
    fallback.extend(f"QDev{i + 1}" for i in range(q_device.shape[1]))
    return fallback


def plot_reactive_dispatch(
    out_dir: Path,
    payload: dict,
    *,
    q_pv: np.ndarray,
    q_ess: np.ndarray,
    q_device: np.ndarray,
) -> Path:
    hours = np.arange(q_pv.shape[0], dtype=int)
    q_all = np.concatenate([q_pv, q_ess, q_device], axis=1)
    labels = control_labels(payload, q_pv, q_ess, q_device)
    if q_all.shape[1] == 0:
        raise ValueError("No reactive dispatch arrays found in result JSON")

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.0), dpi=160, sharex=True)
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
        extent=(-0.5, 23.5, 0.5, 33.5),
    )
    ax.set_xlabel("Hour")
    ax.set_ylabel("Bus")
    ax.set_title("24-hour nodal voltage")
    ax.set_xticks(np.arange(24))
    ax.set_yticks(np.arange(1, 34, 2))
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Voltage (p.u.)")
    fig.tight_layout()
    out_path = out_dir / "24h_voltage_heatmap.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_voltage_envelope(out_dir: Path, voltage: np.ndarray, pre_voltage: np.ndarray | None = None) -> Path:
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
        ax.plot(
            hours,
            pre_min,
            linestyle="--",
            linewidth=1.5,
            color="#E45756",
            label="Before min, all controls = 0",
        )
        ax.plot(
            hours,
            pre_max,
            linestyle="--",
            linewidth=1.5,
            color="#B279A2",
            label="Before max, all controls = 0",
        )
        ax.plot(
            hours,
            pre_mean,
            linestyle=":",
            linewidth=1.5,
            color="#9D755D",
            label="Before mean, all controls = 0",
        )

    ax.fill_between(hours, v_min, v_max, color="#4C78A8", alpha=0.18, label="Min-max range")
    ax.plot(hours, v_min, marker="o", linewidth=1.8, color="#4C78A8", label="Min voltage")
    ax.plot(hours, v_max, marker="s", linewidth=1.8, color="#F58518", label="Max voltage")
    ax.plot(hours, v_mean, marker="^", linewidth=1.6, color="#54A24B", label="Mean voltage")
    ax.axhline(0.95, color="tab:red", linestyle="--", linewidth=1.0)
    ax.axhline(1.05, color="tab:red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title("24-hour voltage envelope")
    ax.set_xticks(hours)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path = out_dir / "24h_voltage_envelope.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Exp17 24h RPO dispatch and voltage results.")
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = load_result(args.result)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    horizon = int(payload.get("horizon", 24))
    if horizon != 24:
        raise ValueError(f"This plotting script expects a 24-hour result, got horizon={horizon}")

    q_pv = as_2d_array(payload, "Q_pv_opt", horizon=horizon)
    q_ess = as_2d_array(payload, "Q_ess_opt", horizon=horizon)
    q_device = as_2d_array(payload, "Q_device_opt", horizon=horizon)
    voltage = extract_voltage(payload, horizon=horizon)
    pre_voltage = extract_zero_control_voltage(payload, horizon=horizon)

    dispatch_path = plot_reactive_dispatch(out_dir, payload, q_pv=q_pv, q_ess=q_ess, q_device=q_device)
    heatmap_path = plot_voltage_heatmap(out_dir, voltage)
    envelope_path = plot_voltage_envelope(out_dir, voltage, pre_voltage)

    print("24h Exp17 RPO plots written:")
    print(f"  {dispatch_path}")
    print(f"  {heatmap_path}")
    print(f"  {envelope_path}")


if __name__ == "__main__":
    main()
