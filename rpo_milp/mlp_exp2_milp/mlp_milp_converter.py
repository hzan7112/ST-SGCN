"""Embed the ``model/mlp`` four-scalar MLP into a Gurobi MILP.

This converter targets the engine produced by ``model/mlp/train_mlp.py``. The
model consumes raw IEEE-33 node net injections ``[P_net, Q_net]`` and predicts
four denormalized system-level scalars aligned with ``model/exp24``:

    Vdev_total, Vworst, WorstI, Ploss_total

The RPO safety constraints are ``Vworst <= 0`` and ``WorstI <= 0`` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import torch

try:
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError:
    gp = None
    GRB = None


DEFAULT_ENGINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "checkpoints"
    / "mlp_exp20_four_scalars_96bin_milp_engine.pt"
)

RADIAL_BRANCHES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
    [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
    [15, 16], [16, 17], [1, 18], [18, 19], [19, 20], [20, 21],
    [2, 22], [22, 23], [23, 24], [5, 25], [25, 26], [26, 27],
    [27, 28], [28, 29], [29, 30], [30, 31], [31, 32],
]


class MLPMILPOutputs(NamedTuple):
    Vdev_total: object
    Vworst: object
    WorstI: object
    Ploss_total: object
    V_nodes: object


def _safe_torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_numpy(value, *, dtype=float) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _scalar(value, name: str) -> float:
    array = _as_numpy(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar, got shape {array.shape}")
    return float(array[0])


class MLPMILPConverter:
    """Convert ``model/mlp/train_mlp.py``'s ReLU MLP into MILP constraints."""

    OUTPUTS = (
        ("Vdev_total", "YV_dev_mean", "YV_dev_std"),
        ("Vworst", "YV_worst_mean", "YV_worst_std"),
        ("WorstI", "YI_worst_mean", "YI_worst_std"),
        ("Ploss_total", "YP_loss_mean", "YP_loss_std"),
    )

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_ENGINE_PATH,
        *,
        relu_formulation: str = "big_m",
        fallback_big_m: float = 1e3,
        big_m_scale: float = 1.0,
    ):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"model/mlp engine not found: {checkpoint_path}. "
                "Run python model/mlp/train_mlp.py to generate it."
            )
        if relu_formulation not in {"big_m", "general"}:
            raise ValueError("relu_formulation must be 'big_m' or 'general'")
        if fallback_big_m <= 0.0:
            raise ValueError("fallback_big_m must be positive")
        if big_m_scale <= 0.0:
            raise ValueError("big_m_scale must be positive")

        payload = _safe_torch_load(checkpoint_path)
        if not isinstance(payload, dict):
            raise TypeError("MLP checkpoint payload must be a dict")

        self.checkpoint_path = checkpoint_path
        self.payload = payload
        self.relu_formulation = relu_formulation
        self.fallback_big_m = float(fallback_big_m)
        self.big_m_scale = float(big_m_scale)

        state = payload.get("state_dict", payload.get("model_state_dict"))
        if not isinstance(state, dict):
            raise KeyError("checkpoint missing state_dict/model_state_dict")
        self.state = {key: _as_numpy(value) for key, value in state.items()}
        self.config = dict(payload.get("config", {}))
        self.norm = payload.get("norm_stats")
        if not isinstance(self.norm, dict):
            raise KeyError("checkpoint missing norm_stats")

        self.num_nodes = 33
        self.in_features = 2
        self.input_dim = int(payload.get("input_dim", self.config.get("input_dim", 66)))
        if self.input_dim != self.num_nodes * self.in_features:
            raise ValueError(f"input_dim must be 66 for IEEE-33 raw PQ, got {self.input_dim}")

        self.hidden_dims = [
            int(dim)
            for dim in payload.get("hidden_dims", self.config.get("hidden_dims", []))
        ]
        if not self.hidden_dims:
            idx = 0
            while f"hidden_layers.{idx}.weight" in self.state:
                self.hidden_dims.append(int(self.state[f"hidden_layers.{idx}.weight"].shape[0]))
                idx += 1
        if len(self.hidden_dims) == 0:
            raise KeyError("checkpoint does not describe hidden_dims")

        self.edge_list = [
            [int(edge[0]), int(edge[1])]
            for edge in payload.get("edge_list", RADIAL_BRANCHES)
        ]
        self.branch_r_ohm = _as_numpy(payload.get("branch_r_ohm", []), dtype=float).reshape(-1)
        self.line_max_i_ka = float(payload.get("line_max_i_ka", self.config.get("default_line_max_i_ka", 0.20)))

        self.x_mean = _as_numpy(self.norm["X_mean"]).reshape(-1)
        self.x_std = _as_numpy(self.norm["X_std"]).reshape(-1)
        if self.x_mean.shape != (2,) or self.x_std.shape != (2,):
            raise ValueError("X_mean/X_std must be length 2")
        if np.any(np.abs(self.x_std) < 1e-12):
            raise ValueError("normalization std contains zero or too-small values")

        self._validate_network()
        self._prepare_big_m()
        self.last_variables = None

    @property
    def binary_count(self) -> int:
        return int(sum(self.hidden_dims))

    def _weight(self, key: str, *, ndim: int) -> np.ndarray:
        if key not in self.state:
            raise KeyError(f"model weight missing {key}")
        value = self.state[key]
        if value.ndim != ndim:
            raise ValueError(f"{key} must be {ndim}D, got {value.ndim}D")
        return value

    def _validate_network(self) -> None:
        dims = [self.input_dim] + self.hidden_dims
        for idx, width in enumerate(self.hidden_dims):
            weight = self._weight(f"hidden_layers.{idx}.weight", ndim=2)
            bias = self._weight(f"hidden_layers.{idx}.bias", ndim=1)
            expected = (width, dims[idx])
            if weight.shape != expected:
                raise ValueError(f"hidden_layers.{idx}.weight expected {expected}, got {weight.shape}")
            if bias.shape != (width,):
                raise ValueError(f"hidden_layers.{idx}.bias expected {(width,)}, got {bias.shape}")

        out_w = self._weight("output_layer.weight", ndim=2)
        out_b = self._weight("output_layer.bias", ndim=1)
        if out_w.shape != (4, self.hidden_dims[-1]) or out_b.shape != (4,):
            raise ValueError(
                "output_layer must map final hidden layer to "
                "[Vdev_total, Vworst, WorstI, Ploss_total]"
            )
        for _, mean_key, std_key in self.OUTPUTS:
            if mean_key not in self.norm or std_key not in self.norm:
                raise KeyError(f"norm_stats missing {mean_key}/{std_key}")
            if abs(_scalar(self.norm[std_key], std_key)) < 1e-12:
                raise ValueError(f"{std_key} is zero or too small")

    def _prepare_big_m(self) -> None:
        plus_layers = self.payload.get("M_plus_mlp_layers") or []
        minus_layers = self.payload.get("M_minus_mlp_layers") or []
        self.m_plus_layers = [_as_numpy(value).reshape(-1) for value in plus_layers]
        self.m_minus_layers = [_as_numpy(value).reshape(-1) for value in minus_layers]
        if self.m_plus_layers and len(self.m_plus_layers) != len(self.hidden_dims):
            raise ValueError("M_plus_mlp_layers length must match hidden_dims")
        if self.m_minus_layers and len(self.m_minus_layers) != len(self.hidden_dims):
            raise ValueError("M_minus_mlp_layers length must match hidden_dims")
        for idx, width in enumerate(self.hidden_dims):
            if self.m_plus_layers and self.m_plus_layers[idx].size != width:
                raise ValueError(f"M_plus_mlp_layers[{idx}] length must be {width}")
            if self.m_minus_layers and self.m_minus_layers[idx].size != width:
                raise ValueError(f"M_minus_mlp_layers[{idx}] length must be {width}")

    def _check_topology(self, topo_mask) -> None:
        if topo_mask is None:
            return
        mask = np.asarray(topo_mask, dtype=bool).reshape(-1)
        edge_count = len(self.edge_list)
        valid = mask.size == edge_count and bool(mask.all())
        if mask.size == edge_count + 5:
            valid = bool(mask[:edge_count].all() and (~mask[edge_count:]).all())
        if not valid:
            raise ValueError("model/mlp engine was trained for the fixed radial topology")

    def _validate_inputs(self, x_vars: Sequence[Sequence]) -> None:
        if len(x_vars) != self.num_nodes:
            raise ValueError(f"X_vars must have {self.num_nodes} rows, got {len(x_vars)}")
        for i, row in enumerate(x_vars):
            if len(row) != self.in_features:
                raise ValueError(f"X_vars[{i}] must contain [P_net, Q_net]")

    @staticmethod
    def _linear_expr(coefficients, variables, bias: float = 0.0):
        if gp is None:
            raise ModuleNotFoundError("gurobipy is required to build MILP expressions")
        expr = gp.LinExpr(float(bias))
        for coefficient, variable in zip(coefficients, variables):
            coefficient = float(coefficient)
            if coefficient != 0.0:
                expr += coefficient * variable
        return expr

    def forward_numpy(self, x_net_mw_mvar: np.ndarray, *, return_intermediates: bool = False):
        x = np.asarray(x_net_mw_mvar, dtype=float)
        single = x.ndim == 2
        if single:
            x = x[None, :, :]
        if x.shape[1:] != (self.num_nodes, self.in_features):
            raise ValueError(f"x_net_mw_mvar must have shape (33, 2) or (N, 33, 2), got {x.shape}")

        x_norm = (x - self.x_mean.reshape(1, 1, -1)) / self.x_std.reshape(1, 1, -1)
        h = x_norm.reshape(x.shape[0], self.input_dim)
        hidden_z = []
        hidden_relu = []
        for idx in range(len(self.hidden_dims)):
            weight = self.state[f"hidden_layers.{idx}.weight"]
            bias = self.state[f"hidden_layers.{idx}.bias"]
            z = h @ weight.T + bias
            h = np.maximum(z, 0.0)
            hidden_z.append(z)
            hidden_relu.append(h)

        y_norm = h @ self.state["output_layer.weight"].T + self.state["output_layer.bias"]
        physical = []
        for idx, (_, mean_key, std_key) in enumerate(self.OUTPUTS):
            physical.append(y_norm[:, idx] * _scalar(self.norm[std_key], std_key) + _scalar(self.norm[mean_key], mean_key))
        outputs = np.stack(physical, axis=1)
        if single:
            outputs = outputs[0]
            y_norm = y_norm[0]

        if not return_intermediates:
            return outputs
        return {
            "outputs": outputs,
            "outputs_norm": y_norm,
            "X_norm": x_norm[0] if single else x_norm,
            "hidden_Z": [z[0] if single else z for z in hidden_z],
            "hidden_relu": [h_item[0] if single else h_item for h_item in hidden_relu],
        }

    def _add_relu(self, model, z_var, name: str, *, m_plus=None, m_minus=None):
        y_var = model.addVar(lb=0.0, name=f"{name}_relu")
        if self.relu_formulation == "general":
            model.addGenConstrMax(y_var, [z_var], constant=0.0, name=f"{name}_max")
            return y_var

        upper = self.fallback_big_m if m_plus is None else max(float(m_plus), 0.0)
        lower = self.fallback_big_m if m_minus is None else max(float(m_minus), 0.0)
        upper *= self.big_m_scale
        lower *= self.big_m_scale
        active = model.addVar(vtype=GRB.BINARY, name=f"{name}_bin")
        model.addConstr(y_var >= z_var, name=f"{name}_lb_z")
        model.addConstr(y_var <= z_var + lower * (1.0 - active), name=f"{name}_ub_inactive")
        model.addConstr(y_var <= upper * active, name=f"{name}_ub_active")
        return y_var

    def _add_normalization(self, model, x_vars, prefix):
        normalized = [
            [model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Xnorm_{i}_{f}") for f in range(self.in_features)]
            for i in range(self.num_nodes)
        ]
        for i in range(self.num_nodes):
            for feature in range(self.in_features):
                inv_std = 1.0 / float(self.x_std[feature])
                expr = inv_std * x_vars[i][feature] - float(self.x_mean[feature]) * inv_std
                model.addConstr(normalized[i][feature] == expr, name=f"{prefix}_norm_{i}_{feature}")
        return normalized

    def embed_mlp_constraints(
        self,
        model,
        X_vars: Sequence[Sequence],
        topo_mask=None,
        *,
        name_prefix: str = "mlp",
    ) -> MLPMILPOutputs:
        if gp is None:
            raise ModuleNotFoundError("gurobipy is required to embed MLP constraints")
        if not isinstance(model, gp.Model):
            raise TypeError("model must be a gurobipy.Model")
        self._validate_inputs(X_vars)
        self._check_topology(topo_mask)
        prefix = str(name_prefix).strip().replace(" ", "_") or "mlp"

        normalized = self._add_normalization(model, X_vars, prefix)
        h_vars = [
            normalized[i][feature]
            for i in range(self.num_nodes)
            for feature in range(self.in_features)
        ]

        hidden_z_layers = []
        hidden_relu_layers = []
        for layer_idx, width in enumerate(self.hidden_dims):
            weight = self.state[f"hidden_layers.{layer_idx}.weight"]
            bias = self.state[f"hidden_layers.{layer_idx}.bias"]
            z_row = []
            relu_row = []
            for d in range(width):
                z_var = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_hidden{layer_idx}_Z_{d}")
                expr = self._linear_expr(weight[d], h_vars, bias[d])
                model.addConstr(z_var == expr, name=f"{prefix}_hidden{layer_idx}_{d}")
                plus = self.m_plus_layers[layer_idx][d] if self.m_plus_layers else None
                minus = self.m_minus_layers[layer_idx][d] if self.m_minus_layers else None
                relu_var = self._add_relu(
                    model,
                    z_var,
                    f"{prefix}_hidden{layer_idx}_{d}",
                    m_plus=plus,
                    m_minus=minus,
                )
                z_row.append(z_var)
                relu_row.append(relu_var)
            hidden_z_layers.append(z_row)
            hidden_relu_layers.append(relu_row)
            h_vars = relu_row

        out_w = self.state["output_layer.weight"]
        out_b = self.state["output_layer.bias"]
        physical_outputs = []
        normalized_outputs = []
        for idx, (output_name, mean_key, std_key) in enumerate(self.OUTPUTS):
            y_norm = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_{output_name}_norm")
            y_expr = self._linear_expr(out_w[idx], h_vars, out_b[idx])
            model.addConstr(y_norm == y_expr, name=f"{prefix}_{output_name}_normalized")

            y_phys = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_{output_name}")
            mean = _scalar(self.norm[mean_key], mean_key)
            std = _scalar(self.norm[std_key], std_key)
            model.addConstr(y_phys == y_norm * std + mean, name=f"{prefix}_{output_name}_denorm")
            normalized_outputs.append(y_norm)
            physical_outputs.append(y_phys)

        vdev_total, vworst, worst_i, ploss_total = physical_outputs
        self.last_variables = {
            "X_norm": normalized,
            "hidden_Z": hidden_z_layers,
            "hidden_relu": hidden_relu_layers,
            "outputs_norm": normalized_outputs,
        }
        return MLPMILPOutputs(
            Vdev_total=vdev_total,
            Ploss_total=ploss_total,
            Vworst=vworst,
            WorstI=worst_i,
            V_nodes=None,
        )

    def embed_sgcn_constraints(
        self,
        model,
        X_vars: Sequence[Sequence],
        topo_mask=None,
        *,
        name_prefix: str = "mlp",
    ) -> MLPMILPOutputs:
        return self.embed_mlp_constraints(model, X_vars, topo_mask=topo_mask, name_prefix=name_prefix)

    def embed_gnn_constraints(
        self,
        model,
        X_vars: Sequence[Sequence],
        topo_mask=None,
        *,
        name_prefix: str = "mlp",
    ) -> MLPMILPOutputs:
        return self.embed_mlp_constraints(model, X_vars, topo_mask=topo_mask, name_prefix=name_prefix)


MLPFourScalarsMILPConverter = MLPMILPConverter


def _main() -> None:
    parser = argparse.ArgumentParser(description="Inspect model/mlp MILP converter configuration")
    parser.add_argument("checkpoint", nargs="?", default=str(DEFAULT_ENGINE_PATH))
    parser.add_argument("--relu-formulation", choices=("big_m", "general"), default="big_m")
    args = parser.parse_args()
    converter = MLPMILPConverter(args.checkpoint, relu_formulation=args.relu_formulation)
    print(f"checkpoint: {converter.checkpoint_path}")
    print(
        "shape: "
        f"nodes={converter.num_nodes}, features={converter.in_features}, "
        f"input_dim={converter.input_dim}, hidden_dims={converter.hidden_dims}"
    )
    print(f"ReLU formulation: {converter.relu_formulation}")
    print(f"binary count: {converter.binary_count}")
    print("outputs: Vdev_total, Vworst, WorstI, Ploss_total")


if __name__ == "__main__":
    _main()
