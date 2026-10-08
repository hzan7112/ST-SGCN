"""Embed the Exp27 dual-linear-SGC surrogate as Gurobi MILP constraints.

Exp27 predicts four system-level scalars:

    Vdev_total, Vworst, WorstI, Ploss_total

The model has two linear SGC encoders:

    Voltage encoder: [P,Q,P_down,Q_down,P_path,Q_path] -> Vdev/Vworst
    Flow encoder:    [P,Q,P_down,Q_down,dP,dQ]          -> WorstI/Ploss

Both SGC encoders are linear when ``use_sgc_relu=False``. The MILP binaries
therefore come only from the task readout heads:

    24 + 24 + 24 + 32 + 32 = 136
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
    / "st_sgcn_exp27_k5_dual_linear_sgc_task_heads_milp_engine.pt"
)


class Exp27MILPOutputs(NamedTuple):
    Vdev_total: object
    Vworst: object
    WorstI: object
    Ploss_total: object


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


class SGCNMILPConverter:
    DIRECT_HEADS = (
        ("vdev_head", "Vdev_total", "YV_dev_mean", "YV_dev_std", "voltage"),
        ("vworst_head", "Vworst", "YV_worst_mean", "YV_worst_std", "voltage"),
        ("iworst_head", "WorstI", "YI_worst_mean", "YI_worst_std", "flow"),
    )
    PLOSS_HEAD = ("ploss_head", "Ploss_total", "YP_loss_mean", "YP_loss_std", "flow")

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
            raise FileNotFoundError(f"Exp27 checkpoint not found: {checkpoint_path}")
        if relu_formulation not in {"big_m", "general"}:
            raise ValueError("relu_formulation must be 'big_m' or 'general'")
        if fallback_big_m <= 0.0:
            raise ValueError("fallback_big_m must be positive")
        if big_m_scale <= 0.0:
            raise ValueError("big_m_scale must be positive")

        payload = _safe_torch_load(checkpoint_path)
        if not isinstance(payload, dict):
            raise TypeError("Exp27 checkpoint payload must be a dict")

        self.checkpoint_path = checkpoint_path
        self.payload = payload
        self.config = dict(payload.get("config", {}))
        self.relu_formulation = relu_formulation
        self.fallback_big_m = float(fallback_big_m)
        self.big_m_scale = float(big_m_scale)

        state = payload.get("state_dict", payload.get("model_state_dict"))
        if not isinstance(state, dict):
            raise KeyError("checkpoint missing state_dict/model_state_dict")
        self.state = {key: _as_numpy(value) for key, value in state.items()}

        self.norm = payload.get("norm_stats")
        if not isinstance(self.norm, dict):
            raise KeyError("checkpoint missing norm_stats")

        self.downstream_matrix = self._required_matrix("downstream_matrix", square=True)
        self.path_power_matrix = self._required_matrix("path_power_matrix", square=True)
        self.num_nodes = int(self.downstream_matrix.shape[0])
        if self.path_power_matrix.shape != (self.num_nodes, self.num_nodes):
            raise ValueError("path_power_matrix shape does not match downstream_matrix")

        parent = payload.get("parent_array")
        if parent is None:
            raise KeyError("checkpoint missing parent_array")
        self.parent = _as_numpy(parent, dtype=int).reshape(-1)
        if self.parent.shape != (self.num_nodes,):
            raise ValueError(f"parent_array must have shape ({self.num_nodes},)")

        self.voltage_in_features = int(
            payload.get("voltage_in_features", self.config.get("voltage_in_features", 6))
        )
        self.flow_in_features = int(
            payload.get("flow_in_features", self.config.get("flow_in_features", 6))
        )
        if self.voltage_in_features != 6 or self.flow_in_features != 6:
            raise ValueError("Exp27 converter expects 6 voltage features and 6 flow features")

        self.voltage_weight = self._weight("voltage_sgc_linear.weight", ndim=2)
        self.voltage_bias = self._weight("voltage_sgc_linear.bias", ndim=1)
        self.flow_weight = self._weight("flow_sgc_linear.weight", ndim=2)
        self.flow_bias = self._weight("flow_sgc_linear.bias", ndim=1)
        self.hidden_dim = int(self.voltage_weight.shape[0])
        if self.flow_weight.shape[0] != self.hidden_dim:
            raise ValueError("voltage/flow encoders must share hidden_dim")
        if self.voltage_bias.shape != (self.hidden_dim,) or self.flow_bias.shape != (self.hidden_dim,):
            raise ValueError("SGC encoder weight/bias dimensions do not match")
        if self.voltage_weight.shape[1] % self.voltage_in_features != 0:
            raise ValueError("voltage SGC input dimension is not divisible by feature count")
        if self.flow_weight.shape[1] % self.flow_in_features != 0:
            raise ValueError("flow SGC input dimension is not divisible by feature count")
        self.order_count = self.voltage_weight.shape[1] // self.voltage_in_features
        if self.flow_weight.shape[1] // self.flow_in_features != self.order_count:
            raise ValueError("voltage and flow encoders have different SGC order counts")

        powers = payload.get("frozen_adj_powers")
        if powers is None:
            powers = state.get("A_powers")
        if powers is None:
            raise KeyError("checkpoint missing frozen_adj_powers/A_powers")
        self.adj_powers = _as_numpy(powers)
        expected = (self.order_count, self.num_nodes, self.num_nodes)
        if self.adj_powers.shape != expected:
            raise ValueError(f"adjacency powers must have shape {expected}, got {self.adj_powers.shape}")

        self.use_sgc_relu = bool(
            payload.get("use_sgc_relu", self.config.get("use_sgc_relu", False))
        )
        self.use_linear_skip = bool(
            payload.get("use_linear_skip", self.config.get("use_linear_skip", True))
        )

        self.x_voltage_mean = _as_numpy(self.norm["X_voltage_mean"]).reshape(-1)
        self.x_voltage_std = _as_numpy(self.norm["X_voltage_std"]).reshape(-1)
        self.x_flow_mean = _as_numpy(self.norm["X_flow_mean"]).reshape(-1)
        self.x_flow_std = _as_numpy(self.norm["X_flow_std"]).reshape(-1)
        for label, mean, std in (
            ("voltage", self.x_voltage_mean, self.x_voltage_std),
            ("flow", self.x_flow_mean, self.x_flow_std),
        ):
            if mean.shape != (6,) or std.shape != (6,):
                raise ValueError(f"{label} feature normalization stats must be length 6")
            if np.any(np.abs(std) < 1e-12):
                raise ValueError(f"{label} feature std contains a zero or tiny value")

        self.edge_list = [
            [int(edge[0]), int(edge[1])]
            for edge in payload.get("edge_list", [])
        ]
        self.branch_r_ohm = _as_numpy(payload.get("branch_r_ohm", []), dtype=float).reshape(-1)
        if self.branch_r_ohm.size and self.branch_r_ohm.size != len(self.edge_list):
            raise ValueError(
                f"branch_r_ohm length must match edge_list: {self.branch_r_ohm.size} vs {len(self.edge_list)}"
            )

        self._validate_heads()
        self._prepare_big_m()
        self.last_variables = None

    @property
    def in_features(self) -> int:
        return self.voltage_in_features

    def _required_matrix(self, key: str, *, square: bool = False) -> np.ndarray:
        if key not in self.payload:
            raise KeyError(f"checkpoint missing {key}")
        matrix = _as_numpy(self.payload[key])
        if matrix.ndim != 2 or (square and matrix.shape[0] != matrix.shape[1]):
            raise ValueError(f"{key} must be a square matrix, got {matrix.shape}")
        return matrix

    def _weight(self, key: str, *, ndim: int) -> np.ndarray:
        if key not in self.state:
            raise KeyError(f"model state missing {key}")
        value = self.state[key]
        if value.ndim != ndim:
            raise ValueError(f"{key} must be {ndim}D, got {value.ndim}D")
        return value

    def _global_dim(self, branch: str) -> int:
        features = self.voltage_in_features if branch == "voltage" else self.flow_in_features
        return self.num_nodes * (self.hidden_dim + features)

    def _validate_direct_head(self, module_name: str, mean_key: str, std_key: str, branch: str) -> int:
        global_dim = self._global_dim(branch)
        hidden_w = self._weight(f"{module_name}.hidden.weight", ndim=2)
        hidden_b = self._weight(f"{module_name}.hidden.bias", ndim=1)
        out_w = self._weight(f"{module_name}.out.weight", ndim=2)
        out_b = self._weight(f"{module_name}.out.bias", ndim=1)
        head_dim = hidden_w.shape[0]
        if hidden_w.shape[1] != global_dim:
            raise ValueError(f"{module_name} input dim must be {global_dim}, got {hidden_w.shape[1]}")
        if hidden_b.shape != (head_dim,):
            raise ValueError(f"{module_name}.hidden weight/bias mismatch")
        if out_w.shape != (1, head_dim) or out_b.shape != (1,):
            raise ValueError(f"{module_name}.out must be hidden_dim -> 1")
        if self.use_linear_skip:
            skip_w = self._weight(f"{module_name}.skip.weight", ndim=2)
            skip_b = self._weight(f"{module_name}.skip.bias", ndim=1)
            if skip_w.shape != (1, global_dim) or skip_b.shape != (1,):
                raise ValueError(f"{module_name}.skip dimensions are invalid")
        if mean_key not in self.norm or std_key not in self.norm:
            raise KeyError(f"norm_stats missing {mean_key}/{std_key}")
        return head_dim

    def _validate_ploss_head(self) -> tuple[int, int]:
        module_name, _, mean_key, std_key, branch = self.PLOSS_HEAD
        global_dim = self._global_dim(branch)
        w1 = self._weight(f"{module_name}.hidden1.weight", ndim=2)
        b1 = self._weight(f"{module_name}.hidden1.bias", ndim=1)
        w2 = self._weight(f"{module_name}.hidden2.weight", ndim=2)
        b2 = self._weight(f"{module_name}.hidden2.bias", ndim=1)
        out_w = self._weight(f"{module_name}.out.weight", ndim=2)
        out_b = self._weight(f"{module_name}.out.bias", ndim=1)
        h1, h2 = int(w1.shape[0]), int(w2.shape[0])
        if w1.shape[1] != global_dim:
            raise ValueError(f"{module_name}.hidden1 input dim must be {global_dim}, got {w1.shape[1]}")
        if b1.shape != (h1,) or w2.shape != (h2, h1) or b2.shape != (h2,):
            raise ValueError(f"{module_name} hidden layer dimensions are invalid")
        if out_w.shape != (1, h2) or out_b.shape != (1,):
            raise ValueError(f"{module_name}.out must be hidden2_dim -> 1")
        if self.use_linear_skip:
            skip_w = self._weight(f"{module_name}.skip.weight", ndim=2)
            skip_b = self._weight(f"{module_name}.skip.bias", ndim=1)
            if skip_w.shape != (1, global_dim) or skip_b.shape != (1,):
                raise ValueError(f"{module_name}.skip dimensions are invalid")
        if mean_key not in self.norm or std_key not in self.norm:
            raise KeyError(f"norm_stats missing {mean_key}/{std_key}")
        return h1, h2

    def _validate_heads(self) -> None:
        self.direct_head_dims = [
            self._validate_direct_head(module_name, mean_key, std_key, branch)
            for module_name, _, mean_key, std_key, branch in self.DIRECT_HEADS
        ]
        self.ploss_head_dims = self._validate_ploss_head()

    def _prepare_big_m(self) -> None:
        self.global_m_plus = None
        self.global_m_minus = None
        if self.payload.get("M_plus_global") is not None:
            self.global_m_plus = _as_numpy(self.payload["M_plus_global"]).reshape(-1)
        if self.payload.get("M_minus_global") is not None:
            self.global_m_minus = _as_numpy(self.payload["M_minus_global"]).reshape(-1)

        total_head_dim = sum(self.direct_head_dims) + sum(self.ploss_head_dims)
        for label, value in (
            ("M_plus_global", self.global_m_plus),
            ("M_minus_global", self.global_m_minus),
        ):
            if value is not None and value.size != total_head_dim:
                raise ValueError(f"{label} length must be {total_head_dim}, got {value.size}")

        self.sgc_m_plus = []
        self.sgc_m_minus = []
        plus_layers = self.payload.get("M_plus_gcn_layers") or []
        minus_layers = self.payload.get("M_minus_gcn_layers") or []
        expected = (self.num_nodes, self.hidden_dim)
        for layers, target, label in (
            (plus_layers, self.sgc_m_plus, "M_plus_gcn_layers"),
            (minus_layers, self.sgc_m_minus, "M_minus_gcn_layers"),
        ):
            for layer in layers:
                value = _as_numpy(layer)
                if value.shape != expected:
                    raise ValueError(f"{label} entries must have shape {expected}, got {value.shape}")
                target.append(value)

    @property
    def binary_count(self) -> int:
        sgc_count = 2 * self.num_nodes * self.hidden_dim if self.use_sgc_relu else 0
        return sum(self.direct_head_dims) + sum(self.ploss_head_dims) + sgc_count

    def _check_topology(self, topo_mask) -> None:
        if topo_mask is None:
            return
        mask = np.asarray(topo_mask, dtype=bool).reshape(-1)
        edge_count = len(self.edge_list)
        valid = mask.size == edge_count and bool(mask.all())
        if mask.size == edge_count + 5:
            valid = bool(mask[:edge_count].all() and (~mask[edge_count:]).all())
        if not valid:
            raise ValueError("Exp27 uses a fixed radial topology; topo_mask cannot change branch states")

    def _validate_inputs(self, x_vars: Sequence[Sequence]) -> None:
        if len(x_vars) != self.num_nodes:
            raise ValueError(f"X_vars first dimension must be {self.num_nodes}, got {len(x_vars)}")
        for i, row in enumerate(x_vars):
            if len(row) != 2:
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

    def _make_augmented_numpy(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = x[:, :, 0]
        q = x[:, :, 1]
        p_down = p @ self.downstream_matrix.T
        q_down = q @ self.downstream_matrix.T
        p_path = p @ self.path_power_matrix.T
        q_path = q @ self.path_power_matrix.T
        voltage = np.stack([p, q, p_down, q_down, p_path, q_path], axis=2)

        p_delta = np.zeros_like(p)
        q_delta = np.zeros_like(q)
        for child, parent in enumerate(self.parent):
            if parent >= 0:
                p_delta[:, child] = p[:, parent] - p[:, child]
                q_delta[:, child] = q[:, parent] - q[:, child]
        flow = np.stack([p, q, p_down, q_down, p_delta, q_delta], axis=2)
        return voltage, flow

    def _encode_numpy(self, x_norm, weight, bias):
        x_orders = np.einsum("kij,bjf->bkif", self.adj_powers, x_norm)
        x_multi = np.transpose(x_orders, (0, 2, 1, 3)).reshape(
            x_norm.shape[0],
            self.num_nodes,
            self.order_count * x_norm.shape[2],
        )
        z_sgc = np.einsum("bif,df->bid", x_multi, weight) + bias
        hidden = np.maximum(z_sgc, 0.0) if self.use_sgc_relu else z_sgc
        global_values = np.concatenate(
            [hidden.reshape(x_norm.shape[0], -1), x_norm.reshape(x_norm.shape[0], -1)],
            axis=1,
        )
        return z_sgc, hidden, global_values

    def _eval_direct_head_numpy(self, module_name, global_values, mean_key, std_key):
        hidden_w = self.state[f"{module_name}.hidden.weight"]
        hidden_b = self.state[f"{module_name}.hidden.bias"]
        out_w = self.state[f"{module_name}.out.weight"][0]
        out_b = float(self.state[f"{module_name}.out.bias"][0])
        z = global_values @ hidden_w.T + hidden_b
        y_norm = np.maximum(z, 0.0) @ out_w + out_b
        if self.use_linear_skip:
            skip_w = self.state[f"{module_name}.skip.weight"][0]
            skip_b = float(self.state[f"{module_name}.skip.bias"][0])
            y_norm = y_norm + global_values @ skip_w + skip_b
        y = y_norm * _scalar(self.norm[std_key], std_key) + _scalar(self.norm[mean_key], mean_key)
        return y_norm, y, z

    def _eval_ploss_head_numpy(self, global_values):
        module_name, _, mean_key, std_key, _ = self.PLOSS_HEAD
        z1 = global_values @ self.state[f"{module_name}.hidden1.weight"].T
        z1 += self.state[f"{module_name}.hidden1.bias"]
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.state[f"{module_name}.hidden2.weight"].T
        z2 += self.state[f"{module_name}.hidden2.bias"]
        h2 = np.maximum(z2, 0.0)
        y_norm = h2 @ self.state[f"{module_name}.out.weight"][0]
        y_norm += float(self.state[f"{module_name}.out.bias"][0])
        if self.use_linear_skip:
            y_norm += global_values @ self.state[f"{module_name}.skip.weight"][0]
            y_norm += float(self.state[f"{module_name}.skip.bias"][0])
        y = y_norm * _scalar(self.norm[std_key], std_key) + _scalar(self.norm[mean_key], mean_key)
        return y_norm, y, z1, z2

    def forward_numpy(self, x_net_mw_mvar: np.ndarray, *, return_intermediates: bool = False):
        x = np.asarray(x_net_mw_mvar, dtype=float)
        single = x.ndim == 2
        if single:
            x = x[None, :, :]
        if x.shape[1:] != (self.num_nodes, 2):
            raise ValueError(f"x_net_mw_mvar must have shape (33,2) or (N,33,2), got {x.shape}")

        x_voltage, x_flow = self._make_augmented_numpy(x)
        xv_norm = (x_voltage - self.x_voltage_mean.reshape(1, 1, -1)) / self.x_voltage_std.reshape(1, 1, -1)
        xf_norm = (x_flow - self.x_flow_mean.reshape(1, 1, -1)) / self.x_flow_std.reshape(1, 1, -1)
        zv, _, gv = self._encode_numpy(xv_norm, self.voltage_weight, self.voltage_bias)
        zf, _, gf = self._encode_numpy(xf_norm, self.flow_weight, self.flow_bias)

        normalized_outputs = []
        physical_outputs = []
        head_z = {}
        for module_name, output_name, mean_key, std_key, branch in self.DIRECT_HEADS:
            global_values = gv if branch == "voltage" else gf
            y_norm, y, z = self._eval_direct_head_numpy(module_name, global_values, mean_key, std_key)
            normalized_outputs.append(y_norm)
            physical_outputs.append(y)
            head_z[output_name] = z
        y_norm, y, z1, z2 = self._eval_ploss_head_numpy(gf)
        normalized_outputs.append(y_norm)
        physical_outputs.append(y)
        head_z["Ploss_total_hidden1"] = z1
        head_z["Ploss_total_hidden2"] = z2

        outputs = np.stack(physical_outputs, axis=1)
        outputs_norm = np.stack(normalized_outputs, axis=1)
        if single:
            outputs = outputs[0]
            outputs_norm = outputs_norm[0]
        if not return_intermediates:
            return outputs
        return {
            "outputs": outputs,
            "outputs_norm": outputs_norm,
            "X_voltage_aug": x_voltage[0] if single else x_voltage,
            "X_flow_aug": x_flow[0] if single else x_flow,
            "X_voltage_norm": xv_norm[0] if single else xv_norm,
            "X_flow_norm": xf_norm[0] if single else xf_norm,
            "Z_voltage_sgc": zv[0] if single else zv,
            "Z_flow_sgc": zf[0] if single else zf,
            "G_voltage": gv[0] if single else gv,
            "G_flow": gf[0] if single else gf,
            "head_Z": head_z,
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

    def _add_augmented_inputs(self, model, x_vars, prefix, kind: str):
        augmented = [
            [model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_{kind}_Xaug_{i}_{f}") for f in range(6)]
            for i in range(self.num_nodes)
        ]
        for i in range(self.num_nodes):
            model.addConstr(augmented[i][0] == x_vars[i][0], name=f"{prefix}_{kind}_P_{i}")
            model.addConstr(augmented[i][1] == x_vars[i][1], name=f"{prefix}_{kind}_Q_{i}")
            for feature, source_index, matrix, label in (
                (2, 0, self.downstream_matrix, "Pdown"),
                (3, 1, self.downstream_matrix, "Qdown"),
            ):
                expr = self._linear_expr(matrix[i], [x_vars[j][source_index] for j in range(self.num_nodes)])
                model.addConstr(augmented[i][feature] == expr, name=f"{prefix}_{kind}_{label}_{i}")
            if kind == "voltage":
                for feature, source_index, matrix, label in (
                    (4, 0, self.path_power_matrix, "Ppath"),
                    (5, 1, self.path_power_matrix, "Qpath"),
                ):
                    expr = self._linear_expr(matrix[i], [x_vars[j][source_index] for j in range(self.num_nodes)])
                    model.addConstr(augmented[i][feature] == expr, name=f"{prefix}_{kind}_{label}_{i}")
            else:
                parent = int(self.parent[i])
                if parent >= 0:
                    model.addConstr(augmented[i][4] == x_vars[parent][0] - x_vars[i][0], name=f"{prefix}_{kind}_dP_{i}")
                    model.addConstr(augmented[i][5] == x_vars[parent][1] - x_vars[i][1], name=f"{prefix}_{kind}_dQ_{i}")
                else:
                    model.addConstr(augmented[i][4] == 0.0, name=f"{prefix}_{kind}_dP_{i}")
                    model.addConstr(augmented[i][5] == 0.0, name=f"{prefix}_{kind}_dQ_{i}")
        return augmented

    def _add_normalization(self, model, augmented, mean, std, prefix, kind: str):
        normalized = [
            [model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_{kind}_Xnorm_{i}_{f}") for f in range(6)]
            for i in range(self.num_nodes)
        ]
        for i in range(self.num_nodes):
            for feature in range(6):
                inv_std = 1.0 / float(std[feature])
                model.addConstr(
                    normalized[i][feature] == inv_std * augmented[i][feature] - float(mean[feature]) * inv_std,
                    name=f"{prefix}_{kind}_norm_{i}_{feature}",
                )
        return normalized

    def _add_sgc_encoder(self, model, normalized, weight, bias, prefix, kind: str, layer_index: int):
        orders = [
            [
                [model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_{kind}_Xorder_{order}_{i}_{feature}") for feature in range(6)]
                for i in range(self.num_nodes)
            ]
            for order in range(self.order_count)
        ]
        for order in range(self.order_count):
            for i in range(self.num_nodes):
                for feature in range(6):
                    expr = self._linear_expr(
                        self.adj_powers[order, i],
                        [normalized[j][feature] for j in range(self.num_nodes)],
                    )
                    model.addConstr(orders[order][i][feature] == expr, name=f"{prefix}_{kind}_order_{order}_{i}_{feature}")

        z_sgc = []
        hidden = []
        for i in range(self.num_nodes):
            sgc_input = [
                orders[order][i][feature]
                for order in range(self.order_count)
                for feature in range(6)
            ]
            z_row = []
            h_row = []
            for d in range(self.hidden_dim):
                z_var = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_{kind}_Zsgc_{i}_{d}")
                model.addConstr(
                    z_var == self._linear_expr(weight[d], sgc_input, bias[d]),
                    name=f"{prefix}_{kind}_sgc_{i}_{d}",
                )
                z_row.append(z_var)
                if self.use_sgc_relu:
                    plus = None
                    minus = None
                    if layer_index < len(self.sgc_m_plus):
                        plus = self.sgc_m_plus[layer_index][i, d]
                    if layer_index < len(self.sgc_m_minus):
                        minus = self.sgc_m_minus[layer_index][i, d]
                    h_row.append(self._add_relu(model, z_var, f"{prefix}_{kind}_sgc_{i}_{d}", m_plus=plus, m_minus=minus))
                else:
                    h_row.append(z_var)
            z_sgc.append(z_row)
            hidden.append(h_row)
        return orders, z_sgc, hidden

    def _global_vars(self, hidden, normalized):
        return [
            hidden[i][d]
            for i in range(self.num_nodes)
            for d in range(self.hidden_dim)
        ] + [
            normalized[i][feature]
            for i in range(self.num_nodes)
            for feature in range(6)
        ]

    def _head_big_m(self, start: int, index: int):
        plus = None if self.global_m_plus is None else self.global_m_plus[start + index]
        minus = None if self.global_m_minus is None else self.global_m_minus[start + index]
        return plus, minus

    def _add_direct_head(self, model, global_vars, head_spec, m_start: int, prefix: str):
        module_name, output_name, mean_key, std_key, _ = head_spec
        hidden_w = self.state[f"{module_name}.hidden.weight"]
        hidden_b = self.state[f"{module_name}.hidden.bias"]
        out_w = self.state[f"{module_name}.out.weight"][0]
        out_b = float(self.state[f"{module_name}.out.bias"][0])

        z_vars = []
        relu_vars = []
        for d in range(hidden_w.shape[0]):
            z_var = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_{output_name}_Z_{d}")
            model.addConstr(z_var == self._linear_expr(hidden_w[d], global_vars, hidden_b[d]), name=f"{prefix}_{output_name}_hidden_{d}")
            plus, minus = self._head_big_m(m_start, d)
            relu_vars.append(self._add_relu(model, z_var, f"{prefix}_{output_name}_{d}", m_plus=plus, m_minus=minus))
            z_vars.append(z_var)

        normalized_output = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_{output_name}_norm")
        output_expr = self._linear_expr(out_w, relu_vars, out_b)
        if self.use_linear_skip:
            skip_w = self.state[f"{module_name}.skip.weight"][0]
            skip_b = float(self.state[f"{module_name}.skip.bias"][0])
            output_expr += self._linear_expr(skip_w, global_vars, skip_b)
        model.addConstr(normalized_output == output_expr, name=f"{prefix}_{output_name}_normalized")

        physical_output = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_{output_name}")
        mean = _scalar(self.norm[mean_key], mean_key)
        std = _scalar(self.norm[std_key], std_key)
        model.addConstr(physical_output == normalized_output * std + mean, name=f"{prefix}_{output_name}_denorm")
        return physical_output, normalized_output, z_vars, relu_vars

    def _add_ploss_head(self, model, global_vars, m_start: int, prefix: str):
        module_name, output_name, mean_key, std_key, _ = self.PLOSS_HEAD
        z1_vars = []
        h1_vars = []
        w1 = self.state[f"{module_name}.hidden1.weight"]
        b1 = self.state[f"{module_name}.hidden1.bias"]
        for d in range(w1.shape[0]):
            z_var = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_{output_name}_Z1_{d}")
            model.addConstr(z_var == self._linear_expr(w1[d], global_vars, b1[d]), name=f"{prefix}_{output_name}_hidden1_{d}")
            plus, minus = self._head_big_m(m_start, d)
            h1_vars.append(self._add_relu(model, z_var, f"{prefix}_{output_name}_h1_{d}", m_plus=plus, m_minus=minus))
            z1_vars.append(z_var)

        z2_vars = []
        h2_vars = []
        w2 = self.state[f"{module_name}.hidden2.weight"]
        b2 = self.state[f"{module_name}.hidden2.bias"]
        h1_dim = len(z1_vars)
        for d in range(w2.shape[0]):
            z_var = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_{output_name}_Z2_{d}")
            model.addConstr(z_var == self._linear_expr(w2[d], h1_vars, b2[d]), name=f"{prefix}_{output_name}_hidden2_{d}")
            plus, minus = self._head_big_m(m_start + h1_dim, d)
            h2_vars.append(self._add_relu(model, z_var, f"{prefix}_{output_name}_h2_{d}", m_plus=plus, m_minus=minus))
            z2_vars.append(z_var)

        normalized_output = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_{output_name}_norm")
        out_w = self.state[f"{module_name}.out.weight"][0]
        out_b = float(self.state[f"{module_name}.out.bias"][0])
        output_expr = self._linear_expr(out_w, h2_vars, out_b)
        if self.use_linear_skip:
            skip_w = self.state[f"{module_name}.skip.weight"][0]
            skip_b = float(self.state[f"{module_name}.skip.bias"][0])
            output_expr += self._linear_expr(skip_w, global_vars, skip_b)
        model.addConstr(normalized_output == output_expr, name=f"{prefix}_{output_name}_normalized")

        physical_output = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_{output_name}")
        mean = _scalar(self.norm[mean_key], mean_key)
        std = _scalar(self.norm[std_key], std_key)
        model.addConstr(physical_output == normalized_output * std + mean, name=f"{prefix}_{output_name}_denorm")
        return physical_output, normalized_output, (z1_vars, z2_vars), (h1_vars, h2_vars)

    def embed_sgcn_constraints(
        self,
        model,
        X_vars: Sequence[Sequence],
        topo_mask=None,
        *,
        name_prefix: str = "exp27",
    ) -> Exp27MILPOutputs:
        if gp is None:
            raise ModuleNotFoundError("gurobipy is required to embed SGCN constraints")
        if not isinstance(model, gp.Model):
            raise TypeError("model must be a gurobipy.Model")
        self._validate_inputs(X_vars)
        self._check_topology(topo_mask)
        prefix = str(name_prefix).strip().replace(" ", "_") or "exp27"

        voltage_aug = self._add_augmented_inputs(model, X_vars, prefix, "voltage")
        flow_aug = self._add_augmented_inputs(model, X_vars, prefix, "flow")
        voltage_norm = self._add_normalization(model, voltage_aug, self.x_voltage_mean, self.x_voltage_std, prefix, "voltage")
        flow_norm = self._add_normalization(model, flow_aug, self.x_flow_mean, self.x_flow_std, prefix, "flow")
        voltage_orders, z_voltage, h_voltage = self._add_sgc_encoder(
            model, voltage_norm, self.voltage_weight, self.voltage_bias, prefix, "voltage", 0
        )
        flow_orders, z_flow, h_flow = self._add_sgc_encoder(
            model, flow_norm, self.flow_weight, self.flow_bias, prefix, "flow", 1
        )
        g_voltage = self._global_vars(h_voltage, voltage_norm)
        g_flow = self._global_vars(h_flow, flow_norm)

        physical_outputs = []
        normalized_outputs = []
        head_z = {}
        head_relu = {}
        m_start = 0
        for spec in self.DIRECT_HEADS:
            global_vars = g_voltage if spec[4] == "voltage" else g_flow
            output, output_norm, z_vars, relu_vars = self._add_direct_head(
                model, global_vars, spec, m_start, prefix
            )
            physical_outputs.append(output)
            normalized_outputs.append(output_norm)
            head_z[spec[1]] = z_vars
            head_relu[spec[1]] = relu_vars
            m_start += len(z_vars)

        output, output_norm, z_vars, relu_vars = self._add_ploss_head(
            model, g_flow, m_start, prefix
        )
        physical_outputs.append(output)
        normalized_outputs.append(output_norm)
        head_z["Ploss_total"] = z_vars
        head_relu["Ploss_total"] = relu_vars

        self.last_variables = {
            "X_aug": voltage_aug,
            "X_norm": voltage_norm,
            "X_flow_aug": flow_aug,
            "X_flow_norm": flow_norm,
            "X_voltage_orders": voltage_orders,
            "X_flow_orders": flow_orders,
            "Z_voltage_sgc": z_voltage,
            "Z_flow_sgc": z_flow,
            "H_voltage": h_voltage,
            "H_flow": h_flow,
            "G_voltage": g_voltage,
            "G_flow": g_flow,
            "head_Z": head_z,
            "head_relu": head_relu,
            "outputs_norm": normalized_outputs,
        }
        return Exp27MILPOutputs(*physical_outputs)

    def embed_gnn_constraints(
        self,
        model,
        X_vars: Sequence[Sequence],
        topo_mask=None,
        *,
        name_prefix: str = "exp27",
    ) -> Exp27MILPOutputs:
        return self.embed_sgcn_constraints(model, X_vars, topo_mask, name_prefix=name_prefix)


Exp27SGCNMILPConverter = SGCNMILPConverter


def _main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Exp27 ST-SGCN MILP converter configuration")
    parser.add_argument("checkpoint", nargs="?", default=str(DEFAULT_ENGINE_PATH))
    parser.add_argument("--relu-formulation", choices=("big_m", "general"), default="big_m")
    args = parser.parse_args()
    converter = SGCNMILPConverter(args.checkpoint, relu_formulation=args.relu_formulation)
    print(f"checkpoint: {converter.checkpoint_path}")
    print(
        "shape: "
        f"nodes={converter.num_nodes}, voltage_features={converter.voltage_in_features}, "
        f"flow_features={converter.flow_in_features}, orders={converter.order_count}, "
        f"hidden={converter.hidden_dim}"
    )
    print(f"ReLU formulation: {converter.relu_formulation}")
    print(f"binary count: {converter.binary_count}")
    print("outputs: Vdev_total, Vworst, WorstI, Ploss_total")


if __name__ == "__main__":
    _main()
