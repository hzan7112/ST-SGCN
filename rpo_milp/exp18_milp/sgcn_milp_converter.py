"""Embed the Exp18 ST-SGCN voltage/current/loss model into a Gurobi MILP.

Exp18 predicts normalized nodal voltages, the system worst current margin, and
total network loss. The WorstI and Ploss scalar outputs share the same global
nonlinear ReLU readout and then use separate linear output heads, so adding
Ploss does not increase the MILP binary-variable count.
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
    / "st_sgcn_k4_h24_n2_g32_exp18_worsti_ploss_shared_milp_engine.pt"
)

RADIAL_BRANCHES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
    [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
    [15, 16], [16, 17], [1, 18], [18, 19], [19, 20], [20, 21],
    [2, 22], [22, 23], [23, 24], [5, 25], [25, 26], [26, 27],
    [27, 28], [28, 29], [29, 30], [30, 31], [31, 32],
]

RADIAL_BRANCH_R_OHM = np.array(
    [
        0.0922, 0.4930, 0.3660, 0.3811, 0.8190, 0.1872, 0.7114, 1.0300,
        1.0440, 0.1966, 0.3744, 1.4680, 0.5416, 0.5910, 0.7463, 1.2890,
        0.3720, 0.1640, 1.5042, 0.4095, 0.7089, 0.4512, 0.8980, 0.8960,
        0.2030, 0.2842, 1.0590, 0.8042, 0.5075, 0.9744, 0.3105, 0.3410,
    ],
    dtype=float,
)


class Exp18MILPOutputs(NamedTuple):
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


class SGCNMILPConverter:
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
            raise FileNotFoundError(f"Exp18 checkpoint not found: {checkpoint_path}")
        if relu_formulation not in {"big_m", "general"}:
            raise ValueError("relu_formulation must be 'big_m' or 'general'")
        if fallback_big_m <= 0.0:
            raise ValueError("fallback_big_m must be positive")
        if big_m_scale <= 0.0:
            raise ValueError("big_m_scale must be positive")

        payload = _safe_torch_load(checkpoint_path)
        if not isinstance(payload, dict):
            raise TypeError("Exp18 checkpoint payload must be a dict")

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

        self.downstream_matrix = self._required_matrix("downstream_matrix", square=True)
        self.path_power_matrix = self._required_matrix("path_power_matrix", square=True)
        self.num_nodes = int(self.downstream_matrix.shape[0])
        self.in_features = int(payload.get("in_features", 6))
        if self.num_nodes != 33 or self.in_features != 6:
            raise ValueError("Exp18 converter expects 33 nodes and 6 input features")

        self.edge_list = [[int(e[0]), int(e[1])] for e in payload.get("edge_list", RADIAL_BRANCHES)]
        branch_r = _as_numpy(payload.get("branch_r_ohm", []), dtype=float).reshape(-1)
        if branch_r.size == 0 and self.edge_list == RADIAL_BRANCHES:
            branch_r = RADIAL_BRANCH_R_OHM.copy()
        if branch_r.size != len(self.edge_list):
            raise ValueError(
                "branch_r_ohm length must match edge_list length; "
                f"got {branch_r.size} vs {len(self.edge_list)}"
            )
        self.branch_r_ohm = branch_r

        self.sgc_weight = self._weight("sgc_linear.weight", ndim=2)
        self.sgc_bias = self._weight("sgc_linear.bias", ndim=1)
        self.hidden_dim = int(self.sgc_weight.shape[0])
        self.order_count = self.sgc_weight.shape[1] // self.in_features
        self.node_relu_dim = int(self._weight("node_hidden.bias", ndim=1).shape[0])
        self.global_relu_dim = int(self._weight("global_hidden.bias", ndim=1).shape[0])

        powers = payload.get("frozen_adj_powers")
        if powers is None:
            powers = self.state.get("A_powers")
        if powers is None:
            raise KeyError("checkpoint missing frozen_adj_powers/A_powers")
        self.adj_powers = _as_numpy(powers)
        expected = (self.order_count, self.num_nodes, self.num_nodes)
        if self.adj_powers.shape != expected:
            raise ValueError(f"adjacency powers must have shape {expected}, got {self.adj_powers.shape}")

        self.use_sgc_relu = bool(payload.get("use_sgc_relu", self.config.get("use_sgc_relu", False)))
        self.use_linear_skip = bool(payload.get("use_linear_skip", self.config.get("use_linear_skip", True)))

        self.x_mean = _as_numpy(self.norm["X_mean"]).reshape(-1)
        self.x_std = _as_numpy(self.norm["X_std"]).reshape(-1)
        self.yv_mean = _as_numpy(self.norm["YV_mean_wo_slack"]).reshape(-1)
        self.yv_std = _as_numpy(self.norm["YV_std_wo_slack"]).reshape(-1)
        if self.x_mean.shape != (6,) or self.x_std.shape != (6,):
            raise ValueError("X_mean/X_std must be length 6")
        if self.yv_mean.shape != (32,) or self.yv_std.shape != (32,):
            raise ValueError("YV_mean_wo_slack/YV_std_wo_slack must be length 32")
        if np.any(np.abs(self.x_std) < 1e-12) or np.any(np.abs(self.yv_std) < 1e-12):
            raise ValueError("normalization std contains zero or too-small values")

        self.yi_mean = _scalar(self.norm["YI_worst_mean"], "YI_worst_mean")
        self.yi_std = _scalar(self.norm["YI_worst_std"], "YI_worst_std")
        if abs(self.yi_std) < 1e-12:
            raise ValueError("YI_worst_std is zero or too small")
        self.yp_mean = _scalar(self.norm["YP_loss_mean"], "YP_loss_mean")
        self.yp_std = _scalar(self.norm["YP_loss_std"], "YP_loss_std")
        if abs(self.yp_std) < 1e-12:
            raise ValueError("YP_loss_std is zero or too small")

        self.vlin_w = _as_numpy(payload["voltage_linear_prior_W"])
        self.vlin_b = _as_numpy(payload["voltage_linear_prior_b"]).reshape(-1)
        if self.vlin_w.shape != (32, self.num_nodes * self.in_features):
            raise ValueError(f"voltage_linear_prior_W has unexpected shape {self.vlin_w.shape}")
        if self.vlin_b.shape != (32,):
            raise ValueError(f"voltage_linear_prior_b has unexpected shape {self.vlin_b.shape}")

        self._validate_dimensions()
        self._prepare_big_m()
        self.last_variables = None

    @property
    def binary_count(self) -> int:
        return (
            self.num_nodes * self.node_relu_dim
            + self.global_relu_dim
            + (self.num_nodes * self.hidden_dim if self.use_sgc_relu else 0)
        )

    def _required_matrix(self, key: str, *, square: bool = False) -> np.ndarray:
        if key not in self.payload:
            raise KeyError(f"checkpoint missing {key}")
        matrix = _as_numpy(self.payload[key])
        if matrix.ndim != 2 or (square and matrix.shape[0] != matrix.shape[1]):
            raise ValueError(f"{key} must be a matrix, got {matrix.shape}")
        return matrix

    def _weight(self, key: str, *, ndim: int) -> np.ndarray:
        if key not in self.state:
            raise KeyError(f"model weight missing {key}")
        value = self.state[key]
        if value.ndim != ndim:
            raise ValueError(f"{key} must be {ndim}D, got {value.ndim}D")
        return value

    def _validate_dimensions(self) -> None:
        node_in = self.hidden_dim + 4
        global_in = self.num_nodes * self.hidden_dim + self.num_nodes * self.in_features
        checks = {
            "node_emb.weight": (self.num_nodes, 4),
            "node_hidden.weight": (self.node_relu_dim, node_in),
            "node_hidden.bias": (self.node_relu_dim,),
            "node_out.weight": (1, self.node_relu_dim),
            "node_out.bias": (1,),
            "global_hidden.weight": (self.global_relu_dim, global_in),
            "global_hidden.bias": (self.global_relu_dim,),
            "global_out.weight": (1, self.global_relu_dim),
            "global_out.bias": (1,),
            "global_ploss_out.weight": (1, self.global_relu_dim),
            "global_ploss_out.bias": (1,),
        }
        if self.use_linear_skip:
            checks["node_skip.weight"] = (1, node_in)
            checks["node_skip.bias"] = (1,)
            checks["global_skip.weight"] = (1, global_in)
            checks["global_skip.bias"] = (1,)
        for key, shape in checks.items():
            if self.state[key].shape != shape:
                raise ValueError(f"{key} expected shape {shape}, got {self.state[key].shape}")

    def _prepare_big_m(self) -> None:
        self.sgc_m_plus = None
        self.sgc_m_minus = None
        plus_layers = self.payload.get("M_plus_gcn_layers") or []
        minus_layers = self.payload.get("M_minus_gcn_layers") or []
        if plus_layers:
            self.sgc_m_plus = _as_numpy(plus_layers[0])
        if minus_layers:
            self.sgc_m_minus = _as_numpy(minus_layers[0])

        self.node_m_plus = _as_numpy(self.payload.get("M_plus_node", []))
        self.node_m_minus = _as_numpy(self.payload.get("M_minus_node", []))
        if self.node_m_plus.size == 0:
            self.node_m_plus = None
        if self.node_m_minus.size == 0:
            self.node_m_minus = None

        self.global_m_plus = _as_numpy(self.payload.get("M_plus_global", []))
        self.global_m_minus = _as_numpy(self.payload.get("M_minus_global", []))
        if self.global_m_plus.size == 0:
            self.global_m_plus = None
        if self.global_m_minus.size == 0:
            self.global_m_minus = None

    def _check_topology(self, topo_mask) -> None:
        if topo_mask is None:
            return
        mask = np.asarray(topo_mask, dtype=bool).reshape(-1)
        edge_count = len(self.edge_list)
        valid = mask.size == edge_count and bool(mask.all())
        if mask.size == edge_count + 5:
            valid = bool(mask[:edge_count].all() and (~mask[edge_count:]).all())
        if not valid:
            raise ValueError("Exp18 uses the frozen radial topology; topo_mask cannot change it")

    def _validate_inputs(self, x_vars: Sequence[Sequence]) -> None:
        if len(x_vars) != self.num_nodes:
            raise ValueError(f"X_vars must have {self.num_nodes} rows, got {len(x_vars)}")
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

    def _add_augmented_inputs(self, model, x_vars, prefix):
        augmented = [
            [model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Xaug_{i}_{f}") for f in range(self.in_features)]
            for i in range(self.num_nodes)
        ]
        for i in range(self.num_nodes):
            model.addConstr(augmented[i][0] == x_vars[i][0], name=f"{prefix}_P_{i}")
            model.addConstr(augmented[i][1] == x_vars[i][1], name=f"{prefix}_Q_{i}")
            for feature, source_index, matrix, label in (
                (2, 0, self.downstream_matrix, "Pdown"),
                (3, 1, self.downstream_matrix, "Qdown"),
                (4, 0, self.path_power_matrix, "Ppath"),
                (5, 1, self.path_power_matrix, "Qpath"),
            ):
                expr = self._linear_expr(matrix[i], [x_vars[j][source_index] for j in range(self.num_nodes)])
                model.addConstr(augmented[i][feature] == expr, name=f"{prefix}_{label}_{i}")
        return augmented

    def _add_normalization(self, model, augmented, prefix):
        normalized = [
            [model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Xnorm_{i}_{f}") for f in range(self.in_features)]
            for i in range(self.num_nodes)
        ]
        for i in range(self.num_nodes):
            for feature in range(self.in_features):
                inv_std = 1.0 / float(self.x_std[feature])
                expr = inv_std * augmented[i][feature] - float(self.x_mean[feature]) * inv_std
                model.addConstr(normalized[i][feature] == expr, name=f"{prefix}_norm_{i}_{feature}")
        return normalized

    def _add_sgc_encoder(self, model, normalized, prefix):
        orders = [
            [
                [model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Xorder_{order}_{i}_{f}") for f in range(self.in_features)]
                for i in range(self.num_nodes)
            ]
            for order in range(self.order_count)
        ]
        for order in range(self.order_count):
            for i in range(self.num_nodes):
                for feature in range(self.in_features):
                    expr = self._linear_expr(
                        self.adj_powers[order, i],
                        [normalized[j][feature] for j in range(self.num_nodes)],
                    )
                    model.addConstr(orders[order][i][feature] == expr, name=f"{prefix}_order_{order}_{i}_{feature}")

        z_sgc = []
        hidden = []
        for i in range(self.num_nodes):
            sgc_input = [
                orders[order][i][feature]
                for order in range(self.order_count)
                for feature in range(self.in_features)
            ]
            z_row = []
            h_row = []
            for d in range(self.hidden_dim):
                z_var = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Zsgc_{i}_{d}")
                expr = self._linear_expr(self.sgc_weight[d], sgc_input, self.sgc_bias[d])
                model.addConstr(z_var == expr, name=f"{prefix}_sgc_{i}_{d}")
                z_row.append(z_var)
                if self.use_sgc_relu:
                    plus = None if self.sgc_m_plus is None else self.sgc_m_plus[i, d]
                    minus = None if self.sgc_m_minus is None else self.sgc_m_minus[i, d]
                    h_row.append(self._add_relu(model, z_var, f"{prefix}_sgc_{i}_{d}", m_plus=plus, m_minus=minus))
                else:
                    h_row.append(z_var)
            z_sgc.append(z_row)
            hidden.append(h_row)
        return orders, z_sgc, hidden

    def _add_voltage_outputs(self, model, hidden, normalized, prefix):
        node_emb = self.state["node_emb.weight"]
        hidden_w = self.state["node_hidden.weight"]
        hidden_b = self.state["node_hidden.bias"]
        out_w = self.state["node_out.weight"][0]
        out_b = float(self.state["node_out.bias"][0])
        flat_x = [normalized[i][f] for i in range(self.num_nodes) for f in range(self.in_features)]

        v_nodes = [model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_V_0")]
        model.addConstr(v_nodes[0] == 1.0, name=f"{prefix}_V_slack")
        v_norm = [None]
        node_z = []
        node_relu = []
        for i in range(1, self.num_nodes):
            h_node = hidden[i] + [float(x) for x in node_emb[i]]
            z_row = []
            relu_row = []
            for d in range(self.node_relu_dim):
                z_var = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Vnode_Z_{i}_{d}")
                expr = self._linear_expr(hidden_w[d], h_node, hidden_b[d])
                model.addConstr(z_var == expr, name=f"{prefix}_Vnode_hidden_{i}_{d}")
                plus = None if self.node_m_plus is None else self.node_m_plus[i, d]
                minus = None if self.node_m_minus is None else self.node_m_minus[i, d]
                relu_var = self._add_relu(model, z_var, f"{prefix}_Vnode_{i}_{d}", m_plus=plus, m_minus=minus)
                z_row.append(z_var)
                relu_row.append(relu_var)

            res_norm = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Vres_norm_{i}")
            res_expr = self._linear_expr(out_w, relu_row, out_b)
            if self.use_linear_skip:
                res_expr += self._linear_expr(
                    self.state["node_skip.weight"][0],
                    h_node,
                    float(self.state["node_skip.bias"][0]),
                )
            model.addConstr(res_norm == res_expr, name=f"{prefix}_Vres_norm_constr_{i}")

            lin_norm = self._linear_expr(self.vlin_w[i - 1], flat_x, self.vlin_b[i - 1])
            total_norm = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_V_norm_{i}")
            model.addConstr(total_norm == res_norm + lin_norm, name=f"{prefix}_V_norm_total_{i}")

            physical_v = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_V_{i}")
            model.addConstr(
                physical_v == total_norm * float(self.yv_std[i - 1]) + float(self.yv_mean[i - 1]),
                name=f"{prefix}_V_denorm_{i}",
            )
            v_norm.append(total_norm)
            v_nodes.append(physical_v)
            node_z.append(z_row)
            node_relu.append(relu_row)
        return v_nodes, v_norm, node_z, node_relu

    def _add_global_outputs(self, model, hidden, normalized, prefix):
        global_vars = [
            hidden[i][d]
            for i in range(self.num_nodes)
            for d in range(self.hidden_dim)
        ] + [
            normalized[i][f]
            for i in range(self.num_nodes)
            for f in range(self.in_features)
        ]
        hidden_w = self.state["global_hidden.weight"]
        hidden_b = self.state["global_hidden.bias"]
        z_vars = []
        relu_vars = []
        for d in range(self.global_relu_dim):
            z_var = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Global_Z_{d}")
            expr = self._linear_expr(hidden_w[d], global_vars, hidden_b[d])
            model.addConstr(z_var == expr, name=f"{prefix}_Global_hidden_{d}")
            plus = None if self.global_m_plus is None else self.global_m_plus[d]
            minus = None if self.global_m_minus is None else self.global_m_minus[d]
            relu_var = self._add_relu(model, z_var, f"{prefix}_Global_{d}", m_plus=plus, m_minus=minus)
            z_vars.append(z_var)
            relu_vars.append(relu_var)

        yi_norm = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_WorstI_norm")
        yi_expr = self._linear_expr(
            self.state["global_out.weight"][0],
            relu_vars,
            float(self.state["global_out.bias"][0]),
        )
        if self.use_linear_skip:
            yi_expr += self._linear_expr(
                self.state["global_skip.weight"][0],
                global_vars,
                float(self.state["global_skip.bias"][0]),
            )
        model.addConstr(yi_norm == yi_expr, name=f"{prefix}_WorstI_normalized")
        yi = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_WorstI")
        model.addConstr(yi == yi_norm * self.yi_std + self.yi_mean, name=f"{prefix}_WorstI_denorm")

        ploss_norm = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Ploss_total_norm")
        ploss_expr = self._linear_expr(
            self.state["global_ploss_out.weight"][0],
            relu_vars,
            float(self.state["global_ploss_out.bias"][0]),
        )
        model.addConstr(ploss_norm == ploss_expr, name=f"{prefix}_Ploss_total_normalized")
        ploss = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Ploss_total")
        model.addConstr(
            ploss == ploss_norm * self.yp_std + self.yp_mean,
            name=f"{prefix}_Ploss_total_denorm",
        )
        return yi, yi_norm, ploss, ploss_norm, global_vars, z_vars, relu_vars

    def _add_voltage_metrics(self, model, v_nodes, prefix):
        dev_vars = []
        violation_vars = []
        v_lower = float(self.config.get("v_lower", 0.95))
        v_upper = float(self.config.get("v_upper", 1.05))
        for i in range(1, self.num_nodes):
            dev = model.addVar(lb=0.0, name=f"{prefix}_Vdev_abs_{i}")
            model.addConstr(dev >= v_nodes[i] - 1.0, name=f"{prefix}_Vdev_pos_{i}")
            model.addConstr(dev >= 1.0 - v_nodes[i], name=f"{prefix}_Vdev_neg_{i}")
            dev_vars.append(dev)

            over = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Vover_{i}")
            under = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Vunder_{i}")
            model.addConstr(over == v_nodes[i] - v_upper, name=f"{prefix}_Vover_constr_{i}")
            model.addConstr(under == v_lower - v_nodes[i], name=f"{prefix}_Vunder_constr_{i}")
            violation_vars.extend([over, under])

        vdev_total = model.addVar(lb=0.0, name=f"{prefix}_Vdev_total")
        model.addConstr(vdev_total == gp.quicksum(dev_vars), name=f"{prefix}_Vdev_total_constr")
        vworst = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Vworst")
        model.addGenConstrMax(vworst, violation_vars, name=f"{prefix}_Vworst_max")
        return vdev_total, vworst, dev_vars, violation_vars

    def embed_sgcn_constraints(
        self,
        model,
        X_vars: Sequence[Sequence],
        topo_mask=None,
        *,
        name_prefix: str = "exp18",
    ) -> Exp18MILPOutputs:
        if gp is None:
            raise ModuleNotFoundError("gurobipy is required to embed SGCN constraints")
        if not isinstance(model, gp.Model):
            raise TypeError("model must be a gurobipy.Model")
        self._validate_inputs(X_vars)
        self._check_topology(topo_mask)
        prefix = str(name_prefix).strip().replace(" ", "_") or "exp18"

        augmented = self._add_augmented_inputs(model, X_vars, prefix)
        normalized = self._add_normalization(model, augmented, prefix)
        orders, z_sgc, hidden = self._add_sgc_encoder(model, normalized, prefix)
        v_nodes, v_norm, node_z, node_relu = self._add_voltage_outputs(model, hidden, normalized, prefix)
        worst_i, worst_i_norm, ploss, ploss_norm, global_vars, global_z, global_relu = self._add_global_outputs(
            model, hidden, normalized, prefix
        )
        vdev_total, vworst, vdev_abs, v_violations = self._add_voltage_metrics(model, v_nodes, prefix)

        self.last_variables = {
            "X_aug": augmented,
            "X_norm": normalized,
            "X_orders": orders,
            "Z_sgc": z_sgc,
            "H": hidden,
            "V_nodes": v_nodes,
            "V_norm": v_norm,
            "node_Z": node_z,
            "node_relu": node_relu,
            "G": global_vars,
            "global_Z": global_z,
            "global_relu": global_relu,
            "WorstI_norm": worst_i_norm,
            "Ploss_norm": ploss_norm,
            "Vdev_abs": vdev_abs,
            "V_violations": v_violations,
        }
        return Exp18MILPOutputs(vdev_total, vworst, worst_i, ploss, v_nodes)

    def embed_gnn_constraints(
        self,
        model,
        X_vars: Sequence[Sequence],
        topo_mask=None,
        *,
        name_prefix: str = "exp18",
    ) -> Exp18MILPOutputs:
        return self.embed_sgcn_constraints(model, X_vars, topo_mask, name_prefix=name_prefix)


Exp18SGCNMILPConverter = SGCNMILPConverter


def _main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Exp18 ST-SGCN MILP converter configuration")
    parser.add_argument("checkpoint", nargs="?", default=str(DEFAULT_ENGINE_PATH))
    parser.add_argument("--relu-formulation", choices=("big_m", "general"), default="big_m")
    args = parser.parse_args()
    converter = SGCNMILPConverter(args.checkpoint, relu_formulation=args.relu_formulation)
    print(f"checkpoint: {converter.checkpoint_path}")
    print(
        "shape: "
        f"nodes={converter.num_nodes}, features={converter.in_features}, "
        f"orders={converter.order_count}, hidden={converter.hidden_dim}, "
        f"node_relu={converter.node_relu_dim}, global_relu={converter.global_relu_dim}"
    )
    print(f"ReLU formulation: {converter.relu_formulation}")
    print(f"binary count: {converter.binary_count}")
    print("outputs: Vdev_total, Vworst, WorstI, Ploss_total, V_nodes")


if __name__ == "__main__":
    _main()
