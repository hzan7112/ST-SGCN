"""把 Exp24 ST-SGCN 精确展开为 Gurobi MILP 约束。

该转换器对应 :mod:`model.exp24.model` 中的
``STSGCNFourDirectScalars``。输入是 33 个节点的原始净注入 ``[P, Q]``，
输出是已经反归一化的四个系统级标量：累计电压偏差、最坏电压裕度、
最坏电流裕度和总有功网损。

Exp24 使用训练时冻结的辐射状拓扑。因此，可优化 P/Q（尤其是光伏无功），
但不能在同一个已训练模型中改变网络拓扑。
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
    / "st_sgcn_exp24_nodal_pq_four_direct_scalar_heads_milp_engine.pt"
)


class Exp24MILPOutputs(NamedTuple):
    """``embed_sgcn_constraints`` 返回的四个物理量 Gurobi 变量。"""

    Vdev_total: object
    Vworst: object
    WorstI: object
    Ploss_total: object


def _safe_torch_load(path: str | Path):
    """兼容不同 PyTorch 版本，并显式在 CPU 上加载可信本地检查点。"""

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
        raise ValueError(f"{name} 应为标量，实际形状为 {array.shape}")
    return float(array[0])


class SGCNMILPConverter:
    """将 Exp24 的冻结 ST-SGCN 嵌入一个 Gurobi 模型。

    Parameters
    ----------
    checkpoint_path:
        Exp24 训练产生的 ``*_milp_engine.pt``（推荐）或 ``*_best.pt``。
        文件必须同时包含网络权重、归一化统计量、下游/路径矩阵。
    relu_formulation:
        ``"big_m"`` 使用训练导出的逐神经元 Big-M 和显式二进制变量；
        ``"general"`` 使用 Gurobi 的 ``MAX`` 通用约束。
    fallback_big_m:
        检查点未提供 Big-M 时使用的保守回退值。
    big_m_scale:
        对检查点中的 Big-M 再做缩放。训练文件本身已乘 ``big_m_beta``，
        因而默认值为 1。
    """

    HEADS = (
        ("vdev_head", "Vdev_total", "YV_dev_mean", "YV_dev_std"),
        ("vworst_head", "Vworst", "YV_worst_mean", "YV_worst_std"),
        ("iworst_head", "WorstI", "YI_worst_mean", "YI_worst_std"),
        ("ploss_head", "Ploss_total", "YP_loss_mean", "YP_loss_std"),
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
            raise FileNotFoundError(f"找不到 Exp24 检查点：{checkpoint_path}")
        if relu_formulation not in {"big_m", "general"}:
            raise ValueError("relu_formulation 必须是 'big_m' 或 'general'")
        if fallback_big_m <= 0.0:
            raise ValueError("fallback_big_m 必须大于 0")
        if big_m_scale <= 0.0:
            raise ValueError("big_m_scale 必须大于 0")

        payload = _safe_torch_load(checkpoint_path)
        if not isinstance(payload, dict):
            raise TypeError("Exp24 检查点顶层必须是字典")

        self.checkpoint_path = checkpoint_path
        self.payload = payload
        self.relu_formulation = relu_formulation
        self.fallback_big_m = float(fallback_big_m)
        self.big_m_scale = float(big_m_scale)

        state = payload.get("state_dict", payload.get("model_state_dict"))
        if not isinstance(state, dict):
            raise KeyError("检查点缺少 state_dict/model_state_dict")
        self.state = {key: _as_numpy(value) for key, value in state.items()}

        self.config = dict(payload.get("config", {}))
        self.norm = payload.get("norm_stats")
        if not isinstance(self.norm, dict):
            raise KeyError("检查点缺少 norm_stats")

        self.downstream_matrix = self._required_matrix(
            "downstream_matrix", square=True
        )
        self.path_power_matrix = self._required_matrix(
            "path_power_matrix", square=True
        )
        self.num_nodes = int(self.downstream_matrix.shape[0])
        if self.path_power_matrix.shape != (self.num_nodes, self.num_nodes):
            raise ValueError("path_power_matrix 与 downstream_matrix 维度不一致")

        self.in_features = int(payload.get("in_features", 6))
        if self.in_features != 6:
            raise ValueError(
                "Exp24 MILP 转换器要求 6 个特征："
                "[P_net,Q_net,P_down,Q_down,P_path,Q_path]"
            )

        self.sgc_weight = self._weight("sgc_linear.weight", ndim=2)
        self.sgc_bias = self._weight("sgc_linear.bias", ndim=1)
        self.hidden_dim = int(self.sgc_weight.shape[0])
        if self.sgc_bias.shape != (self.hidden_dim,):
            raise ValueError("sgc_linear 的 weight/bias 维度不一致")
        if self.sgc_weight.shape[1] % self.in_features != 0:
            raise ValueError("sgc_linear 输入维度不能被 in_features 整除")
        self.order_count = self.sgc_weight.shape[1] // self.in_features

        powers = payload.get("frozen_adj_powers")
        if powers is None:
            powers = state.get("A_powers")
        if powers is None:
            raise KeyError("检查点缺少 frozen_adj_powers/A_powers")
        self.adj_powers = _as_numpy(powers)
        expected = (self.order_count, self.num_nodes, self.num_nodes)
        if self.adj_powers.shape != expected:
            raise ValueError(
                f"邻接矩阵幂形状应为 {expected}，实际为 {self.adj_powers.shape}"
            )

        self.use_sgc_relu = bool(
            payload.get("use_sgc_relu", self.config.get("use_sgc_relu", False))
        )
        self.use_linear_skip = bool(
            payload.get("use_linear_skip", self.config.get("use_linear_skip", True))
        )

        self.x_mean = _as_numpy(self.norm["X_mean"]).reshape(-1)
        self.x_std = _as_numpy(self.norm["X_std"]).reshape(-1)
        if self.x_mean.shape != (6,) or self.x_std.shape != (6,):
            raise ValueError("X_mean 和 X_std 必须均为长度 6 的向量")
        if np.any(np.abs(self.x_std) < 1e-12):
            raise ValueError("X_std 中存在 0 或过小的值")

        self.edge_list = [
            [int(edge[0]), int(edge[1])]
            for edge in payload.get("edge_list", [])
        ]
        self.branch_r_ohm = _as_numpy(
            payload.get("branch_r_ohm", []),
            dtype=float,
        ).reshape(-1)
        if self.branch_r_ohm.size and self.branch_r_ohm.size != len(self.edge_list):
            raise ValueError(
                "branch_r_ohm length must match edge_list length: "
                f"{self.branch_r_ohm.size} vs {len(self.edge_list)}"
            )
        self._validate_heads()
        self._prepare_big_m()
        self.last_variables = None

    def _required_matrix(self, key: str, *, square: bool = False) -> np.ndarray:
        if key not in self.payload:
            raise KeyError(f"检查点缺少 {key}")
        matrix = _as_numpy(self.payload[key])
        if matrix.ndim != 2 or (square and matrix.shape[0] != matrix.shape[1]):
            raise ValueError(f"{key} 必须为方阵，实际形状为 {matrix.shape}")
        return matrix

    def _weight(self, key: str, *, ndim: int) -> np.ndarray:
        if key not in self.state:
            raise KeyError(f"模型权重缺少 {key}")
        value = self.state[key]
        if value.ndim != ndim:
            raise ValueError(f"{key} 应为 {ndim} 维，实际为 {value.ndim} 维")
        return value

    def _validate_heads(self) -> None:
        global_dim = self.num_nodes * (self.hidden_dim + self.in_features)
        for module_name, _, mean_key, std_key in self.HEADS:
            hidden_w = self._weight(f"{module_name}.hidden.weight", ndim=2)
            hidden_b = self._weight(f"{module_name}.hidden.bias", ndim=1)
            out_w = self._weight(f"{module_name}.out.weight", ndim=2)
            out_b = self._weight(f"{module_name}.out.bias", ndim=1)
            head_dim = hidden_w.shape[0]
            if hidden_w.shape[1] != global_dim:
                raise ValueError(
                    f"{module_name} 输入维度应为 {global_dim}，"
                    f"实际为 {hidden_w.shape[1]}"
                )
            if hidden_b.shape != (head_dim,):
                raise ValueError(f"{module_name}.hidden 的 weight/bias 不匹配")
            if out_w.shape != (1, head_dim) or out_b.shape != (1,):
                raise ValueError(f"{module_name}.out 不是 hidden_dim -> 1 线性层")
            if self.use_linear_skip:
                skip_w = self._weight(f"{module_name}.skip.weight", ndim=2)
                skip_b = self._weight(f"{module_name}.skip.bias", ndim=1)
                if skip_w.shape != (1, global_dim) or skip_b.shape != (1,):
                    raise ValueError(f"{module_name}.skip 维度不正确")
            if mean_key not in self.norm or std_key not in self.norm:
                raise KeyError(f"norm_stats 缺少 {mean_key}/{std_key}")

    def _prepare_big_m(self) -> None:
        self.global_m_plus = None
        self.global_m_minus = None
        if self.payload.get("M_plus_global") is not None:
            self.global_m_plus = _as_numpy(
                self.payload["M_plus_global"]
            ).reshape(-1)
        if self.payload.get("M_minus_global") is not None:
            self.global_m_minus = _as_numpy(
                self.payload["M_minus_global"]
            ).reshape(-1)

        total_head_dim = sum(
            self.state[f"{name}.hidden.weight"].shape[0]
            for name, *_ in self.HEADS
        )
        for label, value in (
            ("M_plus_global", self.global_m_plus),
            ("M_minus_global", self.global_m_minus),
        ):
            if value is not None and value.size != total_head_dim:
                raise ValueError(
                    f"{label} 长度应为 {total_head_dim}，实际为 {value.size}"
                )

        self.sgc_m_plus = None
        self.sgc_m_minus = None
        plus_layers = self.payload.get("M_plus_gcn_layers") or []
        minus_layers = self.payload.get("M_minus_gcn_layers") or []
        if plus_layers:
            self.sgc_m_plus = _as_numpy(plus_layers[0])
        if minus_layers:
            self.sgc_m_minus = _as_numpy(minus_layers[0])
        expected = (self.num_nodes, self.hidden_dim)
        for label, value in (
            ("M_plus_gcn_layers[0]", self.sgc_m_plus),
            ("M_minus_gcn_layers[0]", self.sgc_m_minus),
        ):
            if value is not None and value.shape != expected:
                raise ValueError(f"{label} 形状应为 {expected}，实际为 {value.shape}")

    @property
    def binary_count(self) -> int:
        """该网络显式 Big-M 展开时的二进制变量数。"""

        head_count = sum(
            self.state[f"{name}.hidden.weight"].shape[0]
            for name, *_ in self.HEADS
        )
        return head_count + (
            self.num_nodes * self.hidden_dim if self.use_sgc_relu else 0
        )

    def _check_topology(self, topo_mask) -> None:
        if topo_mask is None:
            return
        mask = np.asarray(topo_mask, dtype=bool).reshape(-1)
        edge_count = len(self.edge_list)
        valid = mask.size == edge_count and bool(mask.all())
        # 兼容旧 IEEE-33 接口：前 32 条辐射支路闭合，后 5 条联络线断开。
        if mask.size == edge_count + 5:
            valid = bool(mask[:edge_count].all() and (~mask[edge_count:]).all())
        if not valid:
            raise ValueError(
                "Exp24 使用固定辐射状拓扑；topo_mask 不能改变训练时的支路状态"
            )

    def _validate_inputs(self, x_vars: Sequence[Sequence]) -> None:
        if len(x_vars) != self.num_nodes:
            raise ValueError(
                f"X_vars 第一维应为 {self.num_nodes}，实际为 {len(x_vars)}"
            )
        for i, row in enumerate(x_vars):
            if len(row) != 2:
                raise ValueError(f"X_vars[{i}] 应只包含 [P_net, Q_net]")

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
        """Evaluate the exact algebra embedded into the MILP, without Gurobi.

        ``x_net_mw_mvar`` is the raw net injection array before augmentation:
        shape ``(33, 2)`` or ``(N, 33, 2)``. The returned output order is
        ``[Vdev_total, Vworst, WorstI, Ploss_total]`` after denormalization.
        """

        x = np.asarray(x_net_mw_mvar, dtype=float)
        single = x.ndim == 2
        if single:
            x = x[None, :, :]
        if x.shape[1:] != (self.num_nodes, 2):
            raise ValueError(
                f"x_net_mw_mvar must have shape (33, 2) or (N, 33, 2), got {x.shape}"
            )

        p = x[:, :, 0]
        q = x[:, :, 1]
        x_aug = np.stack(
            [
                p,
                q,
                p @ self.downstream_matrix.T,
                q @ self.downstream_matrix.T,
                p @ self.path_power_matrix.T,
                q @ self.path_power_matrix.T,
            ],
            axis=2,
        )
        x_norm = (x_aug - self.x_mean.reshape(1, 1, -1)) / self.x_std.reshape(1, 1, -1)

        x_orders = np.einsum("kij,bjf->bkif", self.adj_powers, x_norm)
        x_multi = np.transpose(x_orders, (0, 2, 1, 3)).reshape(
            x.shape[0],
            self.num_nodes,
            self.order_count * self.in_features,
        )
        z_sgc = np.einsum("bif,df->bid", x_multi, self.sgc_weight) + self.sgc_bias
        hidden = np.maximum(z_sgc, 0.0) if self.use_sgc_relu else z_sgc
        global_values = np.concatenate(
            [
                hidden.reshape(x.shape[0], -1),
                x_norm.reshape(x.shape[0], -1),
            ],
            axis=1,
        )

        physical_outputs = []
        normalized_outputs = []
        head_z = {}
        for module_name, output_name, mean_key, std_key in self.HEADS:
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

            mean = _scalar(self.norm[mean_key], mean_key)
            std = _scalar(self.norm[std_key], std_key)
            normalized_outputs.append(y_norm)
            physical_outputs.append(y_norm * std + mean)
            head_z[output_name] = z

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
            "X_aug": x_aug[0] if single else x_aug,
            "X_norm": x_norm[0] if single else x_norm,
            "Z_sgc": z_sgc[0] if single else z_sgc,
            "G": global_values[0] if single else global_values,
            "head_Z": head_z,
        }

    def _add_relu(
        self,
        model,
        z_var,
        name: str,
        *,
        m_plus: float | None = None,
        m_minus: float | None = None,
    ) -> gp.Var:
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
        model.addConstr(
            y_var <= z_var + lower * (1.0 - active),
            name=f"{name}_ub_inactive",
        )
        model.addConstr(y_var <= upper * active, name=f"{name}_ub_active")
        return y_var

    def _add_augmented_inputs(self, model, x_vars, prefix):
        augmented = [
            [
                model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Xaug_{i}_{f}")
                for f in range(self.in_features)
            ]
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
                expr = self._linear_expr(
                    matrix[i], [x_vars[j][source_index] for j in range(self.num_nodes)]
                )
                model.addConstr(
                    augmented[i][feature] == expr,
                    name=f"{prefix}_{label}_{i}",
                )
        return augmented

    def _add_normalization(self, model, augmented, prefix):
        normalized = [
            [
                model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Xnorm_{i}_{f}")
                for f in range(self.in_features)
            ]
            for i in range(self.num_nodes)
        ]
        for i in range(self.num_nodes):
            for feature in range(self.in_features):
                inv_std = 1.0 / float(self.x_std[feature])
                expr = inv_std * augmented[i][feature] - (
                    float(self.x_mean[feature]) * inv_std
                )
                model.addConstr(
                    normalized[i][feature] == expr,
                    name=f"{prefix}_norm_{i}_{feature}",
                )
        return normalized

    def _add_sgc_encoder(self, model, normalized, prefix):
        orders = [
            [
                [
                    model.addVar(
                        lb=-GRB.INFINITY,
                        name=f"{prefix}_Xorder_{order}_{i}_{feature}",
                    )
                    for feature in range(self.in_features)
                ]
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
                    model.addConstr(
                        orders[order][i][feature] == expr,
                        name=f"{prefix}_order_{order}_{i}_{feature}",
                    )

        z_sgc = []
        hidden = []
        for i in range(self.num_nodes):
            # 与 X_orders.permute(0, 2, 1, 3).reshape(...) 一致：order 优先。
            sgc_input = [
                orders[order][i][feature]
                for order in range(self.order_count)
                for feature in range(self.in_features)
            ]
            z_row = []
            h_row = []
            for d in range(self.hidden_dim):
                z_var = model.addVar(
                    lb=-GRB.INFINITY, name=f"{prefix}_Zsgc_{i}_{d}"
                )
                expr = self._linear_expr(
                    self.sgc_weight[d], sgc_input, self.sgc_bias[d]
                )
                model.addConstr(z_var == expr, name=f"{prefix}_sgc_{i}_{d}")
                z_row.append(z_var)
                if self.use_sgc_relu:
                    plus = None if self.sgc_m_plus is None else self.sgc_m_plus[i, d]
                    minus = None if self.sgc_m_minus is None else self.sgc_m_minus[i, d]
                    h_row.append(
                        self._add_relu(
                            model,
                            z_var,
                            f"{prefix}_sgc_{i}_{d}",
                            m_plus=plus,
                            m_minus=minus,
                        )
                    )
                else:
                    h_row.append(z_var)
            z_sgc.append(z_row)
            hidden.append(h_row)
        return orders, z_sgc, hidden

    def _head_big_m(self, start: int, index: int):
        plus = (
            None
            if self.global_m_plus is None
            else self.global_m_plus[start + index]
        )
        minus = (
            None
            if self.global_m_minus is None
            else self.global_m_minus[start + index]
        )
        return plus, minus

    def _add_scalar_head(self, model, global_vars, head_spec, m_start, prefix):
        module_name, output_name, mean_key, std_key = head_spec
        hidden_w = self.state[f"{module_name}.hidden.weight"]
        hidden_b = self.state[f"{module_name}.hidden.bias"]
        out_w = self.state[f"{module_name}.out.weight"][0]
        out_b = float(self.state[f"{module_name}.out.bias"][0])

        z_vars = []
        relu_vars = []
        for d in range(hidden_w.shape[0]):
            z_var = model.addVar(
                lb=-GRB.INFINITY, name=f"{prefix}_{output_name}_Z_{d}"
            )
            expr = self._linear_expr(hidden_w[d], global_vars, hidden_b[d])
            model.addConstr(
                z_var == expr, name=f"{prefix}_{output_name}_hidden_{d}"
            )
            plus, minus = self._head_big_m(m_start, d)
            relu_var = self._add_relu(
                model,
                z_var,
                f"{prefix}_{output_name}_{d}",
                m_plus=plus,
                m_minus=minus,
            )
            z_vars.append(z_var)
            relu_vars.append(relu_var)

        normalized_output = model.addVar(
            lb=-GRB.INFINITY, name=f"{prefix}_{output_name}_norm"
        )
        output_expr = self._linear_expr(out_w, relu_vars, out_b)
        if self.use_linear_skip:
            skip_w = self.state[f"{module_name}.skip.weight"][0]
            skip_b = float(self.state[f"{module_name}.skip.bias"][0])
            output_expr += self._linear_expr(skip_w, global_vars, skip_b)
        model.addConstr(
            normalized_output == output_expr,
            name=f"{prefix}_{output_name}_normalized",
        )

        physical_output = model.addVar(
            lb=-GRB.INFINITY, name=f"{prefix}_{output_name}"
        )
        mean = _scalar(self.norm[mean_key], mean_key)
        std = _scalar(self.norm[std_key], std_key)
        if abs(std) < 1e-12:
            raise ValueError(f"{std_key} 为 0 或过小")
        model.addConstr(
            physical_output == normalized_output * std + mean,
            name=f"{prefix}_{output_name}_denorm",
        )
        return physical_output, normalized_output, z_vars, relu_vars

    def embed_sgcn_constraints(
        self,
        model,
        X_vars: Sequence[Sequence],
        topo_mask=None,
        *,
        name_prefix: str = "exp24",
    ) -> Exp24MILPOutputs:
        """把输入到四个输出的完整 Exp24 前向传播加入 ``model``。

        ``X_vars[i]`` 必须是节点 ``i`` 的原始净注入 ``[P_net, Q_net]``，
        元素可以是 Gurobi Var、LinExpr 或数值。返回值既可按四元组解包，
        也可使用 ``.Vdev_total/.Vworst/.WorstI/.Ploss_total`` 访问。
        """

        if gp is None:
            raise ModuleNotFoundError("gurobipy is required to embed SGCN constraints")
        if not isinstance(model, gp.Model):
            raise TypeError("model 必须是 gurobipy.Model")
        self._validate_inputs(X_vars)
        self._check_topology(topo_mask)
        prefix = str(name_prefix).strip().replace(" ", "_") or "exp24"

        augmented = self._add_augmented_inputs(model, X_vars, prefix)
        normalized = self._add_normalization(model, augmented, prefix)
        orders, z_sgc, hidden = self._add_sgc_encoder(model, normalized, prefix)

        global_vars = [
            hidden[i][d]
            for i in range(self.num_nodes)
            for d in range(self.hidden_dim)
        ] + [
            normalized[i][feature]
            for i in range(self.num_nodes)
            for feature in range(self.in_features)
        ]

        physical_outputs = []
        normalized_outputs = []
        head_z = {}
        head_relu = {}
        m_start = 0
        for spec in self.HEADS:
            output, output_norm, z_vars, relu_vars = self._add_scalar_head(
                model, global_vars, spec, m_start, prefix
            )
            physical_outputs.append(output)
            normalized_outputs.append(output_norm)
            head_z[spec[1]] = z_vars
            head_relu[spec[1]] = relu_vars
            m_start += len(z_vars)

        self.last_variables = {
            "X_aug": augmented,
            "X_norm": normalized,
            "X_orders": orders,
            "Z_sgc": z_sgc,
            "H": hidden,
            "G": global_vars,
            "head_Z": head_z,
            "head_relu": head_relu,
            "outputs_norm": normalized_outputs,
        }
        return Exp24MILPOutputs(*physical_outputs)

    # 与仓库旧版 GurobiMILPConverter 的调用名称兼容。
    def embed_gnn_constraints(
        self,
        model: gp.Model,
        X_vars: Sequence[Sequence],
        topo_mask=None,
        *,
        name_prefix: str = "exp24",
    ) -> Exp24MILPOutputs:
        return self.embed_sgcn_constraints(
            model, X_vars, topo_mask, name_prefix=name_prefix
        )


# 更明确的别名，便于后续无功优化脚本导入。
Exp24SGCNMILPConverter = SGCNMILPConverter


def _main() -> None:
    parser = argparse.ArgumentParser(description="检查 Exp24 ST-SGCN MILP 转换配置")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=str(DEFAULT_ENGINE_PATH),
        help="Exp24 MILP engine/best checkpoint 路径",
    )
    parser.add_argument(
        "--relu-formulation",
        choices=("big_m", "general"),
        default="big_m",
    )
    args = parser.parse_args()
    converter = SGCNMILPConverter(
        args.checkpoint, relu_formulation=args.relu_formulation
    )
    print(f"checkpoint: {converter.checkpoint_path}")
    print(
        "shape: "
        f"nodes={converter.num_nodes}, features={converter.in_features}, "
        f"orders={converter.order_count}, hidden={converter.hidden_dim}"
    )
    print(f"ReLU formulation: {converter.relu_formulation}")
    print(f"binary count: {converter.binary_count}")
    print("outputs: Vdev_total, Vworst, WorstI, Ploss_total")


if __name__ == "__main__":
    _main()

