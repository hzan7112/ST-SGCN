import torch
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from model.src.model import PIGNN, ALL_BRANCHES


def _safe_torch_load(path, map_location):
    """
    兼容不同 PyTorch 版本：
    - 新版本优先使用 weights_only=True，减少 FutureWarning
    - 老版本不支持时自动回退
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class GurobiMILPConverter:
    def __init__(self, model_path, stats_path, hidden_dim=64, num_layers=18):
        """
        初始化转换器，加载 PyTorch 权重和标准化统计量
        """
        self.device = torch.device("cpu")

        self.stats = _safe_torch_load(stats_path, map_location=self.device)

        self.gnn = PIGNN(
            input_dim=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(self.device)

        state_dict = _safe_torch_load(model_path, map_location=self.device)
        self.gnn.load_state_dict(state_dict)
        self.gnn.eval()

        self.hidden_dim = hidden_dim
        self.num_nodes = 33
        self.num_edges = 37

        # Big-M 不宜太小，深层网络建议取更保守的数值
        self.M = 1e3

        self.src_nodes = [edge[0] for edge in ALL_BRANCHES]
        self.dst_nodes = [edge[1] for edge in ALL_BRANCHES]

    def _add_linear_layer(self, m, W, b, x_vars, name_prefix):
        """
        添加纯线性层：
            y = W x + b
        """
        out_dim, in_dim = W.shape
        if len(x_vars) != in_dim:
            raise ValueError(
                f"{name_prefix} 输入维度不匹配：期望 {in_dim}，实际 {len(x_vars)}"
            )

        y_vars = m.addVars(out_dim, lb=-GRB.INFINITY, name=f"{name_prefix}_y")

        for i in range(out_dim):
            expr = gp.LinExpr(float(b[i]))
            for j in range(in_dim):
                expr += float(W[i, j]) * x_vars[j]
            m.addConstr(y_vars[i] == expr, name=f"{name_prefix}_lin_{i}")

        return y_vars

    def _add_relu_layer(self, m, x_vars, name_prefix):
        """
        添加 ReLU 激活：
            y = max(0, x)
        Big-M 线性化形式：
            y >= x
            y <= x + M(1-z)
            y <= Mz
            y >= 0
        """
        dim = len(x_vars)
        y_vars = m.addVars(dim, lb=0.0, name=f"{name_prefix}_relu")
        z_vars = m.addVars(dim, vtype=GRB.BINARY, name=f"{name_prefix}_bin")

        for i in range(dim):
            m.addConstr(y_vars[i] >= x_vars[i], name=f"{name_prefix}_lb1_{i}")
            m.addConstr(y_vars[i] <= x_vars[i] + self.M * (1 - z_vars[i]), name=f"{name_prefix}_ub1_{i}")
            m.addConstr(y_vars[i] <= self.M * z_vars[i], name=f"{name_prefix}_ub2_{i}")

        return y_vars

    def embed_gnn_constraints(self, m, X_vars, topo_mask):
        """
        将整个 GNN 展开为 Gurobi 约束

        参数
        ----
        m : gurobipy.Model
        X_vars : list[list]
            形状 [33][2]，每个元素可以是 gurobi Var / LinExpr
            分别表示每个节点的注入 P、Q
        topo_mask : array-like
            长度 37 的布尔向量，表示支路开断状态

        返回
        ----
        V_phys : gurobi tupledict
            33 个节点电压预测（反归一化后的物理量）
        I_margin_phys : gurobi tupledict
            37 条支路电流裕度预测（反归一化后的物理量）
        """
        topo_mask = np.asarray(topo_mask, dtype=bool).reshape(-1)
        if len(topo_mask) != self.num_edges:
            raise ValueError(f"topo_mask 长度必须为 {self.num_edges}，当前为 {len(topo_mask)}")

        # =========================================================
        # 1. 输入归一化
        # =========================================================
        X_norm_vars = [
            [m.addVar(lb=-GRB.INFINITY, name=f"X_norm_{i}_{j}") for j in range(2)]
            for i in range(self.num_nodes)
        ]

        X_mean = np.asarray(self.stats["X_mean"], dtype=float).reshape(-1)
        X_std = np.asarray(self.stats["X_std"], dtype=float).reshape(-1)

        if len(X_mean) != 2 or len(X_std) != 2:
            raise ValueError(
                f"归一化参数维度错误：X_mean={X_mean.shape}, X_std={X_std.shape}，应为长度2"
            )

        for i in range(self.num_nodes):
            for j in range(2):
                std_j = float(X_std[j])
                mean_j = float(X_mean[j])

                if abs(std_j) < 1e-12:
                    raise ValueError(f"X_std[{j}] 过小或为0，当前值为 {std_j}")

                # 不直接写除法，改成标准线性表达式
                expr = (1.0 / std_j) * X_vars[i][j] - (mean_j / std_j)
                m.addConstr(X_norm_vars[i][j] == expr, name=f"norm_X_{i}_{j}")

        # =========================================================
        # 2. 输入嵌入层 H_0 = Linear(X_norm), H = ReLU(H_0)
        # =========================================================
        W_emb = self.gnn.input_embed.weight.detach().cpu().numpy()
        b_emb = self.gnn.input_embed.bias.detach().cpu().numpy()

        H_0_vars = []
        H_vars = []

        for i in range(self.num_nodes):
            x_i = [X_norm_vars[i][0], X_norm_vars[i][1]]

            h0_i = self._add_linear_layer(m, W_emb, b_emb, x_i, f"emb_node_{i}")
            H_0_vars.append(h0_i)

            h_i = self._add_relu_layer(m, h0_i, f"emb_relu_node_{i}")
            H_vars.append(h_i)

        # =========================================================
        # 3. 逐层展开 PureTopologyGCNLayer
        #
        # 对应你的 forward:
        #   H_trans = W_neigh(H)
        #   H_neigh = A_weighted @ H_trans
        #   H_evolve = H_neigh / 4 + W_self(H)
        #   H_target = (1-a) * H_evolve + a * W_root(H_0)
        #   Z = (1-b) * H + b * H_target + bias
        #   H_next = ReLU(Z)
        # =========================================================
        for layer_idx, layer in enumerate(self.gnn.gcn_layers):
            W_self = layer.W_self.weight.detach().cpu().numpy()
            W_neigh = layer.W_neigh.weight.detach().cpu().numpy()
            W_root = layer.W_root.weight.detach().cpu().numpy()
            bias = layer.bias.detach().cpu().numpy()

            edge_z = torch.clamp(layer.edge_pseudo_impedance, min=1e-4).detach().cpu().numpy()
            a_c = float(torch.sigmoid(layer.alpha).item())
            b_c = float(torch.sigmoid(layer.beta).item())

            H_next_vars = []

            for i in range(self.num_nodes):
                # H_neigh[i] = Σ_j A_weighted[i,j] * W_neigh(H_j)
                neigh_msg_expr = [gp.LinExpr(0.0) for _ in range(self.hidden_dim)]

                for e_idx in range(self.num_edges):
                    if not topo_mask[e_idx]:
                        continue

                    u = self.src_nodes[e_idx]
                    v = self.dst_nodes[e_idx]
                    w_val = float(edge_z[e_idx])

                    if u == i or v == i:
                        neighbor = v if u == i else u

                        for d in range(self.hidden_dim):
                            expr_d = neigh_msg_expr[d]
                            for k in range(self.hidden_dim):
                                expr_d += w_val * float(W_neigh[d, k]) * H_vars[neighbor][k]

                Z_i = m.addVars(self.hidden_dim, lb=-GRB.INFINITY, name=f"L{layer_idx}_Z_node_{i}")

                for d in range(self.hidden_dim):
                    self_expr = gp.LinExpr(0.0)
                    root_expr = gp.LinExpr(0.0)

                    for k in range(self.hidden_dim):
                        self_expr += float(W_self[d, k]) * H_vars[i][k]
                        root_expr += float(W_root[d, k]) * H_0_vars[i][k]

                    evolve_expr = (1.0 / 4.0) * neigh_msg_expr[d] + self_expr
                    target_expr = (1.0 - a_c) * evolve_expr + a_c * root_expr
                    z_expr = (1.0 - b_c) * H_vars[i][d] + b_c * target_expr + float(bias[d])

                    m.addConstr(Z_i[d] == z_expr, name=f"L{layer_idx}_Zeq_node_{i}_{d}")

                h_next_i = self._add_relu_layer(m, Z_i, f"L{layer_idx}_relu_node_{i}")
                H_next_vars.append(h_next_i)

            H_vars = H_next_vars

        # =========================================================
        # 4. 节点读出头 node_head
        #    node_head = Linear(hidden_dim -> hidden_dim//2) + ReLU + Linear(... -> 1)
        # =========================================================
        W_n1 = self.gnn.node_head[0].weight.detach().cpu().numpy()
        b_n1 = self.gnn.node_head[0].bias.detach().cpu().numpy()
        W_n2 = self.gnn.node_head[2].weight.detach().cpu().numpy()
        b_n2 = self.gnn.node_head[2].bias.detach().cpu().numpy()

        V_pred_norm = m.addVars(self.num_nodes, lb=-GRB.INFINITY, name="V_pred_norm")

        for i in range(self.num_nodes):
            h_n1 = self._add_linear_layer(m, W_n1, b_n1, H_vars[i], f"nodehead_l1_{i}")
            h_n1_relu = self._add_relu_layer(m, h_n1, f"nodehead_relu_{i}")
            h_n2 = self._add_linear_layer(m, W_n2, b_n2, h_n1_relu, f"nodehead_l2_{i}")

            m.addConstr(V_pred_norm[i] == h_n2[0], name=f"nodehead_out_{i}")

        # =========================================================
        # 5. 支路读出头 edge_head
        #
        # 你的真实模型是：
        #   H_src
        #   H_dst
        #   H_diff = H_src - H_dst
        #   H_edge = cat([H_src, H_dst, H_diff], dim=-1)
        #   H_edge = H_edge * valid_mask
        #   margin_pred = edge_head(H_edge)
        #
        # 所以这里不能再使用 edge_emb
        # =========================================================
        W_e1 = self.gnn.edge_head[0].weight.detach().cpu().numpy()
        b_e1 = self.gnn.edge_head[0].bias.detach().cpu().numpy()
        W_e2 = self.gnn.edge_head[2].weight.detach().cpu().numpy()
        b_e2 = self.gnn.edge_head[2].bias.detach().cpu().numpy()

        margin_pred_norm = m.addVars(self.num_edges, lb=-GRB.INFINITY, name="margin_pred_norm")

        for e_idx in range(self.num_edges):
            u = self.src_nodes[e_idx]
            v = self.dst_nodes[e_idx]

            edge_input = []

            # H_src
            for d in range(self.hidden_dim):
                if topo_mask[e_idx]:
                    edge_input.append(H_vars[u][d])
                else:
                    edge_input.append(0.0)

            # H_dst
            for d in range(self.hidden_dim):
                if topo_mask[e_idx]:
                    edge_input.append(H_vars[v][d])
                else:
                    edge_input.append(0.0)

            # H_diff = H_src - H_dst
            for d in range(self.hidden_dim):
                if topo_mask[e_idx]:
                    diff_var = m.addVar(lb=-GRB.INFINITY, name=f"edge_diff_{e_idx}_{d}")
                    m.addConstr(diff_var == H_vars[u][d] - H_vars[v][d], name=f"edge_diff_eq_{e_idx}_{d}")
                    edge_input.append(diff_var)
                else:
                    edge_input.append(0.0)

            # 这里输入维度应当正好等于 hidden_dim * 3
            h_e1 = self._add_linear_layer(m, W_e1, b_e1, edge_input, f"edgehead_l1_{e_idx}")
            h_e1_relu = self._add_relu_layer(m, h_e1, f"edgehead_relu_{e_idx}")
            h_e2 = self._add_linear_layer(m, W_e2, b_e2, h_e1_relu, f"edgehead_l2_{e_idx}")

            m.addConstr(margin_pred_norm[e_idx] == h_e2[0], name=f"edgehead_out_{e_idx}")

        # =========================================================
        # 6. 输出反归一化
        # =========================================================
        V_mean = float(self.stats["Y_V_mean"])
        V_std = float(self.stats["Y_V_std"])
        I_mean = float(self.stats["Y_I_mean"])
        I_std = float(self.stats["Y_I_std"])

        if abs(V_std) < 1e-12:
            raise ValueError(f"Y_V_std 过小或为0，当前值为 {V_std}")
        if abs(I_std) < 1e-12:
            raise ValueError(f"Y_I_std 过小或为0，当前值为 {I_std}")

        V_phys = m.addVars(self.num_nodes, lb=-GRB.INFINITY, name="V_phys")
        I_margin_phys = m.addVars(self.num_edges, lb=-GRB.INFINITY, name="I_margin_phys")

        for i in range(self.num_nodes):
            m.addConstr(V_phys[i] == V_pred_norm[i] * V_std + V_mean, name=f"denorm_V_{i}")

        for e_idx in range(self.num_edges):
            m.addConstr(I_margin_phys[e_idx] == margin_pred_norm[e_idx] * I_std + I_mean, name=f"denorm_I_{e_idx}")

        return V_phys, I_margin_phys