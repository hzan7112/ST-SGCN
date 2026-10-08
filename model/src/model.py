import torch
import torch.nn as nn
import torch.nn.functional as F


RADIAL_BRANCHES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
    [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
    [15, 16], [16, 17], [1, 18], [18, 19], [19, 20], [20, 21],
    [2, 22], [22, 23], [23, 24], [5, 25], [25, 26], [26, 27],
    [27, 28], [28, 29], [29, 30], [30, 31], [31, 32]
]


def sanitize_edge_list(edge_list):
    return [[int(e[0]), int(e[1])] for e in edge_list]


class StaticTopologySGCN(nn.Module):
    """
    ST-SGCN: Static-Topology Simplified Graph Convolutional Network.

    核心思想：
    1. 静态拓扑下 A_norm 固定；
    2. 预先构造 A^0, A^1, ..., A^K；
    3. 去除普通 GCN 层间非线性传播；
    4. 将多阶邻接传播结果拼接后，用单层线性映射形成节点隐表示；
    5. 节点电压和支路裕度分别由轻量读出头预测。

    默认 use_sgc_relu=False，因此图传播部分不引入 ReLU 二元变量。
    MILP 嵌入时主要二元变量来自 node_head 和 edge_head。
    """

    def __init__(
        self,
        in_features=2,
        hidden_dim=64,
        num_layers=3,
        K=None,
        edge_list=None,
        num_nodes=33,
        node_relu_dim=48,
        edge_relu_dim=48,
        node_emb_dim=16,
        edge_emb_dim=16,
        use_residual=False,
        use_initial_anchor=False,
        use_jk=False,
        include_input_in_jk=True,
        use_linear_skip=True,
        include_order0=True,
        use_sgc_relu=False,
    ):
        super().__init__()

        edge_list = sanitize_edge_list(edge_list if edge_list is not None else RADIAL_BRANCHES)

        if K is None:
            K = num_layers if num_layers is not None else 3

        self.in_features = int(in_features)
        self.hidden_dim = int(hidden_dim)
        self.K = max(1, int(K))
        self.num_layers = 1
        self.requested_num_layers = num_layers
        self.num_nodes = int(num_nodes)
        self.edge_list = edge_list
        self.num_edges = len(edge_list)

        self.node_relu_dim = int(node_relu_dim)
        self.edge_relu_dim = int(edge_relu_dim)
        self.node_emb_dim = int(node_emb_dim)
        self.edge_emb_dim = int(edge_emb_dim)

        self.use_residual = False
        self.use_initial_anchor = False
        self.use_jk = False
        self.include_input_in_jk = include_input_in_jk
        self.use_linear_skip = bool(use_linear_skip)
        self.include_order0 = bool(include_order0)
        self.use_sgc_relu = bool(use_sgc_relu)

        self.src_indices = [e[0] for e in edge_list]
        self.dst_indices = [e[1] for e in edge_list]

        self.register_buffer("src_indices_tensor", torch.tensor(self.src_indices, dtype=torch.long))
        self.register_buffer("dst_indices_tensor", torch.tensor(self.dst_indices, dtype=torch.long))

        A_norm = self._build_adj(edge_list, self.num_nodes)
        A_powers = self._build_adj_powers(A_norm, self.K, self.include_order0)

        self.register_buffer("A_norm", A_norm)
        self.register_buffer("A_powers", A_powers)

        self.order_count = A_powers.size(0)
        self.sgc_input_dim = self.order_count * self.in_features

        self.sgc_linear = nn.Linear(self.sgc_input_dim, self.hidden_dim)

        self.node_emb = nn.Embedding(self.num_nodes, self.node_emb_dim)
        self.edge_emb = nn.Embedding(self.num_edges, self.edge_emb_dim)

        self.readout_dim = self.hidden_dim

        node_input_dim = self.readout_dim + self.node_emb_dim
        edge_input_dim = self.readout_dim * 3 + self.edge_emb_dim

        self.node_hidden = nn.Linear(node_input_dim, self.node_relu_dim)
        self.node_out = nn.Linear(self.node_relu_dim, 1)
        self.node_skip = nn.Linear(node_input_dim, 1) if self.use_linear_skip else None

        self.edge_hidden = nn.Linear(edge_input_dim, self.edge_relu_dim)
        self.edge_out = nn.Linear(self.edge_relu_dim, 1)
        self.edge_skip = nn.Linear(edge_input_dim, 1) if self.use_linear_skip else None

        self.reset_parameters()

    @staticmethod
    def _build_adj(edge_list, num_nodes):
        A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

        for u, v in edge_list:
            A[u, v] = 1.0
            A[v, u] = 1.0

        A_hat = A + torch.eye(num_nodes, dtype=torch.float32)
        deg = A_hat.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)

        return torch.diag(deg_inv_sqrt) @ A_hat @ torch.diag(deg_inv_sqrt)

    @staticmethod
    def _build_adj_powers(A_norm, K, include_order0=True):
        powers = []

        current = torch.eye(A_norm.size(0), dtype=torch.float32)

        if include_order0:
            powers.append(current.clone())

        for _ in range(K):
            current = current @ A_norm
            powers.append(current.clone())

        return torch.stack(powers, dim=0)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.sgc_linear.weight)
        nn.init.zeros_(self.sgc_linear.bias)

        nn.init.normal_(self.node_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.edge_emb.weight, mean=0.0, std=0.02)

        for layer in [self.node_hidden, self.node_out, self.edge_hidden, self.edge_out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        if self.node_skip is not None:
            nn.init.xavier_uniform_(self.node_skip.weight)
            nn.init.zeros_(self.node_skip.bias)

        if self.edge_skip is not None:
            nn.init.xavier_uniform_(self.edge_skip.weight)
            nn.init.zeros_(self.edge_skip.bias)

    def forward(self, X):
        batch_size = X.size(0)
        device = X.device

        A_powers = self.A_powers.to(device)

        X_orders = torch.einsum("kij,bjf->bkif", A_powers, X)
        X_multi = X_orders.permute(0, 2, 1, 3).reshape(
            batch_size,
            self.num_nodes,
            self.order_count * self.in_features,
        )

        Z_sgc = self.sgc_linear(X_multi)

        if self.use_sgc_relu:
            H_readout = F.relu(Z_sgc)
            gcn_Z_list = [Z_sgc]
        else:
            H_readout = Z_sgc
            gcn_Z_list = []

        node_ids = torch.arange(self.num_nodes, device=device)
        node_emb = self.node_emb(node_ids).unsqueeze(0).expand(batch_size, -1, -1)

        H_node = torch.cat([H_readout, node_emb], dim=-1)

        Z_node = self.node_hidden(H_node)
        V_pred = self.node_out(F.relu(Z_node)).squeeze(-1)

        if self.node_skip is not None:
            V_pred = V_pred + self.node_skip(H_node).squeeze(-1)

        src = self.src_indices_tensor.to(device)
        dst = self.dst_indices_tensor.to(device)

        H_src = H_readout.index_select(dim=1, index=src)
        H_dst = H_readout.index_select(dim=1, index=dst)
        H_diff = H_src - H_dst

        edge_ids = torch.arange(self.num_edges, device=device)
        edge_emb = self.edge_emb(edge_ids).unsqueeze(0).expand(batch_size, -1, -1)

        H_edge = torch.cat([H_src, H_dst, H_diff, edge_emb], dim=-1)

        Z_edge = self.edge_hidden(H_edge)
        I_pred = self.edge_out(F.relu(Z_edge)).squeeze(-1)

        if self.edge_skip is not None:
            I_pred = I_pred + self.edge_skip(H_edge).squeeze(-1)

        return V_pred, I_pred, gcn_Z_list, Z_node, Z_edge

    def get_frozen_adj_norm(self):
        return self.A_norm.detach().cpu().numpy()

    def get_frozen_adj_powers(self):
        return self.A_powers.detach().cpu().numpy()

    def get_frozen_node_emb(self):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            ids = torch.arange(self.num_nodes, device=self.node_emb.weight.device)
            out = self.node_emb(ids).cpu().numpy()
        if was_training:
            self.train()
        return out

    def get_frozen_edge_emb(self):
        was_training = self.training
        with torch.no_grad():
            self.eval()
            ids = torch.arange(self.num_edges, device=self.edge_emb.weight.device)
            out = self.edge_emb(ids).cpu().numpy()
        if was_training:
            self.train()
        return out

    def get_binary_count(self):
        sgc_binary = self.num_nodes * self.hidden_dim if self.use_sgc_relu else 0
        node_head_binary = self.num_nodes * self.node_relu_dim
        edge_head_binary = self.num_edges * self.edge_relu_dim

        total_binary = sgc_binary + node_head_binary + edge_head_binary

        return {
            "sgc_binary": sgc_binary,
            "node_head_binary": node_head_binary,
            "edge_head_binary": edge_head_binary,
            "total_binary": total_binary,
            "K": self.K,
            "order_count": self.order_count,
            "use_sgc_relu": self.use_sgc_relu,
        }


STSGCN = StaticTopologySGCN
ST_SGCN = StaticTopologySGCN
SGCModel = StaticTopologySGCN
StandardGCN = StaticTopologySGCN
GCNModel = StaticTopologySGCN

# 仅用于兼容旧训练脚本中的类名导入。
# 当前语义已经不是 DeepFirstOrderGCN，也不是多层非线性 GCN。
DeepFirstOrderGCN = StaticTopologySGCN
MultiOrderNonlinearGCN = StaticTopologySGCN