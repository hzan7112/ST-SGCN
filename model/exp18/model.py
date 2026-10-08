import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import RADIAL_BRANCHES, build_adj_norm, build_adj_powers, sanitize_edge_list


class STSGCNWorstI(nn.Module):
    def __init__(
        self,
        in_features=6,
        hidden_dim=24,
        K=4,
        edge_list=None,
        num_nodes=33,
        node_relu_dim=1,
        global_relu_dim=8,
        include_order0=True,
        use_sgc_relu=False,
        use_linear_skip=True,
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.hidden_dim = int(hidden_dim)
        self.K = int(K)
        self.num_nodes = int(num_nodes)
        self.node_relu_dim = int(node_relu_dim)
        self.global_relu_dim = int(global_relu_dim)
        self.include_order0 = bool(include_order0)
        self.use_sgc_relu = bool(use_sgc_relu)
        self.use_linear_skip = bool(use_linear_skip)

        edge_list = sanitize_edge_list(edge_list if edge_list is not None else RADIAL_BRANCHES)
        self.edge_list = edge_list

        A_norm = build_adj_norm(edge_list, self.num_nodes)
        A_powers = build_adj_powers(A_norm, self.K, self.include_order0)

        self.register_buffer("A_norm", A_norm)
        self.register_buffer("A_powers", A_powers)

        self.order_count = A_powers.size(0)
        self.sgc_linear = nn.Linear(self.order_count * self.in_features, self.hidden_dim)

        self.node_emb = nn.Embedding(self.num_nodes, 4)

        node_input_dim = self.hidden_dim + 4

        self.node_hidden = nn.Linear(node_input_dim, self.node_relu_dim)
        self.node_out = nn.Linear(self.node_relu_dim, 1)
        self.node_skip = nn.Linear(node_input_dim, 1) if self.use_linear_skip else None

        global_input_dim = self.num_nodes * self.hidden_dim + self.num_nodes * self.in_features

        self.global_hidden = nn.Linear(global_input_dim, self.global_relu_dim)
        self.global_out = nn.Linear(self.global_relu_dim, 1)
        self.global_ploss_out = nn.Linear(self.global_relu_dim, 1)
        self.global_skip = nn.Linear(global_input_dim, 1) if self.use_linear_skip else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.sgc_linear.weight)
        nn.init.zeros_(self.sgc_linear.bias)

        nn.init.normal_(self.node_emb.weight, mean=0.0, std=0.02)

        for layer in [
            self.node_hidden,
            self.node_out,
            self.global_hidden,
            self.global_out,
            self.global_ploss_out,
        ]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        if self.node_skip is not None:
            nn.init.xavier_uniform_(self.node_skip.weight)
            nn.init.zeros_(self.node_skip.bias)

        if self.global_skip is not None:
            nn.init.xavier_uniform_(self.global_skip.weight)
            nn.init.zeros_(self.global_skip.bias)

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
            H = F.relu(Z_sgc)
            gcn_Z_list = [Z_sgc]
        else:
            H = Z_sgc
            gcn_Z_list = []

        node_ids = torch.arange(self.num_nodes, device=device)
        node_emb = self.node_emb(node_ids).unsqueeze(0).expand(batch_size, -1, -1)

        H_node = torch.cat([H, node_emb], dim=-1)

        Z_node = self.node_hidden(H_node)
        V_pred = self.node_out(F.relu(Z_node)).squeeze(-1)

        if self.node_skip is not None:
            V_pred = V_pred + self.node_skip(H_node).squeeze(-1)

        G = torch.cat(
            [
                H.reshape(batch_size, -1),
                X.reshape(batch_size, -1),
            ],
            dim=-1,
        )

        Z_global = self.global_hidden(G)
        H_global = F.relu(Z_global)
        YI_worst_pred = self.global_out(H_global)
        Ploss_pred = self.global_ploss_out(H_global)

        if self.global_skip is not None:
            YI_worst_pred = YI_worst_pred + self.global_skip(G)

        return V_pred, YI_worst_pred, Ploss_pred, gcn_Z_list, Z_node, Z_global

    def get_frozen_adj_norm(self):
        return self.A_norm.detach().cpu().numpy()

    def get_frozen_adj_powers(self):
        return self.A_powers.detach().cpu().numpy()

    def get_binary_count(self):
        sgc_binary = self.num_nodes * self.hidden_dim if self.use_sgc_relu else 0
        node_head_binary = self.num_nodes * self.node_relu_dim
        global_head_binary = self.global_relu_dim

        return {
            "sgc_binary": sgc_binary,
            "node_head_binary": node_head_binary,
            "global_head_binary": global_head_binary,
            "total_binary": sgc_binary + node_head_binary + global_head_binary,
            "K": self.K,
            "order_count": self.order_count,
            "use_sgc_relu": self.use_sgc_relu,
            "predict_target": "YI_worst=max(YI), Ploss_total shares global nonlinear readout",
        }


def zero_init_voltage_head(model):
    for name in ["node_out", "node_skip"]:
        layer = getattr(model, name, None)

        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class VoltageLinearResidualWrapper(nn.Module):
    def __init__(self, base_model, W_vlin, b_vlin):
        super().__init__()

        self.base = base_model
        self.register_buffer("W_vlin", W_vlin.float())
        self.register_buffer("b_vlin", b_vlin.float())

    def forward(self, X):
        out = self.base(X)

        if not isinstance(out, tuple) or len(out) < 6:
            raise RuntimeError("base model forward must return V_res, YI_worst_pred, Ploss_pred, gcn_Z_list, Z_node, Z_global.")

        V_res, YI_worst_pred, Ploss_pred, gcn_Z_list, Z_node, Z_global = out[:6]

        V_lin = X.reshape(X.size(0), -1) @ self.W_vlin.T + self.b_vlin

        if V_res.shape[1] == 33:
            V_total = V_res.clone()
            V_total[:, 1:] = V_lin + V_res[:, 1:]
        else:
            V_total = V_lin + V_res

        return V_total, YI_worst_pred, Ploss_pred, gcn_Z_list, Z_node, Z_global

    def get_frozen_adj_norm(self):
        return self.base.get_frozen_adj_norm()

    def get_frozen_adj_powers(self):
        return self.base.get_frozen_adj_powers()

    def get_binary_count(self):
        return self.base.get_binary_count()


def unpack_forward(model, X):
    out = model(X)

    if isinstance(out, tuple):
        return out

    raise RuntimeError("model.forward must return V_pred, YI_worst_pred, Ploss_pred, gcn_Z_list, Z_node, Z_global.")


def denorm_outputs(Vn, YI_worst_n, Ploss_n, norm):
    device = Vn.device
    B = Vn.shape[0]

    YV_mean = norm["YV_mean_wo_slack"].to(device).view(1, -1)
    YV_std = norm["YV_std_wo_slack"].to(device).view(1, -1)

    YI_mean = norm["YI_worst_mean"].to(device).view(1, -1)
    YI_std = norm["YI_worst_std"].to(device).view(1, -1)
    YP_mean = norm["YP_loss_mean"].to(device).view(1, -1)
    YP_std = norm["YP_loss_std"].to(device).view(1, -1)

    V = torch.ones((B, 33), dtype=Vn.dtype, device=device)

    if Vn.shape[1] == 33:
        V[:, 1:] = Vn[:, 1:] * YV_std + YV_mean
    else:
        V[:, 1:] = Vn * YV_std + YV_mean

    YI_worst = YI_worst_n * YI_std + YI_mean
    Ploss = Ploss_n * YP_std + YP_mean

    return V, YI_worst, Ploss

