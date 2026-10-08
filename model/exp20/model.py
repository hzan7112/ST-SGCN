import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import RADIAL_BRANCHES, build_adj_norm, build_adj_powers, sanitize_edge_list


class DirectScalarReadout(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_linear_skip=True):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.skip = nn.Linear(input_dim, 1) if use_linear_skip else None

        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        if self.skip is not None:
            nn.init.xavier_uniform_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

    def forward(self, x):
        z = self.hidden(x)
        y = self.out(F.relu(z))

        if self.skip is not None:
            y = y + self.skip(x)

        return y, z


class STSGCNFourDirectScalars(nn.Module):
    """
    共享 SGC 编码器，三个独立系统级标量读出头。

    输出仅包含：
        1. Vdev_total
        2. Vworst
        3. WorstI
        4. Ploss_total

    模型不预测任何节点电压或支路电流。
    """

    def __init__(
        self,
        in_features=6,
        hidden_dim=24,
        K=4,
        edge_list=None,
        num_nodes=33,
        vdev_head_dim=32,
        vworst_head_dim=32,
        iworst_head_dim=32,
        ploss_head_dim=32,
        include_order0=True,
        use_sgc_relu=False,
        use_linear_skip=True,
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.hidden_dim = int(hidden_dim)
        self.K = int(K)
        self.num_nodes = int(num_nodes)
        self.vdev_head_dim = int(vdev_head_dim)
        self.vworst_head_dim = int(vworst_head_dim)
        self.iworst_head_dim = int(iworst_head_dim)
        self.ploss_head_dim = int(ploss_head_dim)
        self.include_order0 = bool(include_order0)
        self.use_sgc_relu = bool(use_sgc_relu)
        self.use_linear_skip = bool(use_linear_skip)

        self.edge_list = sanitize_edge_list(
            edge_list if edge_list is not None else RADIAL_BRANCHES
        )

        A_norm = build_adj_norm(self.edge_list, self.num_nodes)
        A_powers = build_adj_powers(A_norm, self.K, self.include_order0)

        self.register_buffer("A_norm", A_norm)
        self.register_buffer("A_powers", A_powers)

        self.order_count = int(A_powers.size(0))
        self.sgc_linear = nn.Linear(
            self.order_count * self.in_features,
            self.hidden_dim,
        )

        global_input_dim = (
            self.num_nodes * self.hidden_dim
            + self.num_nodes * self.in_features
        )

        self.vdev_head = DirectScalarReadout(
            global_input_dim,
            self.vdev_head_dim,
            self.use_linear_skip,
        )
        self.vworst_head = DirectScalarReadout(
            global_input_dim,
            self.vworst_head_dim,
            self.use_linear_skip,
        )
        self.iworst_head = DirectScalarReadout(
            global_input_dim,
            self.iworst_head_dim,
            self.use_linear_skip,
        )
        self.ploss_head = DirectScalarReadout(
            global_input_dim,
            self.ploss_head_dim,
            self.use_linear_skip,
        )

        nn.init.xavier_uniform_(self.sgc_linear.weight)
        nn.init.zeros_(self.sgc_linear.bias)

    def forward(self, X):
        batch_size = X.size(0)
        A_powers = self.A_powers.to(X.device)

        X_orders = torch.einsum("kij,bjf->bkif", A_powers, X)
        X_multi = X_orders.permute(0, 2, 1, 3).reshape(
            batch_size,
            self.num_nodes,
            self.order_count * self.in_features,
        )

        Z_sgc = self.sgc_linear(X_multi)
        H = F.relu(Z_sgc) if self.use_sgc_relu else Z_sgc

        G = torch.cat(
            [
                H.reshape(batch_size, -1),
                X.reshape(batch_size, -1),
            ],
            dim=1,
        )

        Vdev_pred, Z_vdev = self.vdev_head(G)
        Vworst_pred, Z_vworst = self.vworst_head(G)
        Iworst_pred, Z_iworst = self.iworst_head(G)
        Ploss_pred, Z_ploss = self.ploss_head(G)

        gcn_Z_list = [Z_sgc] if self.use_sgc_relu else []
        Z_global = torch.cat(
            [Z_vdev, Z_vworst, Z_iworst, Z_ploss],
            dim=1,
        )

        return (
            Vdev_pred,
            Vworst_pred,
            Iworst_pred,
            Ploss_pred,
            gcn_Z_list,
            None,
            Z_global,
        )

    def get_frozen_adj_norm(self):
        return self.A_norm.detach().cpu().numpy()

    def get_frozen_adj_powers(self):
        return self.A_powers.detach().cpu().numpy()

    def get_binary_count(self):
        sgc_binary = (
            self.num_nodes * self.hidden_dim
            if self.use_sgc_relu
            else 0
        )
        readout_binary = (
            self.vdev_head_dim
            + self.vworst_head_dim
            + self.iworst_head_dim
            + self.ploss_head_dim
        )

        return {
            "sgc_binary": sgc_binary,
            "vdev_head_binary": self.vdev_head_dim,
            "vworst_head_binary": self.vworst_head_dim,
            "iworst_head_binary": self.iworst_head_dim,
            "ploss_head_binary": self.ploss_head_dim,
            "readout_binary": readout_binary,
            "total_binary": sgc_binary + readout_binary,
            "K": self.K,
            "order_count": self.order_count,
            "use_sgc_relu": self.use_sgc_relu,
            "predict_target": (
                "four direct scalars: "
                "Vdev_total, Vworst, WorstI, Ploss_total"
            ),
        }


def unpack_forward(model, X):
    out = model(X)
    if isinstance(out, tuple) and len(out) >= 7:
        return out
    raise RuntimeError(
        "model.forward must return "
        "Vdev, Vworst, WorstI, Ploss, gcn_Z_list, Z_node, Z_global."
    )


def denorm_outputs(Vdev_n, Vworst_n, Iworst_n, Ploss_n, norm):
    device = Vdev_n.device

    def denorm(y, mean_key, std_key):
        mean = norm[mean_key].to(device).view(1, -1)
        std = norm[std_key].to(device).view(1, -1)
        return y * std + mean

    return (
        denorm(Vdev_n, "YV_dev_mean", "YV_dev_std"),
        denorm(Vworst_n, "YV_worst_mean", "YV_worst_std"),
        denorm(Iworst_n, "YI_worst_mean", "YI_worst_std"),
        denorm(Ploss_n, "YP_loss_mean", "YP_loss_std"),
    )

