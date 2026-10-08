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


class TwoLayerScalarReadout(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim1=32,
        hidden_dim2=32,
        use_linear_skip=True,
    ):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim1)
        self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.out = nn.Linear(hidden_dim2, 1)
        self.skip = nn.Linear(input_dim, 1) if use_linear_skip else None

        for layer in [self.hidden1, self.hidden2, self.out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        if self.skip is not None:
            nn.init.xavier_uniform_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

    def forward(self, x):
        z1 = self.hidden1(x)
        h1 = F.relu(z1)
        z2 = self.hidden2(h1)
        y = self.out(F.relu(z2))

        if self.skip is not None:
            y = y + self.skip(x)

        return y, z1, z2


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
        voltage_in_features=6,
        current_in_features=6,
        loss_in_features=6,
        hidden_dim=24,
        K=5,
        edge_list=None,
        num_nodes=33,
        vdev_head_dim=32,
        vworst_head_dim=32,
        iworst_head_dim=32,
        ploss_head_dim=64,
        ploss_head_dim2=64,
        include_order0=True,
        use_sgc_relu=False,
        use_linear_skip=True,
        flow_in_features=None,
        in_features=None,
    ):
        super().__init__()

        self.voltage_in_features = int(voltage_in_features)
        if flow_in_features is not None:
            current_in_features = flow_in_features
            loss_in_features = flow_in_features
        self.current_in_features = int(current_in_features)
        self.loss_in_features = int(loss_in_features)
        self.hidden_dim = int(hidden_dim)
        self.K = int(K)
        self.num_nodes = int(num_nodes)
        self.vdev_head_dim = int(vdev_head_dim)
        self.vworst_head_dim = int(vworst_head_dim)
        self.iworst_head_dim = int(iworst_head_dim)
        self.ploss_head_dim = int(ploss_head_dim)
        self.ploss_head_dim2 = int(ploss_head_dim2)
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
        self.voltage_sgc_linear = nn.Linear(
            self.order_count * self.voltage_in_features,
            self.hidden_dim,
        )
        self.current_sgc_linear = nn.Linear(
            self.order_count * self.current_in_features,
            self.hidden_dim,
        )
        self.loss_sgc_linear = nn.Linear(
            self.order_count * self.loss_in_features,
            self.hidden_dim,
        )

        voltage_global_input_dim = (
            self.num_nodes * self.hidden_dim
            + self.num_nodes * self.voltage_in_features
        )
        current_global_input_dim = (
            self.num_nodes * self.hidden_dim
            + self.num_nodes * self.current_in_features
        )
        loss_global_input_dim = (
            self.num_nodes * self.hidden_dim
            + self.num_nodes * self.loss_in_features
        )

        self.vdev_head = DirectScalarReadout(
            voltage_global_input_dim,
            self.vdev_head_dim,
            self.use_linear_skip,
        )
        self.vworst_head = DirectScalarReadout(
            voltage_global_input_dim,
            self.vworst_head_dim,
            self.use_linear_skip,
        )
        self.iworst_head = DirectScalarReadout(
            current_global_input_dim,
            self.iworst_head_dim,
            self.use_linear_skip,
        )
        self.ploss_head = TwoLayerScalarReadout(
            loss_global_input_dim,
            self.ploss_head_dim,
            self.ploss_head_dim2,
            self.use_linear_skip,
        )

        for layer in [
            self.voltage_sgc_linear,
            self.current_sgc_linear,
            self.loss_sgc_linear,
        ]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _encode(self, X, linear):
        batch_size = X.size(0)
        A_powers = self.A_powers.to(X.device)

        X_orders = torch.einsum("kij,bjf->bkif", A_powers, X)
        X_multi = X_orders.permute(0, 2, 1, 3).reshape(
            batch_size,
            self.num_nodes,
            self.order_count * X.size(-1),
        )

        Z_sgc = linear(X_multi)
        H = F.relu(Z_sgc) if self.use_sgc_relu else Z_sgc

        G = torch.cat(
            [
                H.reshape(batch_size, -1),
                X.reshape(batch_size, -1),
            ],
            dim=1,
        )
        return G, Z_sgc

    def forward(self, X_voltage, X_current=None, X_loss=None):
        if X_current is None:
            if not isinstance(X_voltage, (tuple, list)) or len(X_voltage) not in {2, 3}:
                raise RuntimeError("forward expects X_voltage, X_current and X_loss.")
            if len(X_voltage) == 2:
                X_voltage, X_current = X_voltage
                X_loss = X_current
            else:
                X_voltage, X_current, X_loss = X_voltage
        elif X_loss is None:
            X_loss = X_current

        G_voltage, Z_voltage_sgc = self._encode(
            X_voltage,
            self.voltage_sgc_linear,
        )
        G_current, Z_current_sgc = self._encode(
            X_current,
            self.current_sgc_linear,
        )
        G_loss, Z_loss_sgc = self._encode(
            X_loss,
            self.loss_sgc_linear,
        )

        Vdev_pred, Z_vdev = self.vdev_head(G_voltage)
        Vworst_pred, Z_vworst = self.vworst_head(G_voltage)
        Iworst_pred, Z_iworst = self.iworst_head(G_current)
        Ploss_pred, Z_ploss_1, Z_ploss_2 = self.ploss_head(G_loss)

        gcn_Z_list = (
            [Z_voltage_sgc, Z_current_sgc, Z_loss_sgc]
            if self.use_sgc_relu
            else []
        )
        Z_global = torch.cat(
            [Z_vdev, Z_vworst, Z_iworst, Z_ploss_1, Z_ploss_2],
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
            3 * self.num_nodes * self.hidden_dim
            if self.use_sgc_relu
            else 0
        )
        readout_binary = (
            self.vdev_head_dim
            + self.vworst_head_dim
            + self.iworst_head_dim
            + self.ploss_head_dim
            + self.ploss_head_dim2
        )

        return {
            "sgc_binary": sgc_binary,
            "vdev_head_binary": self.vdev_head_dim,
            "vworst_head_binary": self.vworst_head_dim,
            "iworst_head_binary": self.iworst_head_dim,
            "ploss_head_binary": self.ploss_head_dim + self.ploss_head_dim2,
            "ploss_head_layer1_binary": self.ploss_head_dim,
            "ploss_head_layer2_binary": self.ploss_head_dim2,
            "readout_binary": readout_binary,
            "total_binary": sgc_binary + readout_binary,
            "K": self.K,
            "order_count": self.order_count,
            "use_sgc_relu": self.use_sgc_relu,
            "voltage_in_features": self.voltage_in_features,
            "current_in_features": self.current_in_features,
            "loss_in_features": self.loss_in_features,
            "predict_target": (
                "four direct scalars: "
                "Vdev_total, Vworst, WorstI, Ploss_total; "
                "Voltage, current and loss tasks use separate linear SGC encoders"
            ),
        }


def unpack_forward(model, X_voltage, X_current=None, X_loss=None):
    out = model(X_voltage, X_current, X_loss)
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

