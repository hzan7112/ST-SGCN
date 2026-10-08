import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import RADIAL_BRANCHES, build_adj_norm, build_adj_powers, sanitize_edge_list


class PairedScalarReadout(nn.Module):
    """One shared ReLU basis with two task-specific outputs and skips."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        first_name,
        second_name,
        use_linear_skip=True,
    ):
        super().__init__()
        self.output_names = (str(first_name), str(second_name))
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output_layers = nn.ModuleDict({
            name: nn.Linear(hidden_dim, 1)
            for name in self.output_names
        })
        self.skip_layers = (
            nn.ModuleDict({
                name: nn.Linear(input_dim, 1)
                for name in self.output_names
            })
            if use_linear_skip
            else None
        )

        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        for layer in self.output_layers.values():
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        if self.skip_layers is not None:
            for layer in self.skip_layers.values():
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        z = self.hidden(x)
        h = F.relu(z)
        outputs = []
        for name in self.output_names:
            y = self.output_layers[name](h)
            if self.skip_layers is not None:
                y = y + self.skip_layers[name](x)
            outputs.append(y)
        return tuple(outputs), z


class SharedPrivatePairedScalarReadout(nn.Module):
    """Shared ReLU basis plus a private ReLU adapter for each output."""

    def __init__(
        self,
        input_dim,
        shared_dim,
        private_dims,
        output_names,
        use_linear_skip=True,
    ):
        super().__init__()
        self.output_names = tuple(str(name) for name in output_names)
        self.shared_hidden = nn.Linear(input_dim, shared_dim)
        self.private_hidden = nn.ModuleDict({
            name: nn.Linear(input_dim, int(private_dims[name]))
            for name in self.output_names
        })
        self.output_layers = nn.ModuleDict({
            name: nn.Linear(shared_dim + int(private_dims[name]), 1)
            for name in self.output_names
        })
        self.skip_layers = (
            nn.ModuleDict({
                name: nn.Linear(input_dim, 1)
                for name in self.output_names
            })
            if use_linear_skip
            else None
        )

        nn.init.xavier_uniform_(self.shared_hidden.weight)
        nn.init.zeros_(self.shared_hidden.bias)
        for layers in (
            self.private_hidden.values(),
            self.output_layers.values(),
        ):
            for layer in layers:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        if self.skip_layers is not None:
            for layer in self.skip_layers.values():
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        z_shared = self.shared_hidden(x)
        h_shared = F.relu(z_shared)
        outputs = []
        private_z = []
        for name in self.output_names:
            z_private = self.private_hidden[name](x)
            h = torch.cat([h_shared, F.relu(z_private)], dim=1)
            y = self.output_layers[name](h)
            if self.skip_layers is not None:
                y = y + self.skip_layers[name](x)
            outputs.append(y)
            private_z.append(z_private)

        # This ordering is also used by the exported per-neuron Big-M arrays.
        z_all = torch.cat([z_shared, *private_z], dim=1)
        return tuple(outputs), z_all


class STSGCNPairedFourScalars(nn.Module):
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
        voltage_shared_head_dim=32,
        vdev_private_head_dim=8,
        vworst_private_head_dim=8,
        current_loss_pair_head_dim=48,
        include_order0=True,
        use_sgc_relu=False,
        use_linear_skip=True,
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.hidden_dim = int(hidden_dim)
        self.K = int(K)
        self.num_nodes = int(num_nodes)
        self.voltage_shared_head_dim = int(voltage_shared_head_dim)
        self.vdev_private_head_dim = int(vdev_private_head_dim)
        self.vworst_private_head_dim = int(vworst_private_head_dim)
        self.current_loss_pair_head_dim = int(current_loss_pair_head_dim)
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

        self.voltage_head = SharedPrivatePairedScalarReadout(
            global_input_dim,
            self.voltage_shared_head_dim,
            {
                "vdev": self.vdev_private_head_dim,
                "vworst": self.vworst_private_head_dim,
            },
            ("vdev", "vworst"),
            self.use_linear_skip,
        )
        self.current_loss_pair_head = PairedScalarReadout(
            global_input_dim,
            self.current_loss_pair_head_dim,
            "iworst",
            "ploss",
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

        (Vdev_pred, Vworst_pred), Z_voltage = self.voltage_head(G)
        (Iworst_pred, Ploss_pred), Z_current_loss = (
            self.current_loss_pair_head(G)
        )

        gcn_Z_list = [Z_sgc] if self.use_sgc_relu else []
        Z_global = torch.cat(
            [Z_voltage, Z_current_loss],
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
            self.voltage_shared_head_dim
            + self.vdev_private_head_dim
            + self.vworst_private_head_dim
            + self.current_loss_pair_head_dim
        )

        return {
            "sgc_binary": sgc_binary,
            "voltage_shared_head_binary": self.voltage_shared_head_dim,
            "vdev_private_head_binary": self.vdev_private_head_dim,
            "vworst_private_head_binary": self.vworst_private_head_dim,
            "current_loss_pair_head_binary": self.current_loss_pair_head_dim,
            "readout_binary": readout_binary,
            "total_binary": sgc_binary + readout_binary,
            "K": self.K,
            "order_count": self.order_count,
            "use_sgc_relu": self.use_sgc_relu,
            "predict_target": (
                "shared/private voltage plus shared current/loss readouts: "
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

