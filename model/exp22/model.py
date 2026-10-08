import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (
    RADIAL_BRANCHES,
    build_directed_adj_powers,
    sanitize_edge_list,
)


BRANCH_FEATURE_NAMES = (
    "P_down_child",
    "Q_down_child",
    "P_parent",
    "Q_parent",
    "P_child",
    "Q_child",
    "P_parent_minus_child",
    "Q_parent_minus_child",
    "depth_normalized",
    "subtree_size_normalized",
    "leaf_branch",
)


class PairedScalarReadout(nn.Module):
    """One shared ReLU basis with two task-specific outputs and skips."""

    def __init__(self, input_dim, hidden_dim, first_name, second_name, use_linear_skip=True):
        super().__init__()
        self.output_names = (str(first_name), str(second_name))
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output_layers = nn.ModuleDict({
            name: nn.Linear(hidden_dim, 1) for name in self.output_names
        })
        self.skip_layers = (
            nn.ModuleDict({name: nn.Linear(input_dim, 1) for name in self.output_names})
            if use_linear_skip else None
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        for group in (self.output_layers, self.skip_layers):
            if group is not None:
                for layer in group.values():
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
            nn.ModuleDict({name: nn.Linear(input_dim, 1) for name in self.output_names})
            if use_linear_skip else None
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.shared_hidden.weight)
        nn.init.zeros_(self.shared_hidden.bias)
        for group in (self.private_hidden, self.output_layers, self.skip_layers):
            if group is not None:
                for layer in group.values():
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
        return tuple(outputs), torch.cat([z_shared, *private_z], dim=1)


class DirectedHeteroSGCNFourScalars(nn.Module):
    """MILP-friendly directed bus/branch heterogeneous surrogate.

    The encoder and bus/branch coupling layers are linear.  ReLU is confined
    to the two system-level readouts, keeping the default binary count at 96.
    """

    def __init__(
        self,
        in_features=6,
        bus_hidden_dim=24,
        branch_hidden_dim=24,
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
        self.bus_hidden_dim = int(bus_hidden_dim)
        self.branch_hidden_dim = int(branch_hidden_dim)
        self.hidden_dim = self.bus_hidden_dim  # converter compatibility
        self.K = int(K)
        self.num_nodes = int(num_nodes)
        self.num_branches = self.num_nodes - 1
        self.branch_feature_dim = len(BRANCH_FEATURE_NAMES)
        self.voltage_shared_head_dim = int(voltage_shared_head_dim)
        self.vdev_private_head_dim = int(vdev_private_head_dim)
        self.vworst_private_head_dim = int(vworst_private_head_dim)
        self.current_loss_pair_head_dim = int(current_loss_pair_head_dim)
        self.include_order0 = bool(include_order0)
        self.use_sgc_relu = bool(use_sgc_relu)
        self.use_linear_skip = bool(use_linear_skip)
        self.edge_list = sanitize_edge_list(edge_list or RADIAL_BRANCHES)

        topology = build_directed_adj_powers(
            self.edge_list,
            self.K,
            self.num_nodes,
            root=0,
            include_order0=self.include_order0,
        )
        for name in (
            "A_up", "A_down", "A_powers", "parent_nodes", "child_nodes",
            "branch_topology",
        ):
            self.register_buffer(name, topology[name])

        parent_incidence = torch.zeros(self.num_nodes, self.num_branches)
        child_incidence = torch.zeros(self.num_nodes, self.num_branches)
        branch_ids = torch.arange(self.num_branches)
        parent_incidence[self.parent_nodes, branch_ids] = 1.0
        child_incidence[self.child_nodes, branch_ids] = 1.0
        self.register_buffer("parent_incidence", parent_incidence)
        self.register_buffer("child_incidence", child_incidence)

        self.order_count = int(self.A_powers.size(0))
        self.sgc_linear = nn.Linear(
            self.order_count * self.in_features,
            self.bus_hidden_dim,
        )
        self.branch_input_linear = nn.Linear(
            self.branch_feature_dim,
            self.branch_hidden_dim,
        )
        self.branch_coupling_linear = nn.Linear(
            self.branch_hidden_dim + 2 * self.bus_hidden_dim,
            self.branch_hidden_dim,
        )
        self.bus_coupling_linear = nn.Linear(
            self.bus_hidden_dim + 2 * self.branch_hidden_dim,
            self.bus_hidden_dim,
        )

        bus_global_dim = self.num_nodes * (self.bus_hidden_dim + self.in_features)
        branch_global_dim = self.num_branches * (
            self.branch_hidden_dim + self.branch_feature_dim
        )
        self.voltage_head = SharedPrivatePairedScalarReadout(
            bus_global_dim,
            self.voltage_shared_head_dim,
            {"vdev": self.vdev_private_head_dim, "vworst": self.vworst_private_head_dim},
            ("vdev", "vworst"),
            self.use_linear_skip,
        )
        self.current_loss_pair_head = PairedScalarReadout(
            branch_global_dim,
            self.current_loss_pair_head_dim,
            "iworst",
            "ploss",
            self.use_linear_skip,
        )
        self._reset_encoder_parameters()

    def _reset_encoder_parameters(self):
        for layer in (
            self.sgc_linear,
            self.branch_input_linear,
            self.branch_coupling_linear,
            self.bus_coupling_linear,
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def build_branch_features(self, X):
        parent = self.parent_nodes
        child = self.child_nodes
        parent_x = X[:, parent, :]
        child_x = X[:, child, :]
        topology = self.branch_topology.unsqueeze(0).expand(X.size(0), -1, -1)
        return torch.cat(
            [
                child_x[:, :, 2:4],
                parent_x[:, :, 0:2],
                child_x[:, :, 0:2],
                parent_x[:, :, 0:2] - child_x[:, :, 0:2],
                topology,
            ],
            dim=2,
        )

    def forward(self, X):
        batch_size = X.size(0)
        X_orders = torch.einsum("kij,bjf->bkif", self.A_powers, X)
        X_multi = X_orders.permute(0, 2, 1, 3).reshape(
            batch_size, self.num_nodes, self.order_count * self.in_features
        )
        Z_bus_base = self.sgc_linear(X_multi)
        H_bus_base = F.relu(Z_bus_base) if self.use_sgc_relu else Z_bus_base

        X_branch = self.build_branch_features(X)
        H_branch_base = self.branch_input_linear(X_branch)
        H_branch = self.branch_coupling_linear(
            torch.cat(
                [
                    H_branch_base,
                    H_bus_base[:, self.parent_nodes, :],
                    H_bus_base[:, self.child_nodes, :],
                ],
                dim=2,
            )
        )

        outgoing = torch.einsum("ne,bed->bnd", self.parent_incidence, H_branch)
        incoming = torch.einsum("ne,bed->bnd", self.child_incidence, H_branch)
        H_bus = self.bus_coupling_linear(
            torch.cat([H_bus_base, outgoing, incoming], dim=2)
        )

        G_bus = torch.cat([H_bus.reshape(batch_size, -1), X.reshape(batch_size, -1)], dim=1)
        G_branch = torch.cat(
            [H_branch.reshape(batch_size, -1), X_branch.reshape(batch_size, -1)], dim=1
        )
        (Vdev_pred, Vworst_pred), Z_voltage = self.voltage_head(G_bus)
        (Iworst_pred, Ploss_pred), Z_current_loss = self.current_loss_pair_head(G_branch)

        gcn_Z_list = [Z_bus_base] if self.use_sgc_relu else []
        Z_global = torch.cat([Z_voltage, Z_current_loss], dim=1)
        return (
            Vdev_pred,
            Vworst_pred,
            Iworst_pred,
            Ploss_pred,
            gcn_Z_list,
            None,
            Z_global,
        )

    def get_frozen_adj_powers(self):
        return self.A_powers.detach().cpu().numpy()

    def get_binary_count(self):
        sgc_binary = self.num_nodes * self.bus_hidden_dim if self.use_sgc_relu else 0
        readout_binary = (
            self.voltage_shared_head_dim + self.vdev_private_head_dim
            + self.vworst_private_head_dim + self.current_loss_pair_head_dim
        )
        return {
            "sgc_binary": sgc_binary,
            "readout_binary": readout_binary,
            "total_binary": sgc_binary + readout_binary,
            "voltage_shared_head_binary": self.voltage_shared_head_dim,
            "vdev_private_head_binary": self.vdev_private_head_dim,
            "vworst_private_head_binary": self.vworst_private_head_dim,
            "current_loss_pair_head_binary": self.current_loss_pair_head_dim,
            "K": self.K,
            "order_count": self.order_count,
            "num_branches": self.num_branches,
            "branch_feature_dim": self.branch_feature_dim,
            "use_sgc_relu": self.use_sgc_relu,
            "predict_target": "bus: Vdev/Vworst; branch: WorstI/Ploss",
        }


def unpack_forward(model, X):
    out = model(X)
    if isinstance(out, tuple) and len(out) >= 7:
        return out
    raise RuntimeError(
        "model.forward must return Vdev, Vworst, WorstI, Ploss, "
        "gcn_Z_list, Z_node, Z_global."
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
