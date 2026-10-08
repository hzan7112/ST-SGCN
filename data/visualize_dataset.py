import os

# ==========================================
# 0. 环境变量：必须放在第三方库导入之前
# ==========================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, deque, Counter

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ==========================================
# 1. 路径与工具函数
# ==========================================
def find_dataset_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    candidate_names = [
        "ieee33_static_vvo_24h_dataset.pt"
    ]

    candidate_dirs = [
        script_dir,
        os.path.join(script_dir, "data"),
        os.path.join(project_root, "data"),
        os.getcwd(),
        os.path.join(os.getcwd(), "data"),
    ]

    for folder in candidate_dirs:
        for name in candidate_names:
            path = os.path.join(folder, name)
            if os.path.exists(path):
                return path

    return None


def ensure_output_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "dataset_figures")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_key(dataset, key, default=None):
    return dataset[key] if key in dataset else default


# ==========================================
# 2. 拓扑布局
# ==========================================
def build_straight_radial_layout(G, root=0, x_spacing=1.45, y_spacing=0.90):
    if root not in G.nodes:
        raise ValueError(f"根节点 {root} 不在图中。")

    parent = {root: None}
    children = defaultdict(list)
    q = deque([root])
    visited = {root}

    while q:
        u = q.popleft()
        for v in sorted(G.neighbors(u)):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                children[u].append(v)
                q.append(v)

    unvisited = [n for n in sorted(G.nodes()) if n not in visited]
    subtree_leaves = {}

    def count_leaves(u):
        if len(children.get(u, [])) == 0:
            return 1
        return sum(count_leaves(v) for v in children[u])

    def fill_leaves(u):
        subtree_leaves[u] = count_leaves(u)
        for v in children.get(u, []):
            fill_leaves(v)

    fill_leaves(root)

    pos = {}

    def assign_pos(u, x, y_center):
        pos[u] = np.array([x, y_center], dtype=float)
        childs = children.get(u, [])
        if not childs:
            return

        total_height = sum(subtree_leaves[v] for v in childs) * y_spacing
        current_y = y_center + total_height / 2.0

        for v in childs:
            block_h = subtree_leaves[v] * y_spacing
            child_y = current_y - block_h / 2.0
            assign_pos(v, x + x_spacing, child_y)
            current_y -= block_h

    assign_pos(root, 0.0, 0.0)

    if unvisited:
        max_x = max(p[0] for p in pos.values()) if pos else 0.0
        start_y = -len(unvisited) * y_spacing / 2.0
        for i, n in enumerate(unvisited):
            pos[n] = np.array([max_x + 2 * x_spacing, start_y + i * y_spacing], dtype=float)

    return pos


# ==========================================
# 3. 数据读取与统计
# ==========================================
def load_dataset():
    data_path = find_dataset_path()
    if data_path is None:
        print("找不到数据集文件，请检查 data 目录或当前脚本目录。")
        return None, None

    print(f"正在加载静态拓扑数据集: {data_path}")
    dataset = torch.load(data_path, weights_only=False)

    required_keys = ["X", "Y_V", "Y_I", "edge_list"]
    for key in required_keys:
        if key not in dataset:
            print(f"数据集缺少关键字段：{key}")
            print(f"当前可用字段：{list(dataset.keys())}")
            return None, None

    return dataset, data_path


def compute_main_stats(dataset):
    Y_V = to_numpy(dataset["Y_V"])
    Y_I = to_numpy(dataset["Y_I"])
    X = to_numpy(dataset["X"])

    v_min = Y_V.min(axis=1)
    v_max = Y_V.max(axis=1)
    i_max = Y_I.max(axis=1)
    v_char = np.where((v_max - 1.0) >= (1.0 - v_min), v_max, v_min)

    stats = {
        "X": X,
        "Y_V": Y_V,
        "Y_I": Y_I,
        "v_min": v_min,
        "v_max": v_max,
        "i_max": i_max,
        "v_char": v_char,
        "num_samples": Y_V.shape[0],
        "num_nodes": Y_V.shape[1],
        "num_edges": Y_I.shape[1],
    }
    return stats


def print_text_stats(dataset, stats):
    Y_V = stats["Y_V"]
    Y_I = stats["Y_I"]
    v_min = stats["v_min"]
    v_max = stats["v_max"]
    i_max = stats["i_max"]
    v_char = stats["v_char"]

    print("\n================ 数据集基本信息 ================")
    print(f"样本数: {stats['num_samples']}")
    print(f"节点数: {stats['num_nodes']}")
    print(f"支路数: {stats['num_edges']}")
    print(f"X 形状: {stats['X'].shape}")
    print(f"Y_V 形状: {Y_V.shape}")
    print(f"Y_I 形状: {Y_I.shape}")

    print("\n================ 电压统计 ================")
    print(f"全节点电压范围: [{Y_V.min():.5f}, {Y_V.max():.5f}]")
    print(f"样本最小电压范围: [{v_min.min():.5f}, {v_min.max():.5f}]")
    print(f"样本最大电压范围: [{v_max.min():.5f}, {v_max.max():.5f}]")
    print(f"v_char 范围: [{v_char.min():.5f}, {v_char.max():.5f}]")
    print(f"低于 0.95 的节点电压比例: {(Y_V < 0.95).mean() * 100:.2f}%")
    print(f"高于 1.05 的节点电压比例: {(Y_V > 1.05).mean() * 100:.2f}%")

    print("\n================ 支路裕度统计 ================")
    print(f"全支路裕度范围: [{Y_I.min():.5f}, {Y_I.max():.5f}]")
    print(f"样本最大支路裕度范围: [{i_max.min():.5f}, {i_max.max():.5f}]")
    print(f"i_max > 0 样本比例: {(i_max > 0).mean() * 100:.2f}%")
    print(f"i_max <= 0 样本比例: {(i_max <= 0).mean() * 100:.2f}%")

    if "balance_info" in dataset:
        print("\n================ 均衡信息 ================")
        print(dataset["balance_info"])

    if "base_config" in dataset:
        print("\n================ 基础配置 ================")
        print(dataset["base_config"])

    print("================================================\n")


# ==========================================
# 4. 图1：核心物理量分布
# ==========================================
def plot_overview_distribution(stats, output_dir):
    Y_V = stats["Y_V"]
    Y_I = stats["Y_I"]
    v_min = stats["v_min"]
    v_max = stats["v_max"]
    i_max = stats["i_max"]
    v_char = stats["v_char"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    v_flat = Y_V[:, 1:].reshape(-1)
    axes[0, 0].hist(v_flat, bins=70, color="#8ecae6", edgecolor="black", alpha=0.85)
    for x, ls in [(0.90, ":"), (0.95, "--"), (1.00, "-"), (1.05, "--"), (1.10, ":")]:
        axes[0, 0].axvline(x, color="black", linestyle=ls, linewidth=1.1)
    axes[0, 0].set_title("节点电压幅值分布（剔除平衡节点）")
    axes[0, 0].set_xlabel("Voltage / p.u.")
    axes[0, 0].set_ylabel("频数")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[0, 1].hist(v_min, bins=50, alpha=0.70, label="样本最小电压", color="#90be6d", edgecolor="black")
    axes[0, 1].hist(v_max, bins=50, alpha=0.60, label="样本最大电压", color="#f9c74f", edgecolor="black")
    for x, ls in [(0.95, "--"), (1.00, "-"), (1.05, "--")]:
        axes[0, 1].axvline(x, color="black", linestyle=ls, linewidth=1.1)
    axes[0, 1].set_title("样本级最小/最大电压分布")
    axes[0, 1].set_xlabel("Voltage / p.u.")
    axes[0, 1].set_ylabel("频数")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.35)

    i_flat = Y_I.reshape(-1)
    axes[1, 0].hist(i_flat, bins=80, color="#f28482", edgecolor="black", alpha=0.80)
    axes[1, 0].axvline(0.0, color="black", linestyle="--", linewidth=1.4)
    axes[1, 0].set_title("全支路电流安全裕度分布")
    axes[1, 0].set_xlabel("Current margin / p.u.  （>0 表示越限）")
    axes[1, 0].set_ylabel("频数")
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1, 1].hist(i_max, bins=70, color="#ffb703", edgecolor="black", alpha=0.85)
    axes[1, 1].axvline(0.0, color="black", linestyle="--", linewidth=1.4)
    axes[1, 1].set_title("样本最大支路裕度分布")
    axes[1, 1].set_xlabel("Max current margin / p.u.")
    axes[1, 1].set_ylabel("频数")
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("静态拓扑无功优化数据集：电压与支路裕度分布", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(output_dir, "01_static_dataset_overview_distribution.png")
    plt.savefig(save_path, dpi=300)
    print(f"已保存: {save_path}")


# ==========================================
# 5. 图2：均衡效果展示
# ==========================================
def plot_balance_analysis(stats, dataset, output_dir):
    v_char = stats["v_char"]
    i_max = stats["i_max"]

    v_bins = np.linspace(0.90, 1.10, 21)
    v_idx = np.clip(np.digitize(np.clip(v_char, 0.90, 1.10), v_bins) - 1, 0, 19)
    safe = i_max <= 0
    violate = i_max > 0

    safe_count = np.array([(safe & (v_idx == k)).sum() for k in range(20)])
    violate_count = np.array([(violate & (v_idx == k)).sum() for k in range(20)])
    bin_centers = 0.5 * (v_bins[:-1] + v_bins[1:])
    width = (v_bins[1] - v_bins[0]) * 0.42

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    axes[0, 0].bar(bin_centers - width / 2, safe_count, width=width, label="i_max ≤ 0", color="#90be6d", edgecolor="black")
    axes[0, 0].bar(bin_centers + width / 2, violate_count, width=width, label="i_max > 0", color="#f94144", edgecolor="black")
    axes[0, 0].set_title("v_char 分箱下的越限/非越限样本数量")
    axes[0, 0].set_xlabel("v_char = 样本最偏离 1.0 的电压")
    axes[0, 0].set_ylabel("样本数")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[0, 1].scatter(v_char[safe], i_max[safe], s=8, alpha=0.35, label="i_max ≤ 0", color="#277da1")
    axes[0, 1].scatter(v_char[violate], i_max[violate], s=8, alpha=0.35, label="i_max > 0", color="#f94144")
    axes[0, 1].axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[0, 1].axvline(0.95, color="black", linestyle=":", linewidth=1.0)
    axes[0, 1].axvline(1.05, color="black", linestyle=":", linewidth=1.0)
    axes[0, 1].set_title("电压边界特征与最大支路裕度关系")
    axes[0, 1].set_xlabel("v_char")
    axes[0, 1].set_ylabel("i_max")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(linestyle="--", alpha=0.30)

    if "hour" in dataset:
        hour = to_numpy(dataset["hour"]).astype(int)
        hour_count = np.array([(hour == h).sum() for h in range(24)])
        axes[1, 0].bar(np.arange(1, 25), hour_count, color="#8ecae6", edgecolor="black")
        axes[1, 0].set_xticks(np.arange(1, 25))
        axes[1, 0].set_title("24小时样本数量分布")
        axes[1, 0].set_xlabel("Hour")
        axes[1, 0].set_ylabel("样本数")
        axes[1, 0].grid(axis="y", linestyle="--", alpha=0.35)
    else:
        axes[1, 0].axis("off")
        axes[1, 0].text(0.5, 0.5, "数据集中没有 hour 字段", ha="center", va="center", fontsize=14)

    if "mode" in dataset:
        mode = to_numpy(dataset["mode"]).astype(int)
        mode_names = {
            0: "随机",
            1: "高压倾向",
            2: "低压倾向",
            3: "中间区域",
            4: "设备边界",
        }
        cnt = Counter(mode.tolist())
        modes = sorted(cnt.keys())
        axes[1, 1].bar([mode_names.get(m, str(m)) for m in modes], [cnt[m] for m in modes], color="#f9c74f", edgecolor="black")
        axes[1, 1].set_title("采样模式分布")
        axes[1, 1].set_xlabel("采样模式")
        axes[1, 1].set_ylabel("样本数")
        axes[1, 1].grid(axis="y", linestyle="--", alpha=0.35)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "数据集中没有 mode 字段", ha="center", va="center", fontsize=14)

    fig.suptitle("数据均衡效果：电压区间、支路越限与24小时覆盖", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(output_dir, "02_static_dataset_balance_analysis.png")
    plt.savefig(save_path, dpi=300)
    print(f"已保存: {save_path}")


# ==========================================
# 6. 图3：输入净注入与设备出力分布
# ==========================================
def plot_input_and_device_distribution(stats, dataset, output_dir):
    X = stats["X"]
    P_net = X[:, :, 0]
    Q_net = X[:, :, 1]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    axes[0, 0].hist(P_net.reshape(-1), bins=80, color="#8ecae6", edgecolor="black", alpha=0.85)
    axes[0, 0].axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[0, 0].set_title("全节点净注入有功 P_net 分布")
    axes[0, 0].set_xlabel("P_net / MW")
    axes[0, 0].set_ylabel("频数")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[0, 1].hist(Q_net.reshape(-1), bins=80, color="#f28482", edgecolor="black", alpha=0.85)
    axes[0, 1].axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[0, 1].set_title("全节点净注入无功 Q_net 分布")
    axes[0, 1].set_xlabel("Q_net / MVar")
    axes[0, 1].set_ylabel("频数")
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.35)

    axes[0, 2].scatter(P_net.reshape(-1), Q_net.reshape(-1), s=5, alpha=0.20, color="#577590")
    axes[0, 2].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 2].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 2].set_title("节点净注入 P-Q 分布")
    axes[0, 2].set_xlabel("P_net / MW")
    axes[0, 2].set_ylabel("Q_net / MVar")
    axes[0, 2].grid(linestyle="--", alpha=0.30)

    if "pv_p" in dataset and "pv_q" in dataset:
        pv_p = to_numpy(dataset["pv_p"])
        pv_q = to_numpy(dataset["pv_q"])
        axes[1, 0].scatter(pv_p.reshape(-1), pv_q.reshape(-1), s=8, alpha=0.30, color="#219ebc")
        axes[1, 0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axes[1, 0].set_title("PV 有功-无功采样分布")
        axes[1, 0].set_xlabel("P_PV / MW")
        axes[1, 0].set_ylabel("Q_PV / MVar")
        axes[1, 0].grid(linestyle="--", alpha=0.30)
    else:
        axes[1, 0].axis("off")
        axes[1, 0].text(0.5, 0.5, "无 pv_p / pv_q 字段", ha="center", va="center", fontsize=13)

    if "ess_q" in dataset:
        ess_q = to_numpy(dataset["ess_q"])
        axes[1, 1].hist(ess_q.reshape(-1), bins=60, color="#90be6d", edgecolor="black", alpha=0.85)
        axes[1, 1].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        axes[1, 1].set_title("ESS 无功出力分布")
        axes[1, 1].set_xlabel("Q_ESS / MVar")
        axes[1, 1].set_ylabel("频数")
        axes[1, 1].grid(axis="y", linestyle="--", alpha=0.35)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "无 ess_q 字段", ha="center", va="center", fontsize=13)

    if "qdev_q" in dataset:
        qdev_q = to_numpy(dataset["qdev_q"])
        names = dataset.get("q_device_names", [f"QDev{i+1}" for i in range(qdev_q.shape[1])])
        axes[1, 2].boxplot([qdev_q[:, i] for i in range(qdev_q.shape[1])], labels=names, showfliers=False)
        axes[1, 2].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axes[1, 2].set_title("SVG/SVC/SC 无功出力分布")
        axes[1, 2].set_xlabel("设备")
        axes[1, 2].set_ylabel("Q / MVar")
        axes[1, 2].grid(axis="y", linestyle="--", alpha=0.35)
    else:
        axes[1, 2].axis("off")
        axes[1, 2].text(0.5, 0.5, "无 qdev_q 字段", ha="center", va="center", fontsize=13)

    fig.suptitle("模型输入与无功设备采样空间分布", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(output_dir, "03_static_dataset_input_device_distribution.png")
    plt.savefig(save_path, dpi=300)
    print(f"已保存: {save_path}")


# ==========================================
# 7. 图4：节点/支路维度的难点分布
# ==========================================
def plot_node_edge_risk_distribution(stats, dataset, output_dir):
    Y_V = stats["Y_V"]
    Y_I = stats["Y_I"]
    edge_list = dataset["edge_list"]

    node_v_mean = Y_V.mean(axis=0)
    node_v_std = Y_V.std(axis=0)
    node_low_rate = (Y_V < 0.95).mean(axis=0) * 100
    node_high_rate = (Y_V > 1.05).mean(axis=0) * 100

    edge_i_mean = Y_I.mean(axis=0)
    edge_i_std = Y_I.std(axis=0)
    edge_over_rate = (Y_I > 0).mean(axis=0) * 100

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    node_idx = np.arange(1, Y_V.shape[1] + 1)
    axes[0, 0].plot(node_idx, node_v_mean, marker="o", linewidth=1.2, label="均值")
    axes[0, 0].fill_between(node_idx, node_v_mean - node_v_std, node_v_mean + node_v_std, alpha=0.25, label="±1σ")
    axes[0, 0].axhline(0.95, color="black", linestyle="--", linewidth=1.0)
    axes[0, 0].axhline(1.05, color="black", linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("各节点电压均值与波动")
    axes[0, 0].set_xlabel("节点编号")
    axes[0, 0].set_ylabel("Voltage / p.u.")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(linestyle="--", alpha=0.30)

    axes[0, 1].bar(node_idx - 0.2, node_low_rate, width=0.4, label="V < 0.95", color="#277da1", edgecolor="black")
    axes[0, 1].bar(node_idx + 0.2, node_high_rate, width=0.4, label="V > 1.05", color="#f94144", edgecolor="black")
    axes[0, 1].set_title("各节点电压越限出现比例")
    axes[0, 1].set_xlabel("节点编号")
    axes[0, 1].set_ylabel("比例 / %")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.30)

    edge_idx = np.arange(1, Y_I.shape[1] + 1)
    axes[1, 0].plot(edge_idx, edge_i_mean, marker="o", linewidth=1.2, label="均值", color="#f3722c")
    axes[1, 0].fill_between(edge_idx, edge_i_mean - edge_i_std, edge_i_mean + edge_i_std, alpha=0.25, label="±1σ", color="#f3722c")
    axes[1, 0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("各支路电流裕度均值与波动")
    axes[1, 0].set_xlabel("支路编号")
    axes[1, 0].set_ylabel("Current margin / p.u.")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(linestyle="--", alpha=0.30)

    axes[1, 1].bar(edge_idx, edge_over_rate, color="#f94144", edgecolor="black", alpha=0.85)
    axes[1, 1].set_title("各支路越限出现比例")
    axes[1, 1].set_xlabel("支路编号")
    axes[1, 1].set_ylabel("i_margin > 0 比例 / %")
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.30)

    fig.suptitle("节点与支路维度的安全边界分布", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(output_dir, "04_static_dataset_node_edge_risk.png")
    plt.savefig(save_path, dpi=300)
    print(f"已保存: {save_path}")

    top_edges = np.argsort(-edge_over_rate)[:8]
    print("越限比例最高的支路：")
    for e in top_edges:
        fr, to = edge_list[e]
        print(f"  支路 {e + 1:02d}: {int(fr) + 1}->{int(to) + 1}, 越限比例 {edge_over_rate[e]:.2f}%")


# ==========================================
# 8. 图5：静态拓扑与资源位置
# ==========================================
def plot_topology_with_devices(dataset, output_dir):
    edge_list = dataset["edge_list"]

    pv_nodes = to_numpy(get_key(dataset, "pv_nodes", [])).astype(int).tolist()
    ess_nodes = to_numpy(get_key(dataset, "ess_nodes", [])).astype(int).tolist()
    q_device_nodes = to_numpy(get_key(dataset, "q_device_nodes", [])).astype(int).tolist()
    q_device_names = get_key(dataset, "q_device_names", [f"QDev{i+1}" for i in range(len(q_device_nodes))])

    G = nx.Graph()
    G.add_nodes_from(range(33))
    active_edges = [tuple(map(int, edge[:2])) for edge in edge_list]
    G.add_edges_from(active_edges)

    pos = build_straight_radial_layout(G, root=0, x_spacing=1.45, y_spacing=0.90)

    fig, ax = plt.subplots(figsize=(12, 8))

    normal_nodes = [n for n in G.nodes() if n != 0]
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=active_edges, edge_color="black", width=1.4)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=normal_nodes, node_color="lightgray",
                           edgecolors="black", node_size=300, node_shape="o", label="普通节点")
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[0], node_color="#f94144",
                           edgecolors="black", node_size=550, node_shape="s", label="平衡节点")

    if pv_nodes:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=pv_nodes, node_color="#ffb703",
                               edgecolors="black", node_size=520, node_shape="^", label="PV")
    if ess_nodes:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=ess_nodes, node_color="#90be6d",
                               edgecolors="black", node_size=520, node_shape="D", label="ESS")
    if q_device_nodes:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=q_device_nodes, node_color="#219ebc",
                               edgecolors="black", node_size=520, node_shape="h", label="SVG/SVC/SC")

    labels = {n: str(n + 1) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=9, font_family="sans-serif")

    q_label_lines = []
    for name, node in zip(q_device_names, q_device_nodes):
        q_label_lines.append(f"{name}@{int(node) + 1}")
    if q_label_lines:
        ax.text(0.01, 0.02, "无功设备: " + ", ".join(q_label_lines),
                transform=ax.transAxes, fontsize=10, va="bottom", ha="left")

    ax.set_title(f"IEEE 33 静态径向拓扑与无功资源位置（{len(active_edges)} 条支路）", fontsize=16)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper left", frameon=False)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "05_static_topology_with_devices.png")
    plt.savefig(save_path, dpi=300)
    print(f"已保存: {save_path}")


# ==========================================
# 9. 主程序
# ==========================================
def main():
    dataset, data_path = load_dataset()
    if dataset is None:
        return

    output_dir = ensure_output_dir()
    stats = compute_main_stats(dataset)

    print_text_stats(dataset, stats)

    print("正在绘制 01 核心物理量分布图...")
    plot_overview_distribution(stats, output_dir)

    print("正在绘制 02 均衡效果分析图...")
    plot_balance_analysis(stats, dataset, output_dir)

    print("正在绘制 03 输入与设备出力分布图...")
    plot_input_and_device_distribution(stats, dataset, output_dir)

    print("正在绘制 04 节点/支路风险分布图...")
    plot_node_edge_risk_distribution(stats, dataset, output_dir)

    print("正在绘制 05 拓扑与设备位置图...")
    plot_topology_with_devices(dataset, output_dir)

    print("\n绘图完成。")
    print(f"数据集路径: {data_path}")
    print(f"图像输出目录: {output_dir}")
    plt.show()


if __name__ == "__main__":
    main()