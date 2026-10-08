import os

# ===========================
# 解决 OpenMP 冲突 + 线程控制
# ===========================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ===========================
# IEEE大论文绘图风格标准
# ===========================
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.linewidth'] = 0.8
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8
rcParams['font.size'] = 11

# ===========================
# 1. 数据加载 (适配 ST-MGCN 静态拓扑)
# ===========================
# 请确保路径与你生成数据的保存路径一致
data_path = r"E:\CollegeApps\PythonProject\ST-MGCN\data\st_mgcn_ieee33.pt"
data = torch.load(data_path, weights_only=False)

X = data['X'].numpy()          # Shape: (Batch, 33, 2)
Y_V = data['Y_V'].numpy()      # Shape: (Batch, 33)
Y_I = data['Y_I'].numpy()      # Shape: (Batch, 32)
edge_list = data['edge_list']  # 静态物理拓扑 (List)

print("✅ ST-MGCN 数据加载完成：")
print(f"输入特征 X: {X.shape}")
print(f"电压标签 Y_V: {Y_V.shape}")
print(f"裕度标签 Y_I: {Y_I.shape}")
print(f"固定支路数: {len(edge_list)}")

# ===========================
# 2. 剔除平衡节点（bus 0）
# 说明：
# 平衡节点电压恒为 1.0 p.u.，且其注入功率为平衡系统的松弛变量，
# 将其纳入直方图会导致极端的长尾或脉冲干扰，因此在统计节点分布时必须剔除。
# ===========================
non_slack_mask = np.arange(X.shape[1]) != 0

# 数据展开成一维，准备画直方图
P_net = X[:, non_slack_mask, 0].reshape(-1)
Q_net = X[:, non_slack_mask, 1].reshape(-1)
V_all = Y_V[:, non_slack_mask].reshape(-1)

# 线路裕度是基于 32 条支路的，直接展开
I_margin = Y_I.reshape(-1)

# ===========================
# 3. 绘图 (2x2 极简紧凑布局)
# ===========================
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# ---------------------------
# (a) P 分布 (净有功注入)
# ---------------------------
ax = axes[0, 0]
ax.hist(P_net, bins=80, alpha=0.75, color='#1f77b4', edgecolor='black', linewidth=0.4)
ax.set_xlabel("Net Active Power Injection $P$ (MW)")
ax.set_ylabel("Frequency")
ax.grid(True, alpha=0.15, linewidth=0.5, linestyle='--')
ax.text(0.5, -0.22, "(a) Active Power Distribution", transform=ax.transAxes, ha='center')

# ---------------------------
# (b) Q 分布 (净无功注入)
# ---------------------------
ax = axes[0, 1]
ax.hist(Q_net, bins=80, alpha=0.75, color='#ff7f0e', edgecolor='black', linewidth=0.4)
ax.set_xlabel("Net Reactive Power Injection $Q$ (MVar)")
ax.set_ylabel("Frequency")
ax.grid(True, alpha=0.15, linewidth=0.5, linestyle='--')
ax.text(0.5, -0.22, "(b) Reactive Power Distribution", transform=ax.transAxes, ha='center')

# ---------------------------
# (c) 电压分布 (不含平衡节点)
# ---------------------------
ax = axes[1, 0]
ax.hist(V_all, bins=80, alpha=0.75, color='#2ca02c', edgecolor='black', linewidth=0.4)
ax.set_xlabel("Node Voltage Amplitude $V$ (p.u.)")
ax.set_ylabel("Frequency")
ax.grid(True, alpha=0.15, linewidth=0.5, linestyle='--')
# 标出安全边界参考线
ax.axvline(x=0.90, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
ax.axvline(x=1.10, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
ax.text(0.5, -0.22, "(c) Voltage Amplitude Distribution", transform=ax.transAxes, ha='center')

# ---------------------------
# (d) 支路安全裕度分布
# ---------------------------
ax = axes[1, 1]
ax.hist(I_margin, bins=80, alpha=0.75, color='#d62728', edgecolor='black', linewidth=0.4)
ax.set_xlabel("Branch Current Margin $I_{\mathrm{margin}}$")
ax.set_ylabel("Frequency")
ax.grid(True, alpha=0.15, linewidth=0.5, linestyle='--')
# 标出热稳定极限红线
ax.axvline(x=0.0, color='black', linestyle='--', linewidth=1.5)
ax.text(0.5, -0.22, "(d) Branch Current Margin Distribution", transform=ax.transAxes, ha='center')

# 调整子图间距，防止标签重叠
plt.subplots_adjust(hspace=0.4, wspace=0.25)

# ===========================
# 4. 保存高质量图片
# ===========================
save_pdf = "st_mgcn_dataset_distributions.pdf"
save_tiff = "st_mgcn_dataset_distributions.tiff"

plt.savefig(save_pdf, dpi=300, bbox_inches='tight')
plt.savefig(
    save_tiff,
    dpi=300,
    bbox_inches='tight',
    format='tiff',
    pil_kwargs={"compression": "tiff_lzw"}
)

print(f"✅ 高质量图像已保存: \n  -> {save_pdf}\n  -> {save_tiff}")
plt.show()