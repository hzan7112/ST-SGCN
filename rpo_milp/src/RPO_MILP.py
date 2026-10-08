import os

# ==========================================
# 0. 环境变量（必须放在第三方库导入前）
# ==========================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from rpo_milp.src.gurobi_milp_converter import GurobiMILPConverter


# ==========================================
# 1. 电网基础配置
# ==========================================
def prepare_base_profiles():
    pload_standard = np.array([
        0, 100, 90, 120, 60, 60, 200, 200, 60, 60,
        45, 60, 60, 120, 60, 60, 60, 90, 90, 90,
        90, 90, 90, 420, 420, 60, 60, 60, 420, 400,
        450, 410, 60
    ], dtype=float)

    qload_standard = np.array([
        0, 60, 40, 80, 30, 20, 100, 100, 20, 20,
        30, 35, 35, 80, 10, 20, 20, 40, 40, 40,
        40, 40, 50, 200, 200, 25, 25, 20, 70, 600,
        70, 100, 40
    ], dtype=float)

    Pload_mw = pload_standard / 1000.0
    Qload_mvar = qload_standard / 1000.0

    pv_nodes = [7, 14, 22, 29]
    S_rated_mva = np.array([1.5, 2.0, 3.0, 3.5], dtype=float)

    return Pload_mw, Qload_mvar, pv_nodes, S_rated_mva


def get_standard_radial_topology():
    """
    37 条支路中，前 32 条为标准辐射网闭合，
    后 5 条联络线断开。
    """
    topo_mask = np.zeros(37, dtype=bool)
    topo_mask[:32] = True
    return topo_mask


def get_default_operating_point():
    """
    默认运行点：
    - 负荷取标准值
    - 光伏取中午附近可用有功
    """
    P_load, Q_load, pv_nodes, S_rated = prepare_base_profiles()
    P_pv_available = 0.8 * S_rated
    topo_mask = get_standard_radial_topology()
    return P_load, Q_load, P_pv_available, topo_mask


# ==========================================
# 2. 数据检查
# ==========================================
def validate_inputs(P_load, Q_load, P_pv_available, topo_mask, pv_nodes, S_rated):
    P_load = np.asarray(P_load, dtype=float).reshape(-1)
    Q_load = np.asarray(Q_load, dtype=float).reshape(-1)
    P_pv_available = np.asarray(P_pv_available, dtype=float).reshape(-1)
    topo_mask = np.asarray(topo_mask, dtype=bool).reshape(-1)

    if len(P_load) != 33:
        raise ValueError(f"P_load 长度必须为 33，当前为 {len(P_load)}")
    if len(Q_load) != 33:
        raise ValueError(f"Q_load 长度必须为 33，当前为 {len(Q_load)}")
    if len(P_pv_available) != len(pv_nodes):
        raise ValueError(f"P_pv_available 长度必须为 {len(pv_nodes)}，当前为 {len(P_pv_available)}")
    if len(topo_mask) != 37:
        raise ValueError(f"topo_mask 长度必须为 37，当前为 {len(topo_mask)}")

    for idx, (bus, p_avail, s_max) in enumerate(zip(pv_nodes, P_pv_available, S_rated)):
        if p_avail < 0:
            raise ValueError(f"光伏节点 {bus} 的可用有功不能为负，当前为 {p_avail}")
        if p_avail > s_max + 1e-9:
            raise ValueError(
                f"光伏节点 {bus} 的可用有功 {p_avail:.4f} MW 超过额定容量 {s_max:.4f} MVA，"
                f"会导致无功容量约束不可行。"
            )

    return P_load, Q_load, P_pv_available, topo_mask


# ==========================================
# 3. 无功优化主函数
# ==========================================
def run_reactive_power_optimization(
    P_load,
    Q_load,
    P_pv_available,
    topo_mask,
    model_path="models/best_pignn_deep_pure.pth",
    stats_path="models/norm_stats.pt",
    hidden_dim=64,
    num_layers=18
):
    print("开始执行无功优化...")

    pv_nodes = [7, 14, 22, 29]
    S_rated = np.array([1.5, 2.0, 3.0, 3.5], dtype=float)

    P_load, Q_load, P_pv_available, topo_mask = validate_inputs(
        P_load, Q_load, P_pv_available, topo_mask,
        pv_nodes=pv_nodes,
        S_rated=S_rated
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件：{model_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"找不到归一化参数文件：{stats_path}")

    # 1. 初始化模型
    m = gp.Model("VQC_Optimization")
    m.Params.TimeLimit = 30000000
    m.Params.MIPGap = 0.01
    m.Params.OutputFlag = 1

    converter = GurobiMILPConverter(
        model_path=model_path,
        stats_path=stats_path,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )

    # 2. 定义无功决策变量
    Q_pv = m.addVars(len(pv_nodes), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Q_pv")

    # 3. 光伏容量约束
    for idx, (bus, s_max, p_avail) in enumerate(zip(pv_nodes, S_rated, P_pv_available)):
        rhs = float(s_max ** 2 - p_avail ** 2)
        if rhs < -1e-10:
            raise ValueError(
                f"节点 {bus} 的可用有功 {p_avail:.4f} 超过额定容量 {s_max:.4f}，"
                f"导致 sqrt(S^2-P^2) 无定义。"
            )
        rhs = max(rhs, 0.0)

        m.addQConstr(
            Q_pv[idx] * Q_pv[idx] <= rhs,
            name=f"PV_Cap_{bus}"
        )

    # 4. 构建节点注入功率矩阵 X_vars
    X_vars = [[None, None] for _ in range(33)]
    pv_to_idx = {bus: idx for idx, bus in enumerate(pv_nodes)}

    for i in range(33):
        p_inj = -float(P_load[i])
        q_inj = -float(Q_load[i])

        if i in pv_to_idx:
            idx = pv_to_idx[i]
            p_inj += float(P_pv_available[idx])

            X_vars[i][0] = gp.LinExpr(p_inj)
            X_vars[i][1] = gp.LinExpr(q_inj) + Q_pv[idx]
        else:
            X_vars[i][0] = gp.LinExpr(p_inj)
            X_vars[i][1] = gp.LinExpr(q_inj)

    # 5. 注入 GNN 代理模型约束
    print("正在向 Gurobi 注入 GNN 代理模型约束，请稍候...")
    V_pred, I_margin_pred = converter.embed_gnn_constraints(m, X_vars, topo_mask)

    # 6. 物理安全边界约束
    for i in range(1, 33):
        m.addConstr(V_pred[i] >= 0.95, name=f"V_min_{i}")
        m.addConstr(V_pred[i] <= 1.05, name=f"V_max_{i}")

    epsilon_offset = 0.05
    for e_idx in range(37):
        if topo_mask[e_idx]:
            m.addConstr(
                I_margin_pred[e_idx] <= epsilon_offset,
                name=f"I_margin_safe_{e_idx}"
            )

    # 7. 目标函数：最小化电压偏差平方和
    obj_expr = gp.QuadExpr()
    for i in range(1, 33):
        obj_expr += (V_pred[i] - 1.0) * (V_pred[i] - 1.0)

    m.setObjective(obj_expr, GRB.MINIMIZE)

    # 8. 求解
    print("开始求解...")
    m.optimize()

    result = {
        "status": int(m.status),
        "objective": None,
        "Q_pv_opt": None,
        "V_pred_opt": None,
        "I_margin_pred_opt": None,
    }

    if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and m.SolCount > 0:
        q_solution = np.array([Q_pv[idx].X for idx in range(len(pv_nodes))], dtype=float)
        v_solution = np.array([V_pred[i].X for i in range(33)], dtype=float)
        i_margin_solution = np.array([I_margin_pred[e].X for e in range(37)], dtype=float)

        result["objective"] = float(m.ObjVal)
        result["Q_pv_opt"] = q_solution
        result["V_pred_opt"] = v_solution
        result["I_margin_pred_opt"] = i_margin_solution

        print("\n最优/当前可行无功调度方案：")
        for idx, bus in enumerate(pv_nodes):
            q_cap = np.sqrt(max(S_rated[idx] ** 2 - P_pv_available[idx] ** 2, 0.0))
            print(
                f"节点 {bus} | P_avail = {P_pv_available[idx]:.4f} MW | "
                f"Q* = {q_solution[idx]:.6f} MVar | "
                f"理论|Q|max = {q_cap:.6f}"
            )

        print(f"\n目标函数值: {m.ObjVal:.8f}")
        print(f"预测电压最小值: {v_solution[1:].min():.6f} p.u.")
        print(f"预测电压最大值: {v_solution[1:].max():.6f} p.u.")

        closed_margin = i_margin_solution[topo_mask]
        if len(closed_margin) > 0:
            print(f"闭合支路预测电流裕度最大值: {closed_margin.max():.6f}")

        if m.status == GRB.OPTIMAL:
            print("求解状态：OPTIMAL")
        elif m.status == GRB.TIME_LIMIT:
            print("求解状态：TIME_LIMIT（已返回当前最好可行解）")
        else:
            print("求解状态：SUBOPTIMAL（已返回当前可行解）")
    else:
        print(f"未找到可行解，模型状态码: {m.status}")

    return result


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    P_load_arr, Q_load_arr, P_pv_arr, topo_mask_arr = get_default_operating_point()

    print("当前运行配置如下：")
    print(f"P_load_arr shape = {P_load_arr.shape}")
    print(f"Q_load_arr shape = {Q_load_arr.shape}")
    print(f"P_pv_arr = {P_pv_arr}")
    print(f"topo_mask_arr 闭合支路数 = {np.sum(topo_mask_arr)} / 37")

    run_reactive_power_optimization(
        P_load=P_load_arr,
        Q_load=Q_load_arr,
        P_pv_available=P_pv_arr,
        topo_mask=topo_mask_arr,
        model_path="models/best_pignn_deep_pure.pth",
        stats_path="models/norm_stats.pt",
        hidden_dim=64,
        num_layers=18
    )