"""Single-step reactive power optimization using the model/mlp MILP surrogate.

This script follows the optimization setting in ``rpo_milp/exp17_milp``:
operating-point selection, PV/ESS/reactive-device capability constraints,
conditional ``Q_net`` bounds, optional trust regions, and slack diagnostics are
kept aligned with Exp17.

The surrogate is the MLP produced by ``model/mlp/train_mlp.py``. It consumes
raw IEEE-33 net injections ``[P_net, Q_net]`` and predicts:

    Vdev_total, Vworst, WorstI, Ploss_total

The default surrogate safety constraints are ``Vworst <= 0`` and
``WorstI <= 0``. ``Ploss_total`` is reported in results, while the RPO
objective remains Exp17-style voltage-deviation minimization by default.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rpo_milp.exp17_milp import single_step_rpo as exp17_rpo


DEFAULT_ENGINE_PATH = (
    REPO_ROOT
    / "checkpoints"
    / "mlp_exp20_four_scalars_96bin_milp_engine.pt"
)
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "ieee33_nodal_pq_correlated_raw_pool_50k.pt"


def require_gurobi():
    return exp17_rpo.require_gurobi()


def get_converter_class():
    require_gurobi()
    from rpo_milp.mlp_milp.mlp_milp_converter import MLPMILPConverter

    return MLPMILPConverter


prepare_base_profiles = exp17_rpo.prepare_base_profiles
get_standard_radial_topology = exp17_rpo.get_standard_radial_topology
get_default_operating_point = exp17_rpo.get_default_operating_point
safe_torch_load = exp17_rpo.safe_torch_load
to_numpy = exp17_rpo.to_numpy
worst_voltage_margin = exp17_rpo.worst_voltage_margin
choose_dataset_sample = exp17_rpo.choose_dataset_sample
get_dataset_operating_point = exp17_rpo.get_dataset_operating_point
parse_vector_arg = exp17_rpo.parse_vector_arg
normalize_optional_vector = exp17_rpo.normalize_optional_vector
validate_inputs = exp17_rpo.validate_inputs
build_injection_expressions = exp17_rpo.build_injection_expressions
build_dataset_injection_expressions = exp17_rpo.build_dataset_injection_expressions
add_pv_capacity_constraints = exp17_rpo.add_pv_capacity_constraints
add_ess_capacity_constraints = exp17_rpo.add_ess_capacity_constraints
add_q_device_bounds = exp17_rpo.add_q_device_bounds
build_control_labels = exp17_rpo.build_control_labels
build_control_base_vector = exp17_rpo.build_control_base_vector
control_variable_list = exp17_rpo.control_variable_list
build_control_node_vector = exp17_rpo.build_control_node_vector
build_control_bound_vectors = exp17_rpo.build_control_bound_vectors
add_conditional_qnet_bounds = exp17_rpo.add_conditional_qnet_bounds
add_control_trust_region = exp17_rpo.add_control_trust_region
add_control_deviation_penalty = exp17_rpo.add_control_deviation_penalty
status_name = exp17_rpo.status_name
print_solution = exp17_rpo.print_solution


def add_mlp_physical_output_bounds(model, outputs, *, enabled: bool) -> dict:
    """Add simple physical bounds for the direct MLP scalar heads."""

    if not enabled:
        return {}
    model.addConstr(outputs.Vdev_total >= 0.0, name="mlp_physical_Vdev_total_nonnegative")
    model.addConstr(outputs.Vworst >= -0.05, name="mlp_physical_Vworst_lower")
    model.addConstr(outputs.WorstI >= -1.0, name="mlp_physical_WorstI_lower")
    model.addConstr(outputs.Ploss_total >= 0.0, name="mlp_physical_Ploss_total_nonnegative")
    return {
        "Vdev_total_min": 0.0,
        "Vworst_min": -0.05,
        "WorstI_min": -1.0,
        "Ploss_total_min": 0.0,
    }


def add_mlp_safety_constraints(
    model,
    outputs,
    *,
    voltage_margin: float,
    current_margin: float,
    diagnose_slack: bool,
    slack_penalty: float,
    base_objective,
    minimize_sense,
):
    """Constrain learned voltage and current safety margins."""

    if diagnose_slack:
        slack_v = model.addVar(lb=0.0, name="slack_Vworst")
        slack_i = model.addVar(lb=0.0, name="slack_WorstI")
        model.addConstr(outputs.Vworst <= float(voltage_margin) + slack_v, name="safety_Vworst_with_slack")
        model.addConstr(outputs.WorstI <= float(current_margin) + slack_i, name="safety_WorstI_with_slack")
        model.setObjective(base_objective + float(slack_penalty) * (slack_v + slack_i), minimize_sense)
        return slack_v, slack_i
    model.addConstr(outputs.Vworst <= float(voltage_margin), name="safety_Vworst")
    model.addConstr(outputs.WorstI <= float(current_margin), name="safety_WorstI")
    return None, None


def build_objective_expression(
    outputs,
    objective: str,
    *,
    loss_weight: float,
    voltage_weight: float,
    loss_expr=None,
):
    if objective == "vdev":
        return outputs.Vdev_total
    if objective == "ploss":
        return outputs.Ploss_total
    if objective == "ploss_vdev":
        return outputs.Ploss_total + float(voltage_weight) * outputs.Vdev_total
    raise ValueError("objective must be one of: vdev, ploss, ploss_vdev")


def objective_value_from_components(
    output_values,
    objective: str,
    *,
    loss_weight: float,
    voltage_weight: float,
    loss_value=None,
):
    if output_values is None:
        return None
    if objective == "vdev":
        return float(output_values["Vdev_total"])
    if objective == "ploss":
        return float(output_values["Ploss_total"])
    if objective == "ploss_vdev":
        return float(output_values["Ploss_total"]) + float(voltage_weight) * float(output_values["Vdev_total"])
    raise ValueError("objective must be one of: vdev, ploss, ploss_vdev")


def select_loss_objective_expression(
    converter,
    outputs,
    *,
    loss_objective_model: str,
    base_kv: float,
    base_mva: float,
    surrogate_loss_weight: float,
):
    if loss_objective_model in {"none", "disabled"}:
        return None
    if loss_objective_model in {"surrogate", "mlp"}:
        return outputs.Ploss_total
    raise ValueError("MLP supports --loss-objective-model none or surrogate")


def add_surrogate_loss_consistency_constraints(
    model,
    converter,
    outputs,
    *,
    loss_objective_model: str,
    base_kv: float,
    base_mva: float,
    relative_tol: float,
    absolute_tol: float,
):
    return {
        "enabled": False,
        "reason": "model/mlp reports Ploss_total directly; no auxiliary loss consistency constraint is needed.",
    }


def run_single_step_rpo(
    p_load,
    q_load,
    p_pv_available,
    topo_mask,
    *,
    pv_nodes,
    s_rated,
    engine_path=DEFAULT_ENGINE_PATH,
    relu_formulation="big_m",
    big_m_scale=1.0,
    fallback_big_m=1e3,
    objective="vdev",
    loss_objective_model="none",
    surrogate_loss_weight=0.5,
    surrogate_loss_consistency_rel=0.5,
    surrogate_loss_consistency_abs=0.002,
    loss_weight=0.0,
    voltage_weight=1e-3,
    lambda_q=0.0,
    voltage_margin=0.0,
    current_margin=0.0,
    physical_output_bounds=True,
    trust_region_fraction=1.0,
    control_deviation_penalty=0.0,
    base_kv=12.66,
    base_mva=1.0,
    time_limit=300.0,
    mip_gap=0.01,
    dual_reductions=0,
    diagnose_slack=False,
    slack_penalty=1e4,
    output_flag=1,
    export_lp=None,
    iis_path=None,
    x_base=None,
    pv_q_base=None,
    ess_nodes=None,
    ess_s_rated=None,
    ess_p_base=None,
    ess_q_base=None,
    q_device_nodes=None,
    q_device_min=None,
    q_device_max=None,
    q_device_q_base=None,
    q_device_names=None,
):
    gp_mod, grb_mod = require_gurobi()
    converter_class = get_converter_class()

    p_load, q_load, p_pv_available, topo_mask, pv_nodes, s_rated = validate_inputs(
        p_load,
        q_load,
        p_pv_available,
        topo_mask,
        pv_nodes,
        s_rated,
    )
    ess_nodes = normalize_optional_vector(ess_nodes, dtype=int)
    ess_s_rated = normalize_optional_vector(ess_s_rated)
    ess_p_base = normalize_optional_vector(ess_p_base)
    ess_q_base = normalize_optional_vector(ess_q_base)
    q_device_nodes = normalize_optional_vector(q_device_nodes, dtype=int)
    q_device_min = normalize_optional_vector(q_device_min)
    q_device_max = normalize_optional_vector(q_device_max)
    q_device_q_base = normalize_optional_vector(q_device_q_base)
    q_device_names = [] if q_device_names is None else [str(x) for x in q_device_names]
    if float(lambda_q) != 0.0:
        raise ValueError("MLP RPO follows Exp17 voltage-style optimization; set lambda_q=0")

    if len(ess_nodes) and not (
        len(ess_s_rated) == len(ess_p_base) == len(ess_q_base) == len(ess_nodes)
    ):
        raise ValueError("ESS node, rating, active-power, and reactive-power arrays must have the same length")
    if len(q_device_nodes) and not (
        len(q_device_min) == len(q_device_max) == len(q_device_q_base) == len(q_device_nodes)
    ):
        raise ValueError("Q-device node, min, max, and base reactive arrays must have the same length")

    model = gp_mod.Model("MLP_SingleStep_RPO")
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = float(mip_gap)
    model.Params.OutputFlag = int(output_flag)
    model.Params.NonConvex = 2
    model.Params.DualReductions = int(dual_reductions)

    converter = converter_class(
        engine_path,
        relu_formulation=relu_formulation,
        fallback_big_m=fallback_big_m,
        big_m_scale=big_m_scale,
    )

    q_pv = model.addVars(len(pv_nodes), lb=-grb_mod.INFINITY, ub=grb_mod.INFINITY, name="Q_pv")
    q_ess = model.addVars(len(ess_nodes), lb=-grb_mod.INFINITY, ub=grb_mod.INFINITY, name="Q_ess")
    q_device = model.addVars(
        len(q_device_nodes),
        lb=-grb_mod.INFINITY,
        ub=grb_mod.INFINITY,
        name="Q_device",
    )
    q_caps = add_pv_capacity_constraints(model, q_pv, p_pv_available, pv_nodes, s_rated)
    ess_q_caps = add_ess_capacity_constraints(model, q_ess, ess_p_base, ess_nodes, ess_s_rated)
    qdev_min, qdev_max = add_q_device_bounds(
        model,
        q_device,
        q_device_nodes,
        q_device_min,
        q_device_max,
    )

    labels = build_control_labels(pv_nodes, ess_nodes, q_device_nodes, q_device_names)
    control_vars = control_variable_list(
        q_pv,
        q_ess,
        q_device,
        len(pv_nodes),
        len(ess_nodes),
        len(q_device_nodes),
    )
    control_nodes = build_control_node_vector(pv_nodes, ess_nodes, q_device_nodes)
    control_lower, control_upper = build_control_bound_vectors(
        q_caps,
        ess_q_caps,
        qdev_min,
        qdev_max,
    )
    control_base = build_control_base_vector(
        pv_q_base if x_base is not None else None,
        ess_q_base,
        q_device_q_base,
        n_pv=len(pv_nodes),
        n_ess=len(ess_nodes),
        n_qdev=len(q_device_nodes),
    )
    q_reference, trust_radius = add_control_trust_region(
        model,
        control_vars,
        control_base.copy(),
        control_lower,
        control_upper,
        labels,
        fraction=float(trust_region_fraction),
    )

    if x_base is None:
        x_vars = build_injection_expressions(
            model,
            q_pv,
            p_load,
            q_load,
            p_pv_available,
            pv_nodes,
        )
    else:
        if pv_q_base is None:
            raise ValueError("pv_q_base is required when x_base is provided")
        x_vars = build_dataset_injection_expressions(
            model,
            q_pv,
            q_ess,
            q_device,
            x_base,
            pv_q_base,
            pv_nodes,
            ess_q_base,
            ess_nodes,
            q_device_q_base,
            q_device_nodes,
        )

    conditional_qnet_bounds = add_conditional_qnet_bounds(
        model,
        x_vars,
        x_base=x_base,
        q_load=q_load,
        control_nodes=control_nodes,
        control_base=control_base,
        control_lower=control_lower,
        control_upper=control_upper,
    )

    print("Embedding model/mlp surrogate constraints...")
    outputs = converter.embed_mlp_constraints(
        model,
        x_vars,
        topo_mask=topo_mask,
        name_prefix="mlp_rpo",
    )

    output_physical_bounds = add_mlp_physical_output_bounds(
        model,
        outputs,
        enabled=bool(physical_output_bounds),
    )
    loss_objective_expr = select_loss_objective_expression(
        converter,
        outputs,
        loss_objective_model=str(loss_objective_model),
        base_kv=float(base_kv),
        base_mva=float(base_mva),
        surrogate_loss_weight=float(surrogate_loss_weight),
    )
    loss_consistency = add_surrogate_loss_consistency_constraints(
        model,
        converter,
        outputs,
        loss_objective_model=str(loss_objective_model),
        base_kv=float(base_kv),
        base_mva=float(base_mva),
        relative_tol=float(surrogate_loss_consistency_rel),
        absolute_tol=float(surrogate_loss_consistency_abs),
    )
    base_objective = build_objective_expression(
        outputs,
        objective,
        loss_weight=loss_weight,
        voltage_weight=voltage_weight,
        loss_expr=loss_objective_expr,
    )
    deviation_penalty, deviation_aux = add_control_deviation_penalty(
        model,
        control_vars,
        q_reference,
        labels,
        weight=float(control_deviation_penalty),
    )
    protected_objective = base_objective + deviation_penalty
    model.setObjective(protected_objective, grb_mod.MINIMIZE)
    slack_v, slack_i = add_mlp_safety_constraints(
        model,
        outputs,
        voltage_margin=voltage_margin,
        current_margin=current_margin,
        diagnose_slack=diagnose_slack,
        slack_penalty=slack_penalty,
        base_objective=protected_objective,
        minimize_sense=grb_mod.MINIMIZE,
    )

    if export_lp:
        export_path = Path(export_lp)
        if not export_path.is_absolute():
            export_path = REPO_ROOT / export_path
        export_path.parent.mkdir(parents=True, exist_ok=True)
        model.write(str(export_path))
        print(f"LP/MPS written before solve: {export_path}")

    print("Starting Gurobi solve...")
    model.optimize()

    result = {
        "status": int(model.status),
        "status_name": status_name(model.status),
        "objective": None,
        "objective_mode": objective,
        "loss_objective_model": str(loss_objective_model),
        "surrogate_loss_weight": float(surrogate_loss_weight),
        "surrogate_loss_consistency": loss_consistency,
        "loss_weight": float(loss_weight),
        "voltage_weight": float(voltage_weight),
        "voltage_margin": float(voltage_margin),
        "current_margin": float(current_margin),
        "Q_pv_opt": None,
        "Q_pv_caps": q_caps,
        "Q_ess_opt": None,
        "Q_ess_caps": ess_q_caps,
        "Q_device_opt": None,
        "Q_device_min": qdev_min,
        "Q_device_max": qdev_max,
        "Q_control_opt": None,
        "Q_control_labels": labels,
        "Q_control_reference": q_reference,
        "Q_control_lower": control_lower,
        "Q_control_upper": control_upper,
        "Q_control_trust_radius": trust_radius,
        "conditional_qnet_bounds": conditional_qnet_bounds,
        "trust_region_fraction": float(trust_region_fraction),
        "physical_output_bounds": output_physical_bounds,
        "lambda_q": float(lambda_q),
        "q_flow_regularization": None,
        "q_flow_penalty": None,
        "q_flow_branch_values": None,
        "control_deviation_penalty_weight": float(control_deviation_penalty),
        "control_deviation_abs": None,
        "loss_objective_value": None,
        "objective_components": None,
        "outputs": None,
        "safety_slack": None,
        "binary_count_estimate": converter.binary_count,
        "sol_count": int(model.SolCount),
    }

    if model.SolCount > 0:
        q_solution = np.array([q_pv[idx].X for idx in range(len(pv_nodes))], dtype=float)
        q_ess_solution = np.array([q_ess[idx].X for idx in range(len(ess_nodes))], dtype=float)
        q_device_solution = np.array([q_device[idx].X for idx in range(len(q_device_nodes))], dtype=float)
        q_control_solution = np.concatenate([q_solution, q_ess_solution, q_device_solution])
        output_values = {
            "Vdev_total": float(outputs.Vdev_total.X),
            "Vworst": float(outputs.Vworst.X),
            "WorstI": float(outputs.WorstI.X),
            "Ploss_total": float(outputs.Ploss_total.X),
            "V_nodes": None,
        }
        loss_objective_value = None if loss_objective_expr is None else float(outputs.Ploss_total.X)
        deviation_values = np.array([dev.X for dev in deviation_aux], dtype=float)
        guarded_objective = objective_value_from_components(
            output_values,
            objective,
            loss_weight=loss_weight,
            voltage_weight=voltage_weight,
            loss_value=loss_objective_value,
        )
        deviation_penalty_value = float(control_deviation_penalty) * float(deviation_values.sum())
        objective_components = {
            "guarded_objective": guarded_objective,
            "loss_objective": loss_objective_value,
            "surrogate_ploss": float(outputs.Ploss_total.X),
            "q_flow_regularization": None,
            "q_flow_penalty": None,
            "control_deviation_penalty": deviation_penalty_value,
            "slack_penalty": None,
        }
        slack_values = None
        if slack_v is not None and slack_i is not None:
            slack_values = {
                "Vworst": float(slack_v.X),
                "WorstI": float(slack_i.X),
            }
            objective_components["slack_penalty"] = float(slack_penalty) * (
                slack_values["Vworst"] + slack_values["WorstI"]
            )

        result.update(
            {
                "objective": float(model.ObjVal),
                "Q_pv_opt": q_solution,
                "Q_ess_opt": q_ess_solution,
                "Q_device_opt": q_device_solution,
                "Q_control_opt": q_control_solution,
                "control_deviation_abs": deviation_values,
                "loss_objective_value": loss_objective_value,
                "objective_components": objective_components,
                "outputs": output_values,
                "safety_slack": slack_values,
            }
        )
        print_solution(
            result,
            pv_nodes=pv_nodes,
            p_pv_available=p_pv_available,
            ess_nodes=ess_nodes,
            q_device_nodes=q_device_nodes,
            q_device_names=q_device_names,
        )
    elif model.status == grb_mod.INFEASIBLE and iis_path:
        iis_out = Path(iis_path)
        if not iis_out.is_absolute():
            iis_out = REPO_ROOT / iis_out
        iis_out.parent.mkdir(parents=True, exist_ok=True)
        print("Model infeasible; computing IIS...")
        model.computeIIS()
        model.write(str(iis_out))
        print(f"IIS written: {iis_out}")
    else:
        print(f"No feasible solution. Status={model.status} ({status_name(model.status)})")
        if model.status == grb_mod.INF_OR_UNBD and int(dual_reductions) != 0:
            print("Hint: rerun with --dual-reductions 0 to distinguish infeasible vs unbounded.")
        if not diagnose_slack:
            print("Hint: rerun with --diagnose-slack to minimize safety-constraint violations.")

    return result


def _set_parser_default(parser: argparse.ArgumentParser, dest: str, value) -> None:
    parser.set_defaults(**{dest: value})
    for action in parser._actions:
        if action.dest == dest:
            action.default = value
            return


def _set_parser_help(parser: argparse.ArgumentParser, dest: str, help_text: str) -> None:
    for action in parser._actions:
        if action.dest == dest:
            action.help = help_text
            return


def _set_parser_choices(parser: argparse.ArgumentParser, dest: str, choices) -> None:
    for action in parser._actions:
        if action.dest == dest:
            action.choices = choices
            return


def build_parser():
    parser = exp17_rpo.build_parser()
    parser.description = "Single-step reactive power optimization with model/mlp MLP-MILP."
    _set_parser_default(parser, "engine", str(DEFAULT_ENGINE_PATH))
    _set_parser_default(parser, "data", str(DEFAULT_DATA_PATH))
    _set_parser_choices(parser, "objective", ["vdev", "ploss", "ploss_vdev"])
    _set_parser_choices(parser, "loss_objective_model", ["none", "surrogate"])
    _set_parser_help(
        parser,
        "objective",
        "MILP objective over MLP heads. vdev matches Exp17 RPO; ploss/ploss_vdev use the model/mlp Ploss_total head.",
    )
    _set_parser_help(parser, "loss_objective_model", "Optional alias for the MLP Ploss_total head; default keeps Exp17-style vdev objective.")
    _set_parser_help(parser, "surrogate_loss_weight", "Unused unless custom callers combine loss heads; kept for CLI compatibility.")
    _set_parser_help(parser, "surrogate_loss_consistency_rel", "No auxiliary loss consistency constraint is used for model/mlp.")
    _set_parser_help(parser, "surrogate_loss_consistency_abs", "No auxiliary loss consistency constraint is used for model/mlp.")
    _set_parser_help(parser, "voltage_weight", "Weight on Vdev_total when objective=ploss_vdev.")
    _set_parser_help(parser, "lambda_q", "Unused for the MLP scalar-head objective; must remain 0.")
    _set_parser_help(parser, "voltage_margin", "Safety margin for the model/mlp Vworst head; default enforces Vworst <= 0.")
    _set_parser_help(parser, "current_margin", "Safety margin for the model/mlp WorstI head; default enforces WorstI <= 0.")
    _set_parser_help(parser, "base_kv", "Unused by the MLP surrogate objective; kept for CLI compatibility.")
    _set_parser_help(parser, "base_mva", "Unused by the MLP surrogate objective; kept for CLI compatibility.")
    _set_parser_help(parser, "diagnose_slack", "Relax Vworst/WorstI safety constraints with nonnegative slacks.")
    return parser


def main():
    args = build_parser().parse_args()
    x_base = None
    pv_q_base = None
    ess_nodes = np.array([], dtype=int)
    ess_s_rated = np.array([], dtype=float)
    ess_p_base = np.array([], dtype=float)
    ess_q_base = np.array([], dtype=float)
    q_device_nodes = np.array([], dtype=int)
    q_device_min = np.array([], dtype=float)
    q_device_max = np.array([], dtype=float)
    q_device_q_base = np.array([], dtype=float)
    q_device_names = []

    if args.source == "dataset":
        cfg = {"v_lower": 0.95, "v_upper": 1.05}
        x_base, p_pv, pv_q_base, topo_mask, pv_nodes, s_rated, meta = (
            get_dataset_operating_point(args.data, cfg, sample_index=args.sample_index)
        )
        p_load = np.zeros(33, dtype=float)
        q_load = np.zeros(33, dtype=float)
        p_pv_override = parse_vector_arg(args.p_pv, expected=len(pv_nodes), name="p_pv")
        if p_pv_override is not None:
            p_pv = p_pv_override

        print("Operating point from dataset:")
        print(f"  data: {meta['data_path']}")
        print(f"  sample_index: {meta['sample_index']} ({meta['sample_reason']})")
        if meta["hour"] is not None:
            print(f"  hour={meta['hour']}, mode={meta['mode']}")
        if meta["Pload_sum"] is not None:
            print(f"  Pload sum: {meta['Pload_sum']:.6f} MW")
            print(f"  Qload sum: {meta['Qload_sum']:.6f} MVar")
        print(f"  X net P sum: {meta['X_P_sum']:.6f} MW")
        print(f"  X net Q sum: {meta['X_Q_sum']:.6f} MVar")
        print(f"  PV buses: {pv_nodes.tolist()}")
        print(f"  S_pv: {s_rated.tolist()}")
        print(f"  P_pv: {p_pv.tolist()}")
        print(f"  baseline pv_q: {pv_q_base.tolist()}")
        ess_nodes = meta["ess_nodes"]
        ess_s_rated = meta["S_ess_mva"]
        ess_p_base = meta["ess_p_base"]
        ess_q_base = meta["ess_q_base"]
        q_device_nodes = meta["q_device_nodes"]
        q_device_min = meta["q_device_min"]
        q_device_max = meta["q_device_max"]
        q_device_q_base = meta["q_device_q_base"]
        q_device_names = meta["q_device_names"]
        if len(ess_nodes):
            print(f"  ESS buses: {ess_nodes.tolist()}")
            print(f"  ESS P/Q base: {ess_p_base.tolist()} / {ess_q_base.tolist()}")
        if len(q_device_nodes):
            print(f"  Q-device buses: {q_device_nodes.tolist()}")
            print(f"  Q-device base: {q_device_q_base.tolist()}")
    else:
        p_load, q_load, p_pv, topo_mask, pv_nodes, s_rated = get_default_operating_point(
            pv_pu=args.pv_pu
        )
        p_load_override = parse_vector_arg(args.p_load, expected=33, name="p_load")
        q_load_override = parse_vector_arg(args.q_load, expected=33, name="q_load")
        p_pv_override = parse_vector_arg(args.p_pv, expected=len(pv_nodes), name="p_pv")
        if p_load_override is not None:
            p_load = p_load_override
        if q_load_override is not None:
            q_load = q_load_override
        if p_pv_override is not None:
            p_pv = p_pv_override

        print("Operating point from profile:")
        print(f"  P_load sum: {float(np.sum(p_load)):.6f} MW")
        print(f"  Q_load sum: {float(np.sum(q_load)):.6f} MVar")
        print(f"  PV buses: {pv_nodes.tolist()}")
        print(f"  P_pv: {p_pv.tolist()}")
    print(f"  closed topology branches: {int(np.sum(topo_mask))} / {len(topo_mask)}")

    run_single_step_rpo(
        p_load,
        q_load,
        p_pv,
        topo_mask,
        pv_nodes=pv_nodes,
        s_rated=s_rated,
        engine_path=args.engine,
        relu_formulation=args.relu_formulation,
        big_m_scale=args.big_m_scale,
        fallback_big_m=args.fallback_big_m,
        objective=args.objective,
        loss_objective_model=args.loss_objective_model,
        surrogate_loss_weight=args.surrogate_loss_weight,
        surrogate_loss_consistency_rel=args.surrogate_loss_consistency_rel,
        surrogate_loss_consistency_abs=args.surrogate_loss_consistency_abs,
        loss_weight=args.loss_weight,
        voltage_weight=args.voltage_weight,
        lambda_q=args.lambda_q,
        voltage_margin=args.voltage_margin,
        current_margin=args.current_margin,
        physical_output_bounds=not args.no_physical_output_bounds,
        trust_region_fraction=args.trust_region_fraction,
        control_deviation_penalty=args.control_deviation_penalty,
        base_kv=args.base_kv,
        base_mva=args.base_mva,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        dual_reductions=args.dual_reductions,
        diagnose_slack=args.diagnose_slack,
        slack_penalty=args.slack_penalty,
        output_flag=0 if args.quiet else 1,
        export_lp=args.export_lp,
        iis_path=args.iis,
        x_base=x_base,
        pv_q_base=pv_q_base,
        ess_nodes=ess_nodes,
        ess_s_rated=ess_s_rated,
        ess_p_base=ess_p_base,
        ess_q_base=ess_q_base,
        q_device_nodes=q_device_nodes,
        q_device_min=q_device_min,
        q_device_max=q_device_max,
        q_device_q_base=q_device_q_base,
        q_device_names=q_device_names,
    )


if __name__ == "__main__":
    main()
