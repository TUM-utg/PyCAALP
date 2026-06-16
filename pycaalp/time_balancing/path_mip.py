"""
Option 1b — Path-Enumeration MIP
==================================
Enumerate the k shortest assembly paths and solve a small phase-assignment
MIP for each one.  Since the path fixes all x variables, the linking
constraint z(op,p) >= x[e] + y(l,p) - 1 reduces to z(op,p) >= y(l,p),
making z redundant with y.  The sub-MIP therefore contains only y and alpha
(N×P binary + 1 continuous variables — ~51 vars for N=17, P=3) and SCIP
solves each instance in milliseconds.  The best solution across all k paths
is returned.

Public API
----------
solve_by_path_mip(assembly_digraph_obj, k, num_phases, w_balanced, ...)
    -> operations_list  (or (results, operations_list) when full_result_output=True)
"""

import networkx as nx
from pyscipopt import Model, quicksum

from pycaalp.gapp.paths import k_shortest_paths
from pycaalp.time_balancing.utils import results_in_ascending_order


def _run_single_path_mip(
    path_nodes, assembly_digraph_obj, num_phases, w_balanced, hide_output
):
    digraph = assembly_digraph_obj.assembly_digraph
    main_graph = assembly_digraph_obj.graph
    num_layers = main_graph.number_of_edges()  # N = number of joints

    time_w = nx.get_edge_attributes(main_graph, "time")
    abs_time_w = nx.get_edge_attributes(main_graph, "absolute_time")
    technology = nx.get_edge_attributes(main_graph, "technology")
    handling = nx.get_edge_attributes(main_graph, "handling")
    tolerance = nx.get_edge_attributes(main_graph, "tolerance")
    mass = nx.get_edge_attributes(main_graph, "mass")
    abs_handling = nx.get_edge_attributes(main_graph, "absolute_handling")
    abs_tolerance = nx.get_edge_attributes(main_graph, "absolute_tolerance")
    abs_mass = nx.get_edge_attributes(main_graph, "absolute_mass")

    path_edges = [
        (path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)
    ]
    path_cost = sum(digraph.edges[e]["edge_weight"] for e in path_edges)
    # ordered_layers is always [0, 1, ..., N-1]; path edges cross consecutive layers
    ordered_ops = [digraph.edges[e]["operation"] for e in path_edges]

    equal_effect_factor = (
        assembly_digraph_obj.sum_of_sh_path_weights * num_phases / sum(time_w.values())
    )

    model = Model()
    model.hideOutput(hide_output)

    # z is redundant with y on a fixed path, so only y and alpha are needed
    y = {
        (l, p): model.addVar(vtype="BINARY", name=f"y_{l}_{p}")
        for l in range(num_layers)
        for p in range(num_phases)
    }
    alpha = model.addVar("alpha")

    # path_cost is constant per path; only alpha varies across phase splits
    model.setObjective(w_balanced * equal_effect_factor * alpha, "minimize")

    # Each layer assigned to exactly one phase
    for l in range(num_layers):
        model.addCons(quicksum(y[(l, p)] for p in range(num_phases)) == 1)

    # Phase monotonicity — phases must be contiguous blocks of layers
    for l in range(1, num_layers):
        model.addCons(y[(l, 0)] <= y[(l - 1, 0)])
    for l in range(1, num_layers):
        for p in range(1, num_phases):
            model.addCons(y[(l, p)] <= y[(l - 1, p - 1)] + y[(l - 1, p)])

    # Alpha bounds the maximum phase time (using y directly since z == y on a fixed path)
    for p in range(num_phases):
        model.addCons(
            alpha
            >= quicksum(y[(l, p)] * time_w[ordered_ops[l]] for l in range(num_layers))
        )

    model.optimize()

    status = model.getStatus()
    if status not in ("optimal", "timelimit"):
        raise RuntimeError(f"Path sub-MIP returned status '{status}'")

    layer_to_phase = {
        l: p
        for l in range(num_layers)
        for p in range(num_phases)
        if model.getVal(y[(l, p)]) > 0.5
    }

    alpha_val = model.getVal(alpha)
    total_obj = (
        1 - w_balanced
    ) * path_cost + w_balanced * equal_effect_factor * alpha_val

    # Build results in reversed path order so results_in_ascending_order produces
    # ascending layer order, matching the behaviour of run_mip
    results = {
        k: {}
        for k in (
            "operations",
            "technology",
            "handling",
            "tolerance",
            "time",
            "mass",
            "absolute_handling",
            "absolute_tolerance",
            "absolute_time",
            "absolute_mass",
            "phase",
            "operations_per_phase",
            "time_per_phase",
            "absolute_time_per_phase",
        )
    }
    results["alpha"] = alpha_val

    for edge, op, l in zip(
        reversed(path_edges),
        reversed(ordered_ops),
        reversed(range(num_layers)),
    ):
        ph = layer_to_phase[l]
        results["operations"][edge] = op
        results["technology"][edge] = technology[op]
        results["handling"][edge] = handling[op]
        results["tolerance"][edge] = tolerance[op]
        results["time"][edge] = time_w[op]
        results["mass"][edge] = mass[op]
        results["absolute_handling"][edge] = abs_handling[op]
        results["absolute_tolerance"][edge] = abs_tolerance[op]
        results["absolute_time"][edge] = abs_time_w[op]
        results["absolute_mass"][edge] = abs_mass[op]
        results["phase"][edge] = ph
        results["time_per_phase"][ph] = (
            results["time_per_phase"].get(ph, 0) + time_w[op]
        )
        results["absolute_time_per_phase"][ph] = (
            results["absolute_time_per_phase"].get(ph, 0) + abs_time_w[op]
        )
        results["operations_per_phase"][ph] = (
            results["operations_per_phase"].get(ph, 0) + 1
        )

    results = results_in_ascending_order(results)

    num_phases_used = max(results["phase"].values()) + 1 if results["phase"] else 0
    operations_list = [[] for _ in range(num_phases_used)]
    for edge, op in results["operations"].items():
        operations_list[results["phase"][edge]].append(op)

    return total_obj, results, operations_list


def solve_by_path_mip(
    assembly_digraph_obj,
    k: int = 500,
    num_phases: int = 3,
    w_balanced: float = 0.5,
    hide_output: bool = True,
    full_result_output: bool = False,
    weight_attr: str = "time_balanced_weight",
):
    """Solve the assembly line balancing problem by enumerating k shortest paths.

    For each path the x variables are fixed and a small phase-assignment MIP
    is solved over y and alpha only.  Returns the best solution found.

    Args:
        assembly_digraph_obj: AssemblyDigraph instance.
        k: Number of shortest paths to enumerate.
        num_phases: Number of assembly phases.
        w_balanced: Time-balancing weight in [0, 1].
        hide_output: Suppress SCIP solver output.
        full_result_output: If True return (results, operations_list),
            otherwise return operations_list.
        weight_attr: Edge attribute used for path enumeration.
            "edge_weight" uses pure engineering weights;
            "time_balanced_weight" biases toward phase-boundary-crossing edges.
    """
    digraph = assembly_digraph_obj.assembly_digraph
    num_joints = assembly_digraph_obj.graph.number_of_edges()

    paths = k_shortest_paths(
        digraph,
        source="0_1",
        target=f"{num_joints}_1",
        k=k,
        weight=weight_attr,
    )

    best_obj, best_results, best_ops_list = float("inf"), None, None
    for path_nodes in paths:
        obj, results, ops_list = _run_single_path_mip(
            path_nodes, assembly_digraph_obj, num_phases, w_balanced, hide_output
        )
        if obj < best_obj:
            best_obj, best_results, best_ops_list = obj, results, ops_list

    if best_results is None:
        raise ValueError(f"No feasible path found among the {k} shortest paths")

    if full_result_output:
        return best_results, best_ops_list
    return best_ops_list
