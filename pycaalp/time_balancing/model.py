"""MILP for the assembly line balancing problem.

Problem formulation:
- The problem is formulated as a MILP

Problem variables:
- x_e: binary variable that is equal to 1 if edge e is selected
- y_l_p: binary variable that is equal to 1 if layer l is assigned to phase p
- z_o_p: binary variable that is equal to 1 if operation o is assigned to phase p
- alpha: the objective function variable

Problem constraints:
- Each layer is assigned to one phase
- The sum of all outgoing edges of the root node should be 1
- The sum of all ingoing edges of the final node should be 1
- Ingoing = Outgoing for all other nodes

- x(e) = 1 iff edge e is selected
- y(l,p) = 1 iff layer l is assigned to phase p
- z(o,p) = 1 iff o is performed in phase p
- alpha >= sum(z(o,p)*w(o) for all o in operations)
"""

import pickle
import os
from datetime import datetime
import ast
import networkx as nx
import pyscipopt
from pyscipopt import Model, quicksum
from loguru import logger

from pycaalp.gapp.assembly_digraph import AssemblyDigraph
from pycaalp.time_balancing.utils import results_in_ascending_order


def load_graph_class_from_pickle(file_name: str) -> dict:
    """Loads a dictionary with AssemblyDigraph class attributes from a pickle file
    NOTE: The output is a saved pkl, created from the function:
        gapp/file_formats.py -> assembly_digraph_to

    Args:
        file_name: The path to the pickle file.

    Returns:
        A dict with AssemblyDigraph atrributes.
    """
    with open(file_name, "rb") as _f:
        assem_dig = pickle.load(_f)
    return assem_dig


def get_names_ids_from_solution(assem_dig: dict, operation: dict):
    """Get the names and ids of the operations from the solution.

    Args:
        assem_dig: A dict with AssemblyDigraph attributes.
        operation: Operations of MILP solution.

    Returns:
        Operation dictionary enhanced with part names for operations.
    """
    main_graph_parts = assem_dig["main_graph"].parts
    for key, val in operation.items():
        operation[key] = (
            (main_graph_parts[val[0]], main_graph_parts[val[1]]),
            val,
        )
    return operation


def write_results_to_file(
    model: Model,
    assembly_digraph: nx.DiGraph,
    operations: list[int],
    num_layers: int,
    pickle_filename: str,
    bal_res_filename: str,
    x: dict[tuple:"pyscipopt.scip.Variable"],
    y: dict[tuple:"pyscipopt.scip.Variable"],
    z: dict[tuple:"pyscipopt.scip.Variable"],
    alpha: "pyscipopt.scip.Variable",
    num_phases: int,
    results: dict[str:tuple],  # TODO @ich
    print_all_solutions: bool = False,
) -> None:
    """Write the results of the MILP to a text file.

    Args:
        model:
        assembly_digraph:
        operations:
        num_layers:
        pickle_filename:
        bal_res_filename:
        x:
        y:
        z:
        alpha:
        num_phases:
        operations:
        print_all_solutions:

    """
    with open(bal_res_filename, "a", encoding="utf-8") as _f:
        # Print pkl file used
        _f.write(f"pkl file used: {pickle_filename}\n\n")
        # Print x variables
        _f.write("x variables\n")
        for e in assembly_digraph.edges():
            if model.getVal(x[e]) > 0.9:
                # print(e)
                _f.write(f"x_{e} = {model.getVal(x[e])}\n")

        _f.write("\n")

        # Print y variables
        _f.write("y variables\n")
        for layer in range(num_layers):
            for phase in range(num_phases):
                # print(f"y_{layer}_{phase} = {model.getVal(y[(layer, phase)])}")
                _f.write(f"y_{layer}_{phase} = {model.getVal(y[(layer, phase)])}\n")

        _f.write("\n")

        # Print z variables
        _f.write("z variables\n")
        for operation in operations:
            for phase in range(num_phases):
                # print(f"z_{operation}_{phase} = {model.getVal(z[(operation, phase)])}")
                _f.write(
                    f"z_{operation}_{phase} = {model.getVal(z[(operation, phase)])}\n"
                )

        _f.write("\n")
        # Print alpha variable
        _f.write(f"alpha = {model.getVal(alpha)}\n")

        _f.write("\nX variables: operation\n")
        for key, val in operations.items():
            _f.write(f"{key}: {val}\n")

        if print_all_solutions:
            sols = model.getSols()
            _f.write(f"Number of solutions: {len(sols)}\n")
            for sol in sols:
                _f.write(f"Objective value: {model.getSolObjVal(sol)}\n")
                for v in model.getVars():
                    if v.name.startswith("x") and model.getSolVal(sol, v) == 1:
                        _f.write(f"{v.name}: {model.getSolVal(sol, v)}\n")


def get_pkl_data(pickle_filename: str) -> tuple[nx.DiGraph, nx.Graph]:
    """Get the assebmly digraph and main graph from the pickle file."""
    assem_digr = load_graph_class_from_pickle(
        file_name=pickle_filename,
    )
    assembly_digraph = assem_digr["assembly_digraph"]
    main_graph = assem_digr["main_graph"]
    return assembly_digraph, main_graph


def add_vars(
    model: Model,
    num_layers: int,
    num_phases: int,
    operations,
    assembly_digraph,
    var_type,
):
    """Add the variables to the model."""
    x, y, z = {}, {}, {}

    if var_type == "BINARY":
        for e in assembly_digraph.edges():
            x[e] = model.addVar(vtype="BINARY", name=f"x_{e}")

        for layer in range(num_layers):
            for phase in range(num_phases):
                y[(layer, phase)] = model.addVar(
                    vtype="BINARY", name=f"y_{layer}_{phase}"
                )

        for operation in operations:
            for phase in range(num_phases):
                z[(operation, phase)] = model.addVar(
                    vtype="BINARY", name=f"z_{operation}_{phase}"
                )
    elif var_type == "CONTINUOUS":
        for e in assembly_digraph.edges():
            x[e] = model.addVar(ub=1.0, name=f"x_{e}")

        for layer in range(num_layers):
            for phase in range(num_phases):
                y[(layer, phase)] = model.addVar(ub=1.0, name=f"y_{layer}_{phase}")

        for operation in operations:
            for phase in range(num_phases):
                z[(operation, phase)] = model.addVar(
                    ub=1.0, name=f"z_{operation}_{phase}"
                )
    else:
        raise ValueError("Invalid var_type: {var_type}. Give BINARY or CONTINUOUS")

    alpha = model.addVar("alpha")

    return x, y, z, alpha


def set_objective(x, assembly_digraph, w_balanced, equal_effect_factor, model, alpha):
    """Set the objective function."""
    model.setObjective(
        (1 - w_balanced)
        * quicksum(
            x[e] * assembly_digraph.edges[e]["edge_weight"]
            for e in assembly_digraph.edges()
        )
        + w_balanced * equal_effect_factor * alpha,
        "minimize",
    )


def set_constraints(
    model,
    num_layers,
    num_phases,
    operations,
    assembly_digraph,
    time_weights,
    x,
    y,
    z,
    alpha,
):
    """Set the constraints."""
    # x variable constraints
    # Sum of all outgoing edges of the root node should be 1
    model.addCons(quicksum(x[e] for e in assembly_digraph.out_edges("0_1")) == 1)

    # Sum of all ingoing edges of the final node should be 1
    model.addCons(
        quicksum(x[e] for e in assembly_digraph.in_edges(f"{num_layers}_1")) == 1
    )

    # Ingoing = Outgoing for all other nodes
    for v in assembly_digraph.nodes():
        if v in ("0_1", f"{num_layers}_1"):
            continue
        model.addCons(
            quicksum(x[e] for e in assembly_digraph.in_edges(v))
            == quicksum(x[e] for e in assembly_digraph.out_edges(v))
        )

    # y variable constraint
    for layer in range(num_layers):
        model.addCons(quicksum(y[(layer, phase)] for phase in range(num_phases)) == 1)

    for layer in range(1, num_layers):
        model.addCons(y[layer, 0] <= y[layer - 1, 0])

    for layer in range(1, num_layers):
        for phase in range(1, num_phases):
            model.addCons(
                y[(layer, phase)] <= y[(layer - 1, phase - 1)] + y[(layer - 1, phase)]
            )

    # z variable constraints
    for operation in operations:
        # z(o,p) = 1 iff o is performed in phase p
        model.addCons(
            quicksum(z[(operation, phase)] for phase in range(num_phases)) == 1
        )

    # Add x,y,z constraint
    for e in assembly_digraph.edges():
        # NOTE: what is the layer of the edge a to b in the assebmly digraph?
        # e.g., (1_1, 2_1) is in the assebmly digraph, but what is the layer of this edge?
        # I think start from the root node
        layer = int(e[0].split("_")[0])
        for phase in range(num_phases):
            model.addCons(
                z[(assembly_digraph.edges[e]["operation"], phase)]
                >= x[e] + y[(layer, phase)] - 1
            )

    # alpha and z constraint
    for phase in range(num_phases):
        model.addCons(
            alpha
            >= quicksum(
                z[(operation, phase)] * time_weights[operation]
                for operation in operations
            )
        )


def check_and_print_results(
    model, assembly_digraph, main_graph, operations, print_all_solutions, var_type
):
    """Check if the results are correct and print them."""
    # Check if all the links/operation in the x variables are in the main graph
    results = {}
    results["operations"] = {}
    results["technology"] = {}
    results["handling"] = {}
    results["tolerance"] = {}
    results["time"] = {}
    results["absolute_handling"] = {}
    results["absolute_tolerance"] = {}
    results["absolute_time"] = {}
    results["phase"] = {}
    results["operations_per_phase"] = {}
    results["time_per_phase"] = {}
    results["absolute_time_per_phase"] = {}
    phase_per_operation_list = []
    used_operations = []
    technology = nx.get_edge_attributes(main_graph, "technology")
    handling = nx.get_edge_attributes(main_graph, "handling")
    tolerance = nx.get_edge_attributes(main_graph, "tolerance")
    time = nx.get_edge_attributes(main_graph, "time")
    abs_handling = nx.get_edge_attributes(main_graph, "absolute_handling")
    abs_tolerance = nx.get_edge_attributes(main_graph, "absolute_tolerance")
    abs_time = nx.get_edge_attributes(main_graph, "absolute_time")

    for var in model.getVars():
        if var.name.startswith("x") and model.getVal(var) > 0.99:
            edge_str = ast.literal_eval(var.name[2:])
            oper_used = assembly_digraph.edges[edge_str]["operation"]
            used_operations.append(oper_used)  # for testing
            # print(f"{_x.name}: {oper_used}")
            results["operations"][edge_str] = oper_used
            results["technology"][edge_str] = technology[oper_used]
            results["handling"][edge_str] = handling[oper_used]
            results["tolerance"][edge_str] = tolerance[oper_used]
            results["time"][edge_str] = time[oper_used]
            results["absolute_handling"][edge_str] = abs_handling[oper_used]
            results["absolute_tolerance"][edge_str] = abs_tolerance[oper_used]
            results["absolute_time"][edge_str] = abs_time[oper_used]

        if var.name.startswith("y") and model.getVal(var) > 0.99:
            phase = int(var.name.split("_")[-1])
            phase_per_operation_list.insert(0, phase)

        if var.name == "alpha":
            results["alpha"] = model.getVal(var)

    for ed_str, ph in zip(list(results["operations"].keys()), phase_per_operation_list):
        results["phase"][ed_str] = ph

        results["time_per_phase"][ph] = (
            results["time_per_phase"].get(ph, 0) + time[results["operations"][ed_str]]
        )

        results["absolute_time_per_phase"][ph] = (
            results["absolute_time_per_phase"].get(ph, 0)
            + abs_time[results["operations"][ed_str]]
        )

    for ph in phase_per_operation_list:
        results["operations_per_phase"][ph] = (
            results["operations_per_phase"].get(ph, 0) + 1  # Just accumulate
        )

    # Check if all the used operations are in the operations list
    if var_type == "BINARY":
        for operation in operations:
            if operation not in used_operations:
                raise ValueError("Operation not in used operations")

    # Print all solutions
    if print_all_solutions:
        sols = model.getSols()
        print(f"Number of solutions: {len(sols)}")
        for sol in sols:
            print(f"Objective value: {model.getSolObjVal(sol)}")
            for v in model.getVars():
                if v.name.startswith("x") and model.getSolVal(sol, v) > 0.7:
                    oper_used = assembly_digraph.edges[ast.literal_eval(v.name[2:])][
                        "operation"
                    ]

                    print(
                        f"{v.name}: {model.getSolVal(sol, v)}, {main_graph.parts[oper_used[0]]}, {main_graph.parts_dict[oper_used[1]]} "
                    )

    operations_list = []
    operations_list = [[] for _ in range(max(phase_per_operation_list) + 1)]
    op_list = list(results["operations"].values())
    for i in range(len(op_list) - 1, -1, -1):
        operations_list[phase_per_operation_list[i]].append(op_list[i])

    return results, operations_list


def run_milp(
    assembly_digraph: AssemblyDigraph = None,
    pickle_filename: str = None,
    full_result_output: bool = False,
    return_model: bool = False,
    bal_res_filename: str = None,
    write_milp_res: str = False,
    print_all_solutions: str = False,
    num_phases: int = 3,
    w_balanced: int = 0.5,
    relative_gap: float = 0.05,
    hide_output: bool = True,
    var_type: bool = "BINARY",
):
    """Run the MILP for the assembly line balancing problem.

    The PKL file should be a dict and contain the following:
    - assembly_digraph: The assembly digraph.
    - main_graph: The main graph.

    Args: TODO: update
        pickle_filename: The path to the pickle file.
        write_milp_res: If True, write the results to a text file.
        print_all_solutions: If True, print all solutions.
        num_phases: The number of phases.
        factor: The factor for the objective function.
        var_type: The type of variables to use. Either "BINARY" or "CONTINUOUS".

    Returns:
        The objective function value.
    """
    logger.info("Running MIP solver ...")
    # Access the assebmly digraph
    if assembly_digraph:
        assem_digr = assembly_digraph.assembly_digraph
        main_graph = assembly_digraph.graph
    elif pickle_filename:
        assem_digr, main_graph = get_pkl_data(pickle_filename)
    else:
        raise ValueError("Please provide an assembly digraph class or a pickle file")

    # print(f"Running MILP for {pickle_filename}")
    # Assembly digraph attributes
    num_layers = main_graph.number_of_edges()
    operations = list(main_graph.edges())
    # NOTE: constant time weights for each operation, independent of the layer
    time_weights = nx.get_edge_attributes(main_graph, "time")

    # Equal effect factor using shortest path
    equal_effect_factor = (
        assembly_digraph.sum_of_sh_path_weights
        * num_phases
        / sum(time_weights.values())
    )

    # Create MILP model
    model = Model()

    # Set model parameters (https://www.scipopt.org/doc/html/PARAMETERS.php)
    model.hideOutput(hide_output)
    model.setParam("limits/gap", relative_gap)  # relative gap
    # Use parallel mode
    model.setParam("parallel/maxnthreads", 4)

    # STEP 1: ADD VARIABLES
    x, y, z, alpha = add_vars(
        model, num_layers, num_phases, operations, assem_digr, var_type
    )

    # STEP 2: SET OBJECTIVE FUNCTION
    set_objective(x, assem_digr, w_balanced, equal_effect_factor, model, alpha)

    # STEP 3: SET CONSTRAINTS
    set_constraints(
        model,
        num_layers,
        num_phases,
        operations,
        assem_digr,
        time_weights,
        x,
        y,
        z,
        alpha,
    )

    # Solve model
    model.optimize()

    # Print results
    model.printBestSol()

    results, operations_list = check_and_print_results(
        model, assem_digr, main_graph, operations, print_all_solutions, var_type
    )

    results = results_in_ascending_order(results)

    if write_milp_res:
        # Create balancing results dir
        res_dir = "balancing_results/"
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        # Add date and time to the filename
        if bal_res_filename is None:
            bal_res_filename = (
                res_dir
                + pickle_filename.split("/")[-1][:-4]
                + "_"
                + var_type
                + "_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
                + ".txt"
            )

        write_results_to_file(
            model=model,
            assembly_digraph=assem_digr,
            operations=operations,
            num_layers=num_layers,
            pickle_filename=pickle_filename,
            bal_res_filename=bal_res_filename,
            x=x,
            y=y,
            z=z,
            alpha=alpha,
            num_phases=num_phases,
            results=results,
            print_all_solutions=print_all_solutions,
        )

        print(f"\n----------- Results are saved in {bal_res_filename} ------------\n")

    if return_model:
        return model, results
    if full_result_output:
        return results, operations_list
    return operations_list
