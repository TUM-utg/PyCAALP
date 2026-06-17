"""
Option 2 — k-Path Subgraph MIP
================================
Build the union subgraph of the k shortest assembly paths, then run the
full MIP (joint path-selection + phase-assignment) on this reduced digraph.

The subgraph is deterministic and reproducible: it contains exactly those
edges that lie on at least one of the k lightest assembly sequences.  For
large digraphs this dramatically cuts the number of x variables while
guaranteeing that the globally optimal solution over those k paths is found.

Public API
----------
build_kpath_subgraph(assembly_digraph_obj, k) -> nx.DiGraph
solve_by_subgraph_mip(assembly_digraph_obj, k, ...) -> operations_list
    (or (results, operations_list) when full_result_output=True)
"""

from dataclasses import dataclass

import networkx as nx

from pycaalp.gapp.paths import k_shortest_paths
from pycaalp.time_balancing.model import run_mip


@dataclass
class _SubgraphProxy:
    """Minimal duck-type of AssemblyDigraph accepted by run_mip."""

    assembly_digraph: nx.DiGraph
    graph: nx.Graph
    sum_of_sh_path_weights: float


def build_kpath_subgraph(
    assembly_digraph_obj,
    k: int,
    weight_attr: str | list[str] = "time_balanced_weight",
) -> nx.DiGraph:
    """Return the union of the k shortest paths as a subgraph of the assembly digraph.

    All edge attributes (edge_weight, operation, time_balanced_weight, …) are preserved.

    Args:
        assembly_digraph_obj: AssemblyDigraph instance.
        k: Number of shortest paths to enumerate.
        weight_attr: Edge attribute(s) used for path enumeration.
            "edge_weight" uses pure engineering weights;
            "time_balanced_weight" biases toward phase-boundary-crossing edges.
            A list runs the enumeration once per attribute and unions the
            results — e.g. ["edge_weight", "time_balanced_weight"] yields the
            union of up to 2*k paths (k cheapest by engineering cost plus k
            cheapest by balance), combining the strengths of both rankings.
    """
    digraph = assembly_digraph_obj.assembly_digraph
    num_joints = assembly_digraph_obj.graph.number_of_edges()

    weight_attrs = [weight_attr] if isinstance(weight_attr, str) else weight_attr

    edge_set = set()
    for attr in weight_attrs:
        paths = k_shortest_paths(
            digraph,
            source="0_1",
            target=f"{num_joints}_1",
            k=k,
            weight=attr,
        )
        for path_nodes in paths:
            for i in range(len(path_nodes) - 1):
                edge_set.add((path_nodes[i], path_nodes[i + 1]))

    return digraph.edge_subgraph(edge_set).copy()


def solve_by_subgraph_mip(
    assembly_digraph_obj,
    k: int = 500,
    num_phases: int = 3,
    w_balanced: float = 0.5,
    relative_gap: float = 0.0,
    hide_output: bool = True,
    full_result_output: bool = False,
    weight_attr: str | list[str] = "time_balanced_weight",
):
    """Solve the assembly line balancing problem on the k-path union subgraph.

    Builds the subgraph deterministically, wraps it in a proxy, then
    delegates to run_mip unchanged — no modifications to the MIP formulation.

    Args:
        assembly_digraph_obj: AssemblyDigraph instance.
        k: Number of shortest paths whose union forms the subgraph.
        num_phases: Number of assembly phases.
        w_balanced: Time-balancing weight in [0, 1].
        relative_gap: SCIP relative optimality gap limit (0.0 = solve to optimality).
        hide_output: Suppress SCIP solver output.
        full_result_output: If True return (results, operations_list),
            otherwise return operations_list.
        weight_attr: Edge attribute(s) used for path enumeration. A list
            unions the per-attribute enumerations (up to len*k paths); see
            build_kpath_subgraph.
    """
    subgraph = build_kpath_subgraph(assembly_digraph_obj, k, weight_attr=weight_attr)
    proxy = _SubgraphProxy(
        assembly_digraph=subgraph,
        graph=assembly_digraph_obj.graph,
        sum_of_sh_path_weights=assembly_digraph_obj.sum_of_sh_path_weights,
    )
    return run_mip(
        assembly_digraph=proxy,
        num_phases=num_phases,
        w_balanced=w_balanced,
        relative_gap=relative_gap,
        hide_output=hide_output,
        full_result_output=full_result_output,
    )
