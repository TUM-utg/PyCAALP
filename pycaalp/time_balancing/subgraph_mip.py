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

from pycaalp.gapp.paths import diverse_shortest_paths, k_shortest_paths
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
    for attr in weight_attrs:  # It might be more than 1 attr weights
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


def build_blended_union_subgraph(
    assembly_digraph_obj,
    k: int,
    blends,
    tmp_attr: str = "_blend_tmp",
) -> nx.DiGraph:
    """Union of the k shortest paths enumerated by the blended weight at several
    blend values — the "frontier union" analogue of the combined strategy.

    The combined strategy unions two *corner* enumerations (edge_weight, the
    binary balance weight). This unions the blended weight at each value in
    ``blends``: blend=0 reproduces the edge_weight ranking exactly (min-max is
    monotone), blend=1 is the continuous balance ranking, and the intermediate
    values add the genuine compromise paths. The multiple enumerations also
    supply the path diversity a single blended enumeration lacks. Up to
    len(blends)*k paths; enumeration is λ-agnostic (the MIP still solves at the
    true λ).
    """
    digraph = assembly_digraph_obj.assembly_digraph
    num_joints = assembly_digraph_obj.graph.number_of_edges()

    edge_set = set()
    for b in blends:
        assembly_digraph_obj.set_blended_weights(b, out_attr=tmp_attr)
        paths = k_shortest_paths(
            digraph, source="0_1", target=f"{num_joints}_1", k=k, weight=tmp_attr
        )
        for path_nodes in paths:
            for i in range(len(path_nodes) - 1):
                edge_set.add((path_nodes[i], path_nodes[i + 1]))

    return digraph.edge_subgraph(edge_set).copy()


def build_diverse_subgraph(
    assembly_digraph_obj,
    k: int,
    lam: float,
    penalty: float = 0.5,
    base_attr: str = "_diverse_base",
) -> nx.DiGraph:
    """Subgraph from k *diverse* paths enumerated by the blended weight at λ.

    The blended-union strategy unions Yen k-shortest paths at several blends; its
    coverage saturates (near-duplicate paths) so the objective plateaus above the
    optimum in the hard high-λ regime. This enumerates by penalised re-routing
    (``diverse_shortest_paths``) on the single blended weight at the true λ — the
    exact quantity the MIP minimises — so the paths spread across the compromise
    region instead of clustering. Cheaper too: k linear DAG shortest paths rather
    than Yen.

    ``penalty`` controls the diversity/coverage trade-off (see
    ``diverse_shortest_paths``).
    """
    digraph = assembly_digraph_obj.assembly_digraph
    num_joints = assembly_digraph_obj.graph.number_of_edges()

    # Write the blended weight for this λ onto its own attribute, then diversify.
    assembly_digraph_obj.set_blended_weights(lam, out_attr=base_attr)
    paths = diverse_shortest_paths(
        digraph,
        source="0_1",
        target=f"{num_joints}_1",
        k=k,
        weight=base_attr,
        penalty=penalty,
    )

    edge_set = set()
    for path_nodes in paths:
        for i in range(len(path_nodes) - 1):
            edge_set.add((path_nodes[i], path_nodes[i + 1]))

    return digraph.edge_subgraph(edge_set).copy()


def build_adaptive_subgraph(
    assembly_digraph_obj,
    k: int,
    blends,
    lam: float,
    penalty: float = 0.5,
    stall_rounds: int = 2,
    base_attr: str = "_adapt",
) -> nx.DiGraph:
    """bl-union until its edge growth stalls, then switch to diverse re-routing.

    Draws up to ``k`` paths total, growing one edge set:

    * **Phase 1 (bl-union):** grow the subgraph one *round* at a time — a round
      pulls the next Yen path from every blend generator, so after ``r`` rounds
      the edge set is **exactly** ``build_blended_union_subgraph(ad, r, blends)``.
      adaptive therefore lies *on* the bl-union curve until it switches.
    * **switch:** after ``stall_rounds`` consecutive *duplicate rounds* — rounds
      that add **no** new edge, i.e. every path bl-union just drew is already in
      the subgraph. That is the crisp "bl-union has nothing new to contribute"
      signal (no arbitrary rate threshold). ``stall_rounds`` (default 2) requires
      it to persist because Yen's growth is bursty — a lone zero-edge round can
      be a fluke while the next path still introduces an edge; ``stall_rounds=1``
      switches on the very first duplicate round.
    * **Phase 2 (diverse):** spend the remaining budget on penalised re-routing
      on the blended weight at the true λ, unioned with the Phase-1 edges. The
      diverse pass starts unbanned (the optimum-carrying path usually overlaps
      bl-union's edges, so banning them would hide it); the re-routing penalty
      then spreads onto fresh edges — escaping the plateau that traps bl-union in
      the high-λ regime.

    For k below the switch point this is exactly bl-union; past it, the extra
    paths come from the penalty approach. Deterministic.
    """
    digraph = assembly_digraph_obj.assembly_digraph
    src, tgt = "0_1", f"{assembly_digraph_obj.graph.number_of_edges()}_1"

    # Phase 1: one Yen generator per blend, each on its own weight attribute.
    gens = []
    for i, b in enumerate(blends):
        attr = f"{base_attr}_blend_{i}"
        assembly_digraph_obj.set_blended_weights(b, out_attr=attr)
        gens.append(nx.shortest_simple_paths(digraph, src, tgt, weight=attr))

    edge_set = set()
    rounds = 0
    zero_streak = 0  # consecutive rounds that added no new edge (duplicate rounds)
    active = list(range(len(gens)))
    while rounds < k and active:
        before = len(edge_set)
        for gi in list(active):
            try:
                path = next(gens[gi])
            except StopIteration:
                active.remove(gi)
                continue
            edge_set.update(zip(path, path[1:]))
        rounds += 1
        if len(edge_set) == before:  # duplicate round: nothing new
            zero_streak += 1
            if zero_streak >= stall_rounds:
                break  # bl-union has stalled — only duplicate paths left
        else:
            zero_streak = 0

    # Phase 2: diverse re-routing on the blended weight at the true λ, unioned
    # with the Phase-1 edges. Not pre-banned: the optimum-carrying path usually
    # overlaps bl-union's edges, so banning them would hide it — the diverse pass
    # must be free to (re)find it, then spread onto fresh edges from there.
    if rounds < k:
        wattr = f"{base_attr}_lam"
        assembly_digraph_obj.set_blended_weights(lam, out_attr=wattr)
        for path in diverse_shortest_paths(
            digraph, src, tgt, k - rounds, wattr, penalty=penalty
        ):
            edge_set.update(zip(path, path[1:]))

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
    subgraph: nx.DiGraph | None = None,
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
        subgraph: Pre-built k-path subgraph. If given, the build step is skipped
            (lets the caller time enumeration/build and solve separately and
            avoids building the subgraph twice).
    """
    if subgraph is None:
        subgraph = build_kpath_subgraph(
            assembly_digraph_obj, k, weight_attr=weight_attr
        )
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
