"""Operations to measure assembly digraph paths"""

from itertools import islice
from loguru import logger
import networkx as nx


def calculate_num_simple_paths(assem_dig: "AssemblyDigraph") -> int:
    """Calculate all simple paths from layer 0 to the final layer

    Args:
        assem_dig: AssemblyDigraph, including a constructed graph.

    Returns:
        Number of all simple paths from starting node to the final one.
    """
    final_layer = assem_dig.graph.number_of_edges()
    all_simple_paths = nx.all_simple_paths(
        assem_dig.assembly_digraph,
        source="0_1",
        target=f"{final_layer}_1",
    )
    num_all_simple_paths = 0
    for _ in all_simple_paths:
        num_all_simple_paths += 1
    return num_all_simple_paths


def calculate_sum_of_sh_path_weights(cutsets):
    sh_path = nx.shortest_path(
        cutsets.assembly_digraph,
        source="0_1",
        target=f"{cutsets.get_num_layers-1}_1",
        weight="edge_weight",
    )
    edge_weights = nx.get_edge_attributes(cutsets.assembly_digraph, "edge_weight")
    edge_sum = 0
    for i in range(len(sh_path) - 1):
        edge_sum += edge_weights[(sh_path[i], sh_path[i + 1])]

    return edge_sum


def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def set_blended_weights(
    digraph,
    time_weights,
    num_phases,
    lam,
    out_attr="blended_weight",
):
    """Write a per-edge blended enumeration weight in place and return the digraph.

        w_blend(e; λ) = (1 - λ) · ew_norm(e)  +  λ · misalign_norm(e)

    ``ew_norm`` is the min-max-normalised engineering ``edge_weight``. ``misalign``
    is a continuous, *path-additive* proxy for the makespan-balance objective:
    each ideal phase boundary b_k = k · phase_width is crossed by exactly one edge
    of any source→sink path (cumulative time is monotone), and that edge is charged
    the best achievable cut error ``min(b_k − t_u, t_v − b_k) / phase_width`` — i.e.
    how far the nearer endpoint (the only places a phase can actually be cut) sits
    from the ideal boundary. Summed along a path this is exactly "how cleanly this
    sequence can be split into equal-load phases".

    Both terms are min-max normalised to [0, 1] so λ is a fair knob and matches the
    MIP objective's λ. Enumerating k-shortest paths by ``out_attr`` then targets the
    λ-specific *compromise* path directly, instead of unioning the two corner
    enumerations (edge_weight, time_balanced_weight) and hoping one is right.

    NOTE: the balance term is a heuristic surrogate — the true objective uses
    α = max phase time (not a path sum) — so this must be validated empirically.
    """
    cum_time = compute_cumulative_operation_time(digraph, time_weights)
    sinks = [n for n in digraph.nodes() if digraph.out_degree(n) == 0]
    t_max = max((cum_time.get(n, 0.0) for n in sinks), default=0.0) or 1.0
    phase_width = t_max / num_phases if num_phases else t_max
    boundaries = [k * phase_width for k in range(1, num_phases)]

    misalign = {}
    for u, v in digraph.edges():
        t_u, t_v = cum_time.get(u, 0.0), cum_time.get(v, 0.0)
        err = 0.0
        for b in boundaries:
            if t_u < b <= t_v:
                err += min(b - t_u, t_v - b) / phase_width
        misalign[(u, v)] = err

    edge_w = nx.get_edge_attributes(digraph, "edge_weight")

    def _norm(d):
        vals = d.values()
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        rng = hi - lo if hi > lo else 1.0
        return {key: (val - lo) / rng for key, val in d.items()}

    ew_n, mis_n = _norm(edge_w), _norm(misalign)
    for u, v in digraph.edges():
        digraph[u][v][out_attr] = (1 - lam) * ew_n.get((u, v), 0.0) + lam * mis_n.get(
            (u, v), 0.0
        )
    return digraph


def compute_cumulative_operation_time(digraph, time_weights):
    """Compute cumulative operation time for each node in the digraph via forward DP.

    Each node encodes a specific assembly cutset, so the set of assembled joints is
    fixed per node — cumulative time is path-independent.

    Args:
        digraph: nx.DiGraph with 'operation' attribute on edges.
        time_weights: dict mapping operation tuple -> normalized time (from main graph).

    Returns:
        dict: mapping node_id -> cumulative_time
    """
    cum_time = {"0_1": 0.0}

    for node in nx.topological_sort(digraph):
        if node not in cum_time:
            cum_time[node] = 0.0

        for u, v, data in digraph.out_edges(node, data=True):
            operation = data.get("operation")
            if operation:
                op_time = time_weights.get(operation, 0.0)
                cum_time[v] = cum_time[u] + op_time
            else:
                logger.warning(
                    f"Edge ({u}, {v}) has no 'operation' attribute — time skipped"
                )
                if v not in cum_time:
                    cum_time[v] = cum_time[u]

    return cum_time


def compute_phase_boundary_crossing(digraph, time_weights, num_phases=3):
    """Identify edges that cross between phase regions.

    For num_phases=P, divide total operation time into P regions at boundaries
    0, T/P, 2T/P, ..., T. Mark edges that transition between regions.

    Args:
        digraph: nx.DiGraph with operation edges.
        time_weights: dict mapping operation tuple -> normalized time (from main graph).
        num_phases: Number of assembly phases.

    Returns:
        dict: mapping (u,v) edge -> phase_region_crossed (bool)
    """
    cum_time = compute_cumulative_operation_time(digraph, time_weights)

    # Find total time at the sink (node with no outgoing edges)
    sinks = [n for n in digraph.nodes() if digraph.out_degree(n) == 0]
    t_max = max((cum_time.get(n, 0.0) for n in sinks), default=0.0)

    if t_max == 0:
        t_max = 1.0

    phase_width = t_max / num_phases

    crossing = {}
    for u, v, data in digraph.edges(data=True):
        t_u = cum_time.get(u, 0.0)
        t_v = cum_time.get(v, 0.0)

        region_u = int(t_u / phase_width)
        region_v = int(t_v / phase_width)

        # Clamp to valid regions
        region_u = min(region_u, num_phases)
        region_v = min(region_v, num_phases)

        crosses = region_v > region_u
        crossing[(u, v)] = crosses

    return crossing, cum_time


def create_time_balanced_edge_weights(
    digraph,
    time_weights,
    original_weight_attr="edge_weight",
    num_phases=3,
    lambda_param=0.5,
):
    """Modify edge weights to incorporate phase-balance heuristic.

    Edges that cross phase boundaries get a bonus (lower weight), encouraging
    k-shortest-paths to find paths with good phase alignment.

    Args:
        digraph: nx.DiGraph with operation edges.
        time_weights: dict mapping operation tuple -> normalized time (from main graph).
        original_weight_attr: Name of existing weight attribute.
        num_phases: Number of assembly phases.
        lambda_param: Trade-off weight (0=ignore, 1=maximize boundary crossing).

    Returns:
        dict: mapping (u,v) edge -> modified_weight
    """
    crossing, cum_time = compute_phase_boundary_crossing(
        digraph, time_weights, num_phases
    )

    # Normalize original weights
    orig_weights = nx.get_edge_attributes(digraph, original_weight_attr)
    if orig_weights:
        max_w = max(orig_weights.values()) if orig_weights.values() else 1.0
        min_w = min(orig_weights.values()) if orig_weights.values() else 0.0
        w_range = max_w - min_w if max_w > min_w else 1.0
    else:
        max_w = min_w = w_range = 1.0

    modified_weights = {}
    for u, v, data in digraph.edges(data=True):
        orig_w = data.get(original_weight_attr, 0.0)
        normalized_w = (orig_w - min_w) / w_range if w_range > 0 else 0.0

        # Bonus for crossing: reduce weight by a fraction
        crossing_bonus = -0.1 if crossing.get((u, v), False) else 0.0

        modified_w = normalized_w + lambda_param * crossing_bonus
        modified_weights[(u, v)] = max(0.001, modified_w)  # Keep positive

    return modified_weights
