"""Just for tests"""

from itertools import islice
import networkx as nx

from pycaalp.run import create_assembly_digraph, optimize


def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def add_to_cluster(_i, _e, e_next, _op, c):
    c["di_edges"].append((_e, e_next))
    c["operations"].append(_op[0])
    c["operations"].append(_op[1])
    return c


if __name__ == "__main__":
    PART_FNAME = "data/assembly_1/assembly_1_parts.json"

    # ASP
    assembly_digraph = create_assembly_digraph(
        file_name=PART_FNAME,
        reduction_percentage=0,
        num_par_ass=2,
        # one_assembly_policy=False,
        # log_format="DEBUG",
    )
    # print(
    #     k_shortest_paths(
    #         assembly_digraph.assembly_digraph,
    #         source="0_1",
    #         target=f"{assembly_digraph.graph.number_of_edges()}_1",
    #         k=100,
    #         weight="edge_weight",
    #     )
    # )

    method = "dijkstra"
    # weight = "edge_weight"
    weight = "w_conn_subgraphs"

    assert (
        assembly_digraph.assembly_digraph is not None
    ), "assembly_digraph should be populated"
    assert assembly_digraph.graph is not None, "graph should be populated"

    # path = nx.shortest_path(
    #     assembly_digraph.assembly_digraph,
    #     source="0_1",
    #     target=f"{assembly_digraph.graph.number_of_edges()}_1",
    #     weight=weight,
    #     method=method,
    # )

    k = 10
    k_paths = k_shortest_paths(
        assembly_digraph.assembly_digraph,
        source="0_1",
        target=f"{assembly_digraph.graph.number_of_edges()}_1",
        k=k,
        weight=weight,
    )
    # print(list(k_paths))
    k_paths_1 = list(k_paths)[7]

    k_p_1_conn_subgraphs_sequence = [
        assembly_digraph.assembly_digraph[k_paths_1[i]][k_paths_1[i + 1]][
            "connected_subgraphs"
        ]
        for i in range(len(k_paths_1) - 1)
    ]
    k_p_1_operations = [
        assembly_digraph.assembly_digraph[k_paths_1[i]][k_paths_1[i + 1]]["operation"]
        for i in range(len(k_paths_1) - 1)
    ]
    k_p_1_joint_sequence = [
        assembly_digraph.graph.get_edge_data(op[0], op[1])["name"]
        for op in k_p_1_operations
    ]

    print(f"{k_p_1_operations=}")
    print(f"{k_p_1_conn_subgraphs_sequence=}")
    print(f"{k_p_1_joint_sequence=}")

    # Init the clusters
    c_a = {"operations": [], "di_edges": []}
    c_b = {"operations": [], "di_edges": []}
    c_f = {"operations": [], "di_edges": []}
    for i, e in enumerate(k_paths_1):
        if i == assembly_digraph.get_num_layers - 1:
            continue

        op = assembly_digraph.assembly_digraph.get_edge_data(e, k_paths_1[i + 1])[
            "operation"
        ]

        # Check if is the 1 subassembly at the start, not the final join
        if (
            k_p_1_conn_subgraphs_sequence[i] == 0
            and i < assembly_digraph.get_num_layers / 2
        ):
            c_a = add_to_cluster(i, e, k_paths_1[i + 1], op, c_a)
            continue

        # Final joining
        if (
            k_p_1_conn_subgraphs_sequence[i] == 1
            and i > assembly_digraph.get_num_layers / 2
        ) or i == assembly_digraph.get_num_layers - 2:
            c_f = add_to_cluster(i, e, k_paths_1[i + 1], op, c_f)
            continue

        if i < assembly_digraph.get_num_layers - 2:  # BCS the last operation is missing
            if k_p_1_conn_subgraphs_sequence[i + 1] == 1:
                c_f = add_to_cluster(i, e, k_paths_1[i + 1], op, c_f)
                continue

        # Main clustering
        if op[0] in c_a["operations"] or op[1] in c_a["operations"]:  # Add in cluster A
            c_a = add_to_cluster(i, e, k_paths_1[i + 1], op, c_a)
        else:  # Add in cluster B
            c_b = add_to_cluster(i, e, k_paths_1[i + 1], op, c_b)

    print(c_a)
    print(c_b)
    print(c_f)
    assert (
        len(c_a["di_edges"]) + len(c_b["di_edges"]) + len(c_f["di_edges"])
        == assembly_digraph.get_num_layers - 1
    )
