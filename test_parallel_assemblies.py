"""Just for tests"""

from itertools import islice
import networkx as nx

from pycaalp.run import create_assembly_digraph, optimize


def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


if __name__ == "__main__":
    PART_FNAME = "data/assembly_1/assembly_1_parts.json"

    # ASP
    assembly_digraph = create_assembly_digraph(
        file_name=PART_FNAME,
        reduction_percentage=60,
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
    weight = "edge_weight"
    weight = "num_connected_subgraphs"

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

    k = 100
    k_paths = k_shortest_paths(
        assembly_digraph.assembly_digraph,
        source="0_1",
        target=f"{assembly_digraph.graph.number_of_edges()}_1",
        k=k,
        weight=weight,
    )
    # print(list(k_paths))
    k_paths_1 = list(k_paths)[40]

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
    print()

    # operations = [
    #     assembly_digraph.assembly_digraph[path[i]][path[i + 1]]["operation"]
    #     for i in range(len(path) - 1)
    # ]
    # joint_sequence = [
    #     assembly_digraph.graph.get_edge_data(op[0], op[1])["name"] for op in operations
    # ]
    # conn_subgraphs_sequence = [
    #     assembly_digraph.assembly_digraph[path[i]][path[i + 1]]["connected_subgraphs"]
    #     for i in range(len(path) - 1)
    # ]
    # num_conn_subgraphs_sequence = [
    #     assembly_digraph.assembly_digraph[path[i]][path[i + 1]][
    #         "num_connected_subgraphs"
    #     ]
    #     for i in range(len(path) - 1)
    # ]
    # print(f"{operations=}")
    # print(f"{num_conn_subgraphs_sequence=}")
    # print(f"{conn_subgraphs_sequence=}")
    # print(f"{joint_sequence=}")

    # PLP
    # result = optimize(
    #     assembly_digraph=assembly_digraph,
    #     return_model=True,
    #     num_phases=2,
    #     w_balanced=0.8,
    #     relative_gap=0.0,
    #     # hide_output=False,
    # )
    # print(result[1]["absolute_time_per_phase"])
