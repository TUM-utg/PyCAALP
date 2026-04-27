# Ideas to avoid random edge reduction

## Extension of edge weights

```python
>>> import numpy as np
...
... times = np.array([108, 244, 244, 94, 242, 100, 110, 166, 184, 235, 94, 166])
... t_mean = times.mean()   # ≈ 165.6 mm
... t_std  = times.std()    # ≈ 62.1 mm
...
... # Score each joint
... scores = np.abs(times - t_mean) / t_std
...
>>> t_mean
np.float64(165.58333333333334)
>>> t_std
np.float64(60.725417980355545)
>>> scores
array([0.9482575 , 1.29133186, 1.29133186, 1.17880347, 1.25839672,
       1.07999806, 0.91532237, 0.00686149, 0.30327773, 1.14312374,
       1.17880347, 0.00686149])
>>> num_phases = 3
>>> optimal_phase_time = np.sum(times) / num_phases
>>> optimal_phase_time
np.float64(662.3333333333334)
>>> np.sum(times)
np.int64(1987)
>>> phase_score = times/optimal_phase_time
>>> phase_score
array([0.16305989, 0.36839456, 0.36839456, 0.1419225 , 0.36537494,
       0.15098138, 0.16607952, 0.25062909, 0.27780574, 0.35480624,
       0.1419225 , 0.25062909])
>>> num_phases = 4
>>> optimal_phase_time = np.sum(times) / num_phases
>>> optimal_phase_time
np.float64(496.75)
>>> phase_score = times/optimal_phase_time
>>> phase_score
array([0.21741319, 0.49119275, 0.49119275, 0.18922999, 0.48716658,
       0.20130851, 0.22143936, 0.33417212, 0.37040765, 0.47307499,
       0.18922999, 0.33417212])
>>> num_phases = 5
>>> optimal_phase_time = np.sum(times) / num_phases
>>> phase_score = times/optimal_phase_time
>>> phase_score
array([0.27176648, 0.61399094, 0.61399094, 0.23653749, 0.60895823,
       0.25163563, 0.27679919, 0.41771515, 0.46300956, 0.59134373,
       0.23653749, 0.41771515])
>>>
```

## Calculation of simple paths

```python
def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

def find_all_shortest_paths(
    assembly_digraph: nx.DiGraph, main_graph_num_edges: int, method: str = None
) -> dict:
    if not method:
        method = "dijkstra"
    # all_shortest_paths = nx.all_shortest_paths(
    #     assembly_digraph,
    #     source="0_1",
    #     target=f"{main_graph_num_edges}_1",
    #     weight="edge_weight",
    #     method=method,
    # )
    all_shortest_paths = k_shortest_paths(
        assembly_digraph,
        source="0_1",
        target=f"{main_graph_num_edges}_1",
        k=2000,
        weight="edge_weight",
    )
    return find_unique_nodes_from_short_path(all_shortest_paths)
```
