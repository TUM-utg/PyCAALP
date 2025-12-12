def results_in_ascending_order(results: dict) -> dict:
    """Change the appearing order i.e., the order they were added to the data structure,
    of the solver results
    """
    ascending_digraph_edges = list(results["operations"].keys())[::-1]
    ascending_phases = list(results["operations_per_phase"].keys())[::-1]
    new_results = {}
    for res_key, vals in results.items():
        new_results[res_key] = {}
        if isinstance(vals, float):
            new_results[res_key] = vals
        elif isinstance(vals, dict):
            if isinstance(vals, dict) and len(vals.keys()) == len(ascending_phases):
                # Per phase results
                for phase in ascending_phases:
                    new_results[res_key][phase] = vals[phase]
            elif len(vals.keys()) == len(ascending_digraph_edges):
                # Per operation results
                for edge in ascending_digraph_edges:
                    new_results[res_key][edge] = vals[edge]
        else:
            raise TypeError("results attributes should be of type float, or dict")

    return new_results
