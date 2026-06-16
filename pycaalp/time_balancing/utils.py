def results_in_ascending_order(results: dict) -> dict:
    """Change the appearing order i.e., the order they were added to the data structure,
    of the solver results.

    Per-phase dicts have integer keys; per-operation dicts have tuple keys.
    Using key types avoids the ambiguity that arises when num_phases == num_operations.
    """
    ascending_digraph_edges = list(results["operations"].keys())[::-1]
    ascending_phases = list(results["operations_per_phase"].keys())[::-1]
    new_results = {}
    for res_key, vals in results.items():
        new_results[res_key] = {}
        if isinstance(vals, float):
            new_results[res_key] = vals
        elif isinstance(vals, dict):
            if not vals:
                # Empty dict — preserve as-is
                pass
            elif isinstance(next(iter(vals)), int):
                # Per-phase dict (phase index keys are ints)
                for phase in ascending_phases:
                    if phase in vals:
                        new_results[res_key][phase] = vals[phase]
            else:
                # Per-operation dict (digraph-edge tuple keys)
                for edge in ascending_digraph_edges:
                    new_results[res_key][edge] = vals[edge]
        else:
            raise TypeError("results attributes should be of type float, or dict")

    return new_results
