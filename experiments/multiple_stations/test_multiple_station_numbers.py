# pylint: disable=C0103, C0116
import numpy as np
import csv

from pycaalp.gapp.assembly_digraph import AssemblyDigraph
from pycaalp.time_balancing.model import run_mip


def run_variable_num_phase_mip(_num_phases_list, _assembly_digraph, w_bal):
    max_phase_times_list = []
    for num_phases in _num_phases_list:
        result, _ = run_mip(
            assembly_digraph=_assembly_digraph,
            num_phases=num_phases,
            w_balanced=w_bal,
            full_result_output=True,
        )
        max_phase_times_list.append(max(result["absolute_time_per_phase"].values()))

    return max_phase_times_list


if __name__ == "__main__":
    # Assembly digraph options
    file_name = "data/assembly_1/assembly_1_parts.json"
    results_fname = "experiments/multiple_stations/test_multiple_station_numbers.csv"

    # 1st create the assembly graph
    assembly_digraph = AssemblyDigraph(file_name=file_name)
    assembly_digraph.compute_assembly_digraph_complete()

    # MIP options
    num_phases_list = np.arange(1, assembly_digraph.graph.number_of_edges() + 1)
    w_balanced = 1.0

    max_phase_times = run_variable_num_phase_mip(
        num_phases_list, assembly_digraph, w_balanced
    )

    print(max_phase_times)

    with open(results_fname, "w", newline="", encoding="utf-8") as f:

        fieldnames = ["num_phases", "max_phase_times"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for num_p, max_p_time in zip(num_phases_list, max_phase_times):
            writer.writerow({"num_phases": num_p, "max_phase_times": max_p_time})
