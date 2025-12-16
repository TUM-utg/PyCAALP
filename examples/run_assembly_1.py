"""Solve an assembly and line planning problem for assembly 1
The assembly consists of 2 different technologies/manufacturing processes.
The solution prioritizes the minimization of technology changes (MU_TECH=1.0)
"""

from pycaalp.run import create_assembly_digraph, optimize

if __name__ == "__main__":
    # Assembly digraph options
    FILE_NAME = "data/assembly_1/assembly_1_2_tech_parts.json"

    # engineering constraint constants (mu)
    MU_TECH = 1.0
    MU_HAND = 0.0
    MU_TOL = 0.0

    # MIP options
    NUM_PHASES = 3
    LAMBDA_TIME_BAL = 0.5  # time balancing coefficient

    # Generate the weighted assembly directed graph
    assembly_digraph = create_assembly_digraph(
        file_name=FILE_NAME,
        w_tech=MU_TECH,
        w_hand=MU_HAND,
        w_tol=MU_TOL,
    )

    # Solve the phase time balancing problem
    result, best_path = optimize(
        assembly_digraph=assembly_digraph,
        num_phases=NUM_PHASES,
        w_balanced=LAMBDA_TIME_BAL,
        full_result_output=True,
        hide_output=False,
    )

    # Print the resulted assembly sequence and the phase per time
    print(best_path)
    print(f"{result["absolute_time_per_phase"]=}")
