"""TEMPLATE Configuration parameters for multiple attributes run"""

config_data = {
    "assembly_fname": "data/AssemblyX/AssebmlyX_parts.json",  # File name with the assembly parts
    "dfm_fname": "data/AssemblyX/AssemblyX_dfm.json",  # "" if no DFM
    "res_fname": "edge_reduction/results/res.pkl",  # Directory for the resulted pkl file
    "w_tech": 0.5,  # Defaults to 0.3333
    "w_hand": 0.25,  # Defaults to 0.3333
    "w_tol": 0.25,  # Defaults to 0.3333
    # Defaults to 0.5 i.e., equal contribution of engineering constraints and time balancing
    "w_balanced": 0.3,
    # Either range [start, step, end] or absolute values e.g., [20, 30, 40, 50]
    "w_edge_reduction": [0, 5, 50],
    "num_phases": 3,
    "num_rep_runs": 3,
    "relative_gap": 0.03,  # MIP solver relative gap [0-1]
    "plots": [
        "objective_value",
        "total_time",
        "speedup",  # NOTICE: when there are not results for edge reduction=0 this will fail
        "avg_max_time",
    ],
}
