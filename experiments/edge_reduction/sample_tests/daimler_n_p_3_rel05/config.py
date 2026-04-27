"""Configuration parameters for multiple attributes run"""

config_data = {
    "assembly_fname": "data/assembly_1/assembly_1_2_tech_parts.json",
    "dfm_fname": "",  # "" if no DFM
    "res_fname": "experiments/edge_reduction/sample_tests/daimler_n_p_3_rel05/daimler_np3.pkl",
    # Defaults to 0.5 i.e., equal contribution of engineering constraints and time balancing
    "w_balanced": 0.8,
    # Either range [start, step, end] or absolute values e.g., [20, 30, 40, 50]
    "w_edge_reduction": [0, 20, 70],
    "num_phases": 3,
    "num_rep_runs": 3,
    "relative_gap": 0.05,  # MIP solver relative gap [0-1]
    "plots": [
        "objective_value",
        "total_time",
        "speedup",  # NOTICE: when there are not results for edge reduction=0 this will fail
        "avg_max_time",
    ],
}
