"""Configuration parameters for multiple attributes run"""

config_data = {
    "assembly_fname": "data/assembly_1/assembly_1_parts.json",
    "dfm_fname": "",
    "res_fname": "experiments/edge_reduction/sample_tests/assembly_1_n_p_3/assembly_1.pkl",
    # Defaults to 0.5 i.e., equal contribution of engineering constraints and time balancing
    "w_balanced": 0.8,  # Will be overridden when run_sweep.sh runs
    # Either range [start, step, end] or absolute values e.g., [20, 30, 40, 50]
    "w_edge_reduction": [0, 5, 70],
    "num_phases": 2,
    "num_rep_runs": 5,
    "relative_gap": 0.00,  # MIP solver relative gap [0-1]
    "plots": [
        "total_time",
        "speedup",  # NOTICE: when there are not results for edge reduction=0 this will fail
        "avg_max_time",
    ],
}
