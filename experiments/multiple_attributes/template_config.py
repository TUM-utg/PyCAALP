"""TEMPLATE Configuration parameters for multiple attributes run"""

config_data = {
    "assembly_fname": "data/AssemblyX/AssebmlyX_parts.json",  # File name with the assembly parts
    "dfm_fname": "data/AssemblyX/AssemblyX_dfm.json",  # "" if no DFM
    "res_fname": "multiple_attributes/results/res.pkl",  # Directory for the resulted pkl file
    "attribute_combination": {
        # The length of w_balanced list might be of different length compared to the other 3
        "w_tech": [1.0, 0.0, 0.0, 0.0],
        "w_hand": [0.0, 1.0, 0.0, 0.0],
        "w_tol": [0.0, 0.0, 1.0, 0.0],
        "w_mass": [0.0, 0.0, 0.0, 1.0],
        "w_balanced": [0.1, 0.5, 1.0],
    },
    "num_phases": 3,
    "plots": [
        "attribute_changes",  # This won't run with len(w_balanced)>3
        "attribute_development",  # This won't run with len(w_balanced)>3
        "time_per_phase",  # This won't run with len(w_balanced)>3
        "time_vs_wb",
    ],
    "reduction_percentage": 0,  # Edge reduction phase percentage
    "relative_gap": 0.00,  # MIP solver relative gap [0-1]
}
