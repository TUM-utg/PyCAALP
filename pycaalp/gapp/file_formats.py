"""All operations with files for assembly_digraph class"""

import pickle
import json


def assembly_digraph_to_dict(
    assembly_digraph: "AssemblyDigraph", add_shortets_path: bool = False
) -> dict:
    """
    Copy assembly_digraph class attributes to a dictionary. Avoid dependencies when
    saving to a pickle format.

    Args:
        assembly_digraph: A class with all assembly digraph's attributes

    Returns:
        A dictionary with only the important data from the assembly_digraph Class.
    """
    assembly_digraph_dict = {}
    assembly_digraph_dict["assembly_digraph"] = assembly_digraph.assembly_digraph
    assembly_digraph_dict["main_graph"] = assembly_digraph.graph

    if add_shortets_path:
        assembly_digraph_dict["shortest_path"] = assembly_digraph.shortest_path
    return assembly_digraph_dict


def save_to_pkl(value: any, file_name: str) -> any:
    """
    Saves a value to a pickle file.

    Args:
        value: The value to be saved in pickle file.
                Value could be anything, i.e., class, dict ...
        file_name: The path to the pickle file.
    """
    with open(file_name, "wb") as _f:
        pickle.dump(value, _f)


def load_pkl(file_name: str) -> any:
    """
    Loads a pickle file and returns its container.

    Args:
        file_name: The path to the pickle file.

    Returns:
        Pickle container.
    """
    with open(file_name, "rb") as _f:
        pkl_val = pickle.load(_f)
    return pkl_val


def filter_trafos_from_pkl(pkl_file: str, new_pkl_file: str = None) -> None:
    """Remove the 'trafo' attribute from the edges of the main graph in the
    pickle file.

    Args:
        pkl_file: The path to the pickle file.
        new_pkl_file: The path to the new pickle file. If None, the
            original file will be overwritten.
    """
    if new_pkl_file is None:
        new_pkl_file = pkl_file
    gapp_dict = load_pkl(pkl_file)

    if "trafo" not in gapp_dict["main_graph"].edges[0, 1]:
        print("No 'trafo' attribute found in the edges.")
        return

    for _, _, attr in gapp_dict["main_graph"].edges(data=True):
        del attr["trafo"]

    save_to_pkl(gapp_dict, new_pkl_file)


def save_all_res_to_json(res: dict, res_fname: str) -> None:
    """Save time balancing results in json format.
    NOTE: for multiple runs scenario.

    Args:
        res: All the results per weight combination per operation.
        res_fname: JSON filename.
    """
    new_res = {}
    for key, val in res.items():
        val_new = {}
        if isinstance(val, dict):
            for attr, attrvals in val.items():
                if isinstance(val[attr], dict):
                    if isinstance(list(attrvals.keys())[0], tuple):
                        new_attrvals = {}
                        for attrvals_key, attrvals_vals in attrvals.items():
                            new_attrvals[str(attrvals_key)] = attrvals_vals
                            attrvals = new_attrvals
                val_new[attr] = attrvals
        else:
            val_new["total_time"] = val[0]
            val_new["objective_value"] = val[1]
        # Add the new attribute with all tuples set to str
        new_res[str(key)] = val_new
    # Save to the json file
    with open(res_fname, "w", encoding="utf-8") as f:
        json.dump(new_res, f, indent=4)
