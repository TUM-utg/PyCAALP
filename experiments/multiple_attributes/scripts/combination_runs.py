from loguru import logger
from pycaalp.gapp.assembly_digraph import AssemblyDigraph
from pycaalp.time_balancing.model import run_mip


def run_assembly_line_planning(
    w_tech: float,
    w_hand: float,
    w_tol: float,
    w_mass: float,
    w_balanced: float,
    test_assebmly_fname,
    num_phases,
    rel_gap,
    red_perc,
    dfm_fname: str | None = None,
):
    """Run a full assembly and line planning problem with specific weights

    Args: TODO update
            assembly_fname: assembly parts directory
            w_tech: technology weight constant
            w_hand: handling weight constant
            w_tol: tolerance weight constant
            w_mass: mass weight constant
            w_balanced: time balancing weight constant

    Returns: Full operations results
    """
    assembly_digraph = AssemblyDigraph(
        file_name=test_assebmly_fname,
        w_tech=w_tech,
        w_hand=w_hand,
        w_tol=w_tol,
        w_mass=w_mass,
        reduction_percentage=red_perc,
        dfm_file=dfm_fname,
    )
    assembly_digraph.compute_assembly_digraph_complete()
    results, _ = run_mip(
        assembly_digraph=assembly_digraph,
        num_phases=num_phases,
        w_balanced=w_balanced,
        relative_gap=rel_gap,
        full_result_output=True,
    )
    return results


def run_full_run_attr_change(
    line_bal_comb: dict,
    assemb_attr_comb: dict,
    test_assembly_fname: str,
    dfm_fname: str | None = None,
    num_phases: int = 3,
    rel_gap: float = 0.0,
    red_perc: int = 0,
):
    """Given assembly and line planning coefficient combinations, run all combinations

    Args: TODO: update
        attr_name: attribute to return.
        line_bal_comb: line balancing coefficient combinations.
        assemb_attr_comb: assembly balancing coefficient combinations.

    Returns:
        Time balancing results for the attribute name i.e., attribute name per operation
    """
    res = {}
    for w_bal in line_bal_comb["w_balanced"]:
        for w_tech, w_hand, w_tol, w_mass in zip(
            assemb_attr_comb["w_tech"],
            assemb_attr_comb["w_hand"],
            assemb_attr_comb["w_tol"],
            assemb_attr_comb["w_mass"],
        ):
            logger.info(
                f"Running test with: {w_tech=}, {w_hand=}, {w_tol=}, {w_mass=}, {w_bal=}"
            )
            res[(w_tech, w_hand, w_tol, w_mass, w_bal)] = run_assembly_line_planning(
                w_tech=w_tech,
                w_hand=w_hand,
                w_tol=w_tol,
                w_mass=w_mass,
                w_balanced=w_bal,
                test_assebmly_fname=test_assembly_fname,
                num_phases=num_phases,
                rel_gap=rel_gap,
                red_perc=red_perc,
                dfm_fname=dfm_fname,
            )

    return res
