# pylint: disable=C0103, C0116
import pytest
from pycaalp.gapp.assembly_digraph import AssemblyDigraph
from pycaalp.gapp.checks import all_cutset_combs, binomial_coeff


def create_assembly_digraph(**kwargs) -> AssemblyDigraph:
    """
    Accepted arguments:
    - file_name: Parts data file name (JSON)
    - dfm_file: DFM file name (JSON)
    - w_tech: Technology weight. Defaults to 0.3333.
    - w_hand: Handling weight. Defaults to 0.3333.
    - w_tol: Tolerance weight. Defaults to 0.3333.
    - reduction_percentage: Edge reduction percentage

    Returns:
        AssemblyDigraph: Fully constructed assembly digraph
    """
    _assem_digr = AssemblyDigraph(**kwargs)
    _assem_digr.compute_assembly_digraph_complete()
    return _assem_digr


@pytest.fixture
def default_kwargs():
    """Provides default arguments for the assembly digraph tests."""
    return {
        "file_name": "data/assembly_1/assembly_1_2_tech_parts.json",
        "w_tech": 1.0,
        "w_hand": 0.0,
        "w_tol": 0.0,
    }


def test_parallel_assembly(default_kwargs):
    """Test the parallel assembly configuration without the one assembly policy."""
    kwargs = default_kwargs.copy()
    kwargs["one_assembly_policy"] = False

    assembly_digraph = create_assembly_digraph(**kwargs)

    assert assembly_digraph.assembly_digraph.number_of_nodes() == 8192
    assert assembly_digraph.assembly_digraph.number_of_edges() == 53248


def test_one_assembly_policy(default_kwargs):
    """Test the configuration with the one assembly policy."""
    assembly_digraph = create_assembly_digraph(**default_kwargs)

    assert assembly_digraph.assembly_digraph.number_of_nodes() == 343
    assert assembly_digraph.assembly_digraph.number_of_edges() == 1105


def test_all_cutset_combs(default_kwargs):
    """Test the maximum cutset combinations for the given edges."""
    assembly_digraph = create_assembly_digraph(**default_kwargs)
    total_edges = assembly_digraph.graph.number_of_edges()

    max_edges = all_cutset_combs(total_edges)
    max_nodes = sum(binomial_coeff(total_edges, i) for i in range(total_edges + 1))

    assert max_nodes == 8192
    assert max_edges == 53248
