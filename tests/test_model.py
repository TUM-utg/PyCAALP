"""Tests for pycaalp/time_balancing/model.py (MIP solver)."""

import pytest
from pycaalp.gapp.assembly_digraph import AssemblyDigraph
from pycaalp.time_balancing.model import run_mip


EXAMPLE_JSON = "data/example/example_parts.json"
ASSEMBLY_1_JSON = "data/assembly_1/assembly_1_2_tech_parts.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def example_digraph():
    ad = AssemblyDigraph(file_name=EXAMPLE_JSON)
    ad.compute_assembly_digraph_complete()
    return ad


@pytest.fixture(scope="module")
def assembly1_digraph():
    ad = AssemblyDigraph(
        file_name=ASSEMBLY_1_JSON,
        w_tech=1.0, w_hand=0.0, w_tol=0.0, w_mass=0.0,
    )
    ad.compute_assembly_digraph_complete()
    return ad


# ---------------------------------------------------------------------------
# run_mip — basic smoke tests
# ---------------------------------------------------------------------------

def test_run_mip_no_input_raises():
    with pytest.raises(ValueError):
        run_mip()


def test_run_mip_returns_operations_list(example_digraph):
    ops_list = run_mip(
        assembly_digraph=example_digraph,
        num_phases=2,
        w_balanced=0.5,
        hide_output=True,
    )
    assert isinstance(ops_list, list)
    assert len(ops_list) == 2


def test_run_mip_all_joints_assigned(example_digraph):
    """Every joint in the assembly must appear exactly once across all phases."""
    ops_list = run_mip(
        assembly_digraph=example_digraph,
        num_phases=2,
        w_balanced=0.5,
        hide_output=True,
    )
    all_ops = [op for phase in ops_list for op in phase]
    num_joints = example_digraph.graph.number_of_edges()
    assert len(all_ops) == num_joints


def test_run_mip_full_result_output(example_digraph):
    results, ops_list = run_mip(
        assembly_digraph=example_digraph,
        num_phases=2,
        w_balanced=0.5,
        full_result_output=True,
        hide_output=True,
    )
    assert isinstance(results, dict)
    assert "operations" in results
    assert "phase" in results
    assert "alpha" in results
    assert "time_per_phase" in results
    assert "absolute_time_per_phase" in results
    assert "operations_per_phase" in results


def test_run_mip_phase_count_matches_num_phases(example_digraph):
    results, _ = run_mip(
        assembly_digraph=example_digraph,
        num_phases=2,
        w_balanced=0.5,
        full_result_output=True,
        hide_output=True,
    )
    phases_used = set(results["phase"].values())
    assert phases_used.issubset({0, 1})


# ---------------------------------------------------------------------------
# Phase assignment correctness — Bug 6 fix
# ---------------------------------------------------------------------------

def test_phase_assignment_is_consistent(example_digraph):
    """Each operation's phase must match a y-variable assignment for its layer."""
    results, _ = run_mip(
        assembly_digraph=example_digraph,
        num_phases=2,
        w_balanced=0.5,
        full_result_output=True,
        hide_output=True,
    )
    # Every edge in results["operations"] must have a phase in results["phase"]
    for ed_str in results["operations"]:
        assert ed_str in results["phase"]
        assert results["phase"][ed_str] in {0, 1}


def test_time_per_phase_sums_correctly(example_digraph):
    """time_per_phase[p] must equal the sum of time[op] for all ops in phase p."""
    results, _ = run_mip(
        assembly_digraph=example_digraph,
        num_phases=2,
        w_balanced=0.5,
        full_result_output=True,
        hide_output=True,
    )
    import networkx as nx
    time_weights = nx.get_edge_attributes(example_digraph.graph, "time")
    recomputed = {}
    for ed_str, oper in results["operations"].items():
        ph = results["phase"][ed_str]
        recomputed[ph] = recomputed.get(ph, 0) + time_weights[oper]
    for ph, total in recomputed.items():
        assert results["time_per_phase"][ph] == pytest.approx(total, rel=1e-6)


# ---------------------------------------------------------------------------
# Assembly 1 — larger integration test
# ---------------------------------------------------------------------------

def test_assembly1_mip_assigns_all_joints(assembly1_digraph):
    ops_list = run_mip(
        assembly_digraph=assembly1_digraph,
        num_phases=3,
        w_balanced=0.5,
        hide_output=True,
    )
    all_ops = [op for phase in ops_list for op in phase]
    assert len(all_ops) == assembly1_digraph.graph.number_of_edges()


def test_assembly1_three_phases_non_empty(assembly1_digraph):
    ops_list = run_mip(
        assembly_digraph=assembly1_digraph,
        num_phases=3,
        w_balanced=0.5,
        hide_output=True,
    )
    assert len(ops_list) == 3
    # Each phase must have at least one operation
    for phase in ops_list:
        assert len(phase) >= 1
