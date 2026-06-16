"""Comparison tests for path-enumeration MIP (Option 1b) and k-path subgraph MIP (Option 2).

All three methods are solved once per session using module-scoped fixtures.
Individual tests assert against the cached results.

Assembly 1 (N=13 joints, P=3 phases) is used because it is small enough
for the full MIP to solve optimally in a few seconds.
"""

import pytest

from pycaalp.run import create_assembly_digraph, optimize
from pycaalp.time_balancing.path_mip import solve_by_path_mip
from pycaalp.time_balancing.subgraph_mip import build_kpath_subgraph, solve_by_subgraph_mip


ASSEMBLY_1_JSON = "data/assembly_1/assembly_1_2_tech_parts.json"
NUM_JOINTS = 13
NUM_PHASES = 3
W_BALANCED = 0.5
K = 50  # small k keeps fixtures fast; large enough to contain the optimal path

EXPECTED_RESULT_KEYS = {
    "operations", "technology", "handling", "tolerance", "time", "mass",
    "absolute_handling", "absolute_tolerance", "absolute_time", "absolute_mass",
    "phase", "operations_per_phase", "time_per_phase", "absolute_time_per_phase",
    "alpha",
}


# ---------------------------------------------------------------------------
# Fixtures — built once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def assembly1():
    return create_assembly_digraph(
        file_name=ASSEMBLY_1_JSON,
        w_tech=1.0, w_hand=0.0, w_tol=0.0, w_mass=0.0,
    )


@pytest.fixture(scope="module")
def full_mip_results(assembly1):
    results, ops_list = optimize(
        assembly_digraph=assembly1,
        num_phases=NUM_PHASES,
        w_balanced=W_BALANCED,
        full_result_output=True,
    )
    return results, ops_list


@pytest.fixture(scope="module")
def path_mip_results(assembly1):
    results, ops_list = solve_by_path_mip(
        assembly1,
        k=K,
        num_phases=NUM_PHASES,
        w_balanced=W_BALANCED,
        full_result_output=True,
    )
    return results, ops_list


@pytest.fixture(scope="module")
def subgraph_mip_results(assembly1):
    results, ops_list = solve_by_subgraph_mip(
        assembly1,
        k=K,
        num_phases=NUM_PHASES,
        w_balanced=W_BALANCED,
        full_result_output=True,
    )
    return results, ops_list


# ---------------------------------------------------------------------------
# Feasibility — all joints must be assigned
# ---------------------------------------------------------------------------

def test_full_mip_covers_all_joints(full_mip_results):
    _, ops_list = full_mip_results
    assert len([op for phase in ops_list for op in phase]) == NUM_JOINTS


def test_path_mip_covers_all_joints(path_mip_results):
    _, ops_list = path_mip_results
    assert len([op for phase in ops_list for op in phase]) == NUM_JOINTS


def test_subgraph_mip_covers_all_joints(subgraph_mip_results):
    _, ops_list = subgraph_mip_results
    assert len([op for phase in ops_list for op in phase]) == NUM_JOINTS


# ---------------------------------------------------------------------------
# Result dict structure
# ---------------------------------------------------------------------------

def test_full_mip_result_keys(full_mip_results):
    results, _ = full_mip_results
    assert set(results.keys()) == EXPECTED_RESULT_KEYS


def test_path_mip_result_keys(path_mip_results):
    results, _ = path_mip_results
    assert set(results.keys()) == EXPECTED_RESULT_KEYS


def test_subgraph_mip_result_keys(subgraph_mip_results):
    results, _ = subgraph_mip_results
    assert set(results.keys()) == EXPECTED_RESULT_KEYS


# ---------------------------------------------------------------------------
# Phase count and non-empty phases
# ---------------------------------------------------------------------------

def test_full_mip_phase_count(full_mip_results):
    _, ops_list = full_mip_results
    assert len(ops_list) == NUM_PHASES
    assert all(len(phase) > 0 for phase in ops_list)


def test_path_mip_phase_count(path_mip_results):
    _, ops_list = path_mip_results
    assert len(ops_list) == NUM_PHASES
    assert all(len(phase) > 0 for phase in ops_list)


def test_subgraph_mip_phase_count(subgraph_mip_results):
    _, ops_list = subgraph_mip_results
    assert len(ops_list) == NUM_PHASES
    assert all(len(phase) > 0 for phase in ops_list)


# ---------------------------------------------------------------------------
# No joint appears in more than one phase
# ---------------------------------------------------------------------------

def test_full_mip_no_duplicate_joints(full_mip_results):
    _, ops_list = full_mip_results
    all_ops = [op for phase in ops_list for op in phase]
    assert len(all_ops) == len(set(all_ops))


def test_path_mip_no_duplicate_joints(path_mip_results):
    _, ops_list = path_mip_results
    all_ops = [op for phase in ops_list for op in phase]
    assert len(all_ops) == len(set(all_ops))


def test_subgraph_mip_no_duplicate_joints(subgraph_mip_results):
    _, ops_list = subgraph_mip_results
    all_ops = [op for phase in ops_list for op in phase]
    assert len(all_ops) == len(set(all_ops))


# ---------------------------------------------------------------------------
# Alpha consistency — alpha must equal max phase time
# ---------------------------------------------------------------------------

def test_path_mip_alpha_equals_max_phase_time(path_mip_results):
    results, _ = path_mip_results
    assert results["alpha"] == pytest.approx(
        max(results["time_per_phase"].values()), rel=1e-3
    )


def test_subgraph_mip_alpha_equals_max_phase_time(subgraph_mip_results):
    results, _ = subgraph_mip_results
    assert results["alpha"] == pytest.approx(
        max(results["time_per_phase"].values()), rel=1e-3
    )


# ---------------------------------------------------------------------------
# Subgraph structure
# ---------------------------------------------------------------------------

def test_kpath_subgraph_smaller_than_full(assembly1):
    full_edges = assembly1.assembly_digraph.number_of_edges()
    subgraph = build_kpath_subgraph(assembly1, k=K)
    assert subgraph.number_of_edges() < full_edges


def test_kpath_subgraph_has_source_and_sink(assembly1):
    subgraph = build_kpath_subgraph(assembly1, k=K)
    assert "0_1" in subgraph.nodes()
    assert f"{NUM_JOINTS}_1" in subgraph.nodes()


def test_kpath_subgraph_grows_monotonically(assembly1):
    sg_small = build_kpath_subgraph(assembly1, k=10)
    sg_large = build_kpath_subgraph(assembly1, k=K)
    assert sg_large.number_of_edges() >= sg_small.number_of_edges()


# ---------------------------------------------------------------------------
# Solution quality — new methods must stay within a reasonable bound of the
# full MIP optimum (measured on absolute phase time in seconds)
# ---------------------------------------------------------------------------

def test_path_mip_alpha_within_25_percent_of_full(full_mip_results, path_mip_results):
    full_alpha = max(full_mip_results[0]["absolute_time_per_phase"].values())
    path_alpha = max(path_mip_results[0]["absolute_time_per_phase"].values())
    assert path_alpha <= full_alpha * 1.25


def test_subgraph_mip_alpha_within_5_percent_of_full(full_mip_results, subgraph_mip_results):
    full_alpha = max(full_mip_results[0]["absolute_time_per_phase"].values())
    sub_alpha = max(subgraph_mip_results[0]["absolute_time_per_phase"].values())
    assert sub_alpha <= full_alpha * 1.05
