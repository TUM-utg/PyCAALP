"""Tests for pycaalp/gapp/checks.py"""

import pytest
import networkx as nx
from pycaalp.gapp.checks import (
    has_technology_changes,
    count_num_different_technologies,
    create_tech_list_with_occurrences,
    find_min_diff,
    check_technology_changes,
    check_common_node,
    get_num_connected_subgraphs,
    check_one_assembly_policy,
    binomial_coeff,
    all_cutset_combs,
)


# ---------------------------------------------------------------------------
# has_technology_changes
# ---------------------------------------------------------------------------

def test_has_technology_changes_single():
    assert has_technology_changes(["MAG", "MAG", "MAG"]) == 1


def test_has_technology_changes_two():
    assert has_technology_changes(["MAG", "MAG2", "MAG"]) == 2


def test_has_technology_changes_empty():
    assert has_technology_changes([]) == 0


# ---------------------------------------------------------------------------
# count_num_different_technologies
# ---------------------------------------------------------------------------

def test_count_different_tech_one():
    assert count_num_different_technologies(["MAG", "MAG", "MAG"]) == 0


def test_count_different_tech_two():
    assert count_num_different_technologies(["MAG", "MAG2", "MAG"]) == 1


# ---------------------------------------------------------------------------
# create_tech_list_with_occurrences
# ---------------------------------------------------------------------------

def test_tech_occurrences_basic():
    result = create_tech_list_with_occurrences(["MAG", "MAG", "MAG2"])
    assert result == {"MAG": 2, "MAG2": 1}


def test_tech_occurrences_single():
    result = create_tech_list_with_occurrences(["MAG"])
    assert result == {"MAG": 1}


# ---------------------------------------------------------------------------
# find_min_diff (Bug 8 fix: was initialised to -inf, now +inf)
# ---------------------------------------------------------------------------

def test_find_min_diff_two_technologies():
    # {"MAG": 5, "MAG2": 3} → max_key="MAG", diff = 5-3 = 2
    result = find_min_diff({"MAG": 5, "MAG2": 3})
    assert result == 2


def test_find_min_diff_single_technology():
    # Only one tech → loop never runs → should return 0
    result = find_min_diff({"MAG": 5})
    assert result == 0


def test_find_min_diff_three_technologies():
    # {"MAG": 10, "MAG2": 4, "TIG": 7} → max_key="MAG"
    # diffs: 10-4=6, 10-7=3 → min = 3
    result = find_min_diff({"MAG": 10, "MAG2": 4, "TIG": 7})
    assert result == 3


def test_find_min_diff_equal_counts():
    result = find_min_diff({"MAG": 5, "MAG2": 5})
    assert result == 0


# ---------------------------------------------------------------------------
# check_technology_changes
# ---------------------------------------------------------------------------

def test_check_technology_changes_new_type():
    # Before: {"MAG": 2} → min_diff=0 (single tech).
    # After adding "MAG2": {"MAG": 2, "MAG2": 1} → min_diff=1.
    # 1 > 0, so gap did NOT decrease → returns 0.
    result = check_technology_changes({"e1": "MAG", "e2": "MAG"}, "MAG2")
    assert result == 0


def test_check_technology_changes_reduces_gap():
    # Before: {"MAG": 3, "MAG2": 1} → min_diff = 3-1 = 2.
    # Adding "MAG2": {"MAG": 3, "MAG2": 2} → min_diff = 3-2 = 1.
    # 1 < 2 → gap decreased → returns 1.
    result = check_technology_changes({"e1": "MAG", "e2": "MAG", "e3": "MAG", "e4": "MAG2"}, "MAG2")
    assert result == 1


def test_check_technology_changes_same_type():
    # Adding another MAG when balanced: {MAG:2, MAG2:2} → min_diff=0
    # After: {MAG:3, MAG2:2} → min_diff=1, 1 > 0 → returns 0
    result = check_technology_changes({"e1": "MAG", "e2": "MAG2", "e3": "MAG", "e4": "MAG2"}, "MAG")
    assert result == 0


# ---------------------------------------------------------------------------
# check_common_node
# ---------------------------------------------------------------------------

def test_check_common_node_true():
    assert check_common_node([(0, 1), (1, 2)]) is True


def test_check_common_node_false():
    assert check_common_node([(0, 1), (2, 3)]) is False


def test_check_common_node_single_edge():
    assert check_common_node([(0, 1)]) is True  # shares with itself


# ---------------------------------------------------------------------------
# get_num_connected_subgraphs
# ---------------------------------------------------------------------------

def _path_graph(n):
    return nx.path_graph(n)


def test_get_num_connected_subgraphs_connected():
    g = _path_graph(4)  # 0-1-2-3 fully connected
    assert get_num_connected_subgraphs(g) == 1


def test_get_num_connected_subgraphs_isolated_node():
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2])
    g.add_edge(0, 1)
    # node 2 is isolated → only component of size >1 is {0,1}
    assert get_num_connected_subgraphs(g) == 1


def test_get_num_connected_subgraphs_two_components():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (2, 3)])
    assert get_num_connected_subgraphs(g) == 2


# ---------------------------------------------------------------------------
# check_one_assembly_policy
# ---------------------------------------------------------------------------

def test_one_assembly_policy_disabled():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (2, 3)])
    assert check_one_assembly_policy(g, one_assembly_policy=False) is True


def test_one_assembly_policy_single_component():
    g = _path_graph(4)
    assert check_one_assembly_policy(g, one_assembly_policy=True) is True


def test_one_assembly_policy_two_components_fails():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (2, 3)])
    assert check_one_assembly_policy(g, one_assembly_policy=True) is False


def test_one_assembly_policy_parallel_allowed():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (2, 3)])
    assert check_one_assembly_policy(g, one_assembly_policy=True, num_par_ass=2) is True


# ---------------------------------------------------------------------------
# binomial_coeff
# ---------------------------------------------------------------------------

def test_binomial_coeff_basic():
    assert binomial_coeff(5, 2) == 10
    assert binomial_coeff(4, 0) == 1
    assert binomial_coeff(4, 4) == 1


def test_binomial_coeff_invalid():
    with pytest.raises(ValueError):
        binomial_coeff(3, 5)
    with pytest.raises(ValueError):
        binomial_coeff(3, -1)


# ---------------------------------------------------------------------------
# all_cutset_combs (smoke test)
# ---------------------------------------------------------------------------

def test_all_cutset_combs_two_edges(capsys):
    # 2 joints: max edges = 2 * 2^(2-1) = 4
    result = all_cutset_combs(2)
    assert result == 4
