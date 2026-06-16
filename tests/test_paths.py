"""Tests for pycaalp/gapp/paths.py"""

import pytest
import networkx as nx
from pycaalp.gapp.paths import (
    calculate_num_simple_paths,
    calculate_sum_of_sh_path_weights,
    k_shortest_paths,
)


# ---------------------------------------------------------------------------
# Helpers — build a minimal mock AssemblyDigraph-like object
# ---------------------------------------------------------------------------

class _MockDigraph:
    """Minimal stand-in for AssemblyDigraph used by paths functions."""

    def __init__(self, assembly_digraph, num_joints):
        self.assembly_digraph = assembly_digraph
        self._num_joints = num_joints

    @property
    def get_num_layers(self):
        return self._num_joints + 1

    # needed by calculate_num_simple_paths
    class _graph:
        pass

    def _set_num_edges(self, n):
        class _G:
            def number_of_edges(self_inner):
                return n
        self.graph = _G()


def _make_linear_digraph(num_joints: int) -> nx.DiGraph:
    """Linear digraph: 0_1 → 1_1 → ... → N_1, weight = 1.0 per edge."""
    dg = nx.DiGraph()
    for i in range(num_joints):
        dg.add_edge(f"{i}_1", f"{i+1}_1", edge_weight=1.0, operation=(i, i + 1))
    return dg


def _make_branching_digraph(num_joints: int) -> nx.DiGraph:
    """
    Digraph where each inner layer has 2 nodes, giving 2 simple paths.
    0_1 → 1_1 → 2_1
        ↘ 1_2 ↗
    """
    dg = nx.DiGraph()
    dg.add_edge("0_1", "1_1", edge_weight=1.0, operation=(0, 1))
    dg.add_edge("0_1", "1_2", edge_weight=2.0, operation=(0, 1))
    dg.add_edge("1_1", "2_1", edge_weight=1.0, operation=(1, 2))
    dg.add_edge("1_2", "2_1", edge_weight=2.0, operation=(1, 2))
    return dg


# ---------------------------------------------------------------------------
# k_shortest_paths
# ---------------------------------------------------------------------------

def test_k_shortest_paths_linear():
    dg = _make_linear_digraph(3)
    paths = k_shortest_paths(dg, "0_1", "3_1", k=5, weight="edge_weight")
    assert len(paths) == 1
    assert paths[0] == ["0_1", "1_1", "2_1", "3_1"]


def test_k_shortest_paths_branching():
    dg = _make_branching_digraph(2)
    paths = k_shortest_paths(dg, "0_1", "2_1", k=5, weight="edge_weight")
    assert len(paths) == 2
    # Shorter (cheaper) path first
    assert paths[0] == ["0_1", "1_1", "2_1"]


def test_k_shortest_paths_k_limit():
    dg = _make_branching_digraph(2)
    paths = k_shortest_paths(dg, "0_1", "2_1", k=1, weight="edge_weight")
    assert len(paths) == 1


# ---------------------------------------------------------------------------
# calculate_sum_of_sh_path_weights
# ---------------------------------------------------------------------------

def test_sum_of_sh_path_weights_linear():
    num_joints = 3
    dg = _make_linear_digraph(num_joints)
    mock = _MockDigraph(dg, num_joints)
    total = calculate_sum_of_sh_path_weights(mock)
    # 3 edges each with weight 1.0
    assert total == pytest.approx(3.0)


def test_sum_of_sh_path_weights_picks_shortest():
    dg = _make_branching_digraph(2)
    mock = _MockDigraph(dg, 2)
    total = calculate_sum_of_sh_path_weights(mock)
    # Shortest path: 0_1→1_1→2_1, weights 1+1=2
    assert total == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# calculate_num_simple_paths  (Bug 4 fix: source was "0", now "0_1")
# ---------------------------------------------------------------------------

def test_calculate_num_simple_paths_linear():
    num_joints = 3
    dg = _make_linear_digraph(num_joints)
    mock = _MockDigraph(dg, num_joints)
    mock._set_num_edges(num_joints)
    count = calculate_num_simple_paths(mock)
    assert count == 1


def test_calculate_num_simple_paths_branching():
    dg = _make_branching_digraph(2)
    mock = _MockDigraph(dg, 2)
    mock._set_num_edges(2)
    count = calculate_num_simple_paths(mock)
    assert count == 2
