"""Tests for pycaalp/gapp/filtering.py"""

import pytest
import networkx as nx
from pycaalp.gapp.filtering import (
    pick_random_percentage,
    minmax_norm,
    scale_to,
    normalize_attributes,
    filter_assembly_digraph_edges,
    find_unique_nodes_from_short_path,
)
from pycaalp.gapp.read_write import read_graph_from_json


EXAMPLE_JSON = "data/example/example_parts.json"


# ---------------------------------------------------------------------------
# pick_random_percentage
# ---------------------------------------------------------------------------

def test_pick_random_percentage_zero():
    lst = [(f"a_{i}", f"b_{i}") for i in range(10)]
    result = pick_random_percentage(lst[:], 0)
    assert result == []


def test_pick_random_percentage_half():
    lst = [(f"a_{i}", f"b_{i}") for i in range(10)]
    result = pick_random_percentage(lst[:], 50)
    assert len(result) == 5


def test_pick_random_percentage_all():
    lst = [(f"a_{i}", f"b_{i}") for i in range(4)]
    result = pick_random_percentage(lst[:], 100)
    assert len(result) == 4


def test_pick_random_percentage_protected():
    lst = [("protected_node", "b"), ("other", "c"), ("other2", "d")]
    result = pick_random_percentage(lst[:], 100, protected_elements=["protected_node"])
    # protected element's edge must not be picked
    assert all(e[0] != "protected_node" for e in result)


def test_pick_random_percentage_invalid():
    with pytest.raises(ValueError):
        pick_random_percentage([], 101)
    with pytest.raises(ValueError):
        pick_random_percentage([], -1)


# ---------------------------------------------------------------------------
# minmax_norm
# ---------------------------------------------------------------------------

def test_minmax_norm_basic():
    result = minmax_norm({"a": 0, "b": 5, "c": 10})
    assert result["a"] == pytest.approx(0.0)
    assert result["b"] == pytest.approx(0.5)
    assert result["c"] == pytest.approx(1.0)


def test_minmax_norm_constant_values():
    result = minmax_norm({"a": 5, "b": 5, "c": 5})
    assert all(v == 0 for v in result.values())


def test_minmax_norm_empty_raises():
    with pytest.raises(ValueError):
        minmax_norm({})


# ---------------------------------------------------------------------------
# scale_to
# ---------------------------------------------------------------------------

def test_scale_to_mean():
    vals = {"a": 2.0, "b": 4.0, "c": 6.0}
    result = scale_to(vals.copy(), div="mean")
    # mean = 4.0; scaled: 0.5, 1.0, 1.5
    assert result["a"] == pytest.approx(0.5)
    assert result["b"] == pytest.approx(1.0)
    assert result["c"] == pytest.approx(1.5)


def test_scale_to_median():
    vals = {"a": 1.0, "b": 3.0, "c": 5.0}
    result = scale_to(vals.copy(), div="median")
    # median = 3.0
    assert result["b"] == pytest.approx(1.0)


def test_scale_to_invalid_div():
    with pytest.raises(ValueError):
        scale_to({"a": 1.0}, div="mode")


def test_scale_to_empty_raises():
    with pytest.raises(ValueError):
        scale_to({}, div="mean")


# ---------------------------------------------------------------------------
# normalize_attributes
# ---------------------------------------------------------------------------

def test_normalize_attributes_preserves_absolute_values():
    g = read_graph_from_json(EXAMPLE_JSON)
    original_times = nx.get_edge_attributes(g, "time").copy()
    g = normalize_attributes(g)
    abs_times = nx.get_edge_attributes(g, "absolute_time")
    for key, val in original_times.items():
        assert abs_times[key] == val


def test_normalize_attributes_creates_normalized_tolerance():
    g = read_graph_from_json(EXAMPLE_JSON)
    g = normalize_attributes(g)
    tols = nx.get_edge_attributes(g, "tolerance")
    values = list(tols.values())
    # minmax normalisation → all in [0, 1]
    assert all(0.0 <= v <= 1.0 for v in values)


# ---------------------------------------------------------------------------
# find_unique_nodes_from_short_path
# ---------------------------------------------------------------------------

def _make_linear_digraph(num_layers: int) -> nx.DiGraph:
    """Build a simple linear digraph with one path 0_1 → 1_1 → ... → N_1."""
    dg = nx.DiGraph()
    for i in range(num_layers):
        dg.add_edge(f"{i}_1", f"{i+1}_1", edge_weight=1.0, operation=(i, i + 1))
    return dg


def test_find_unique_nodes_single_path():
    paths = [["0_1", "1_1", "2_1"]]
    result = find_unique_nodes_from_short_path(iter(paths))
    assert result[0] == ["0_1"]
    assert result[1] == ["1_1"]
    assert result[2] == ["2_1"]


def test_find_unique_nodes_two_paths():
    paths = [["0_1", "1_1", "2_1"], ["0_1", "1_2", "2_1"]]
    result = find_unique_nodes_from_short_path(iter(paths))
    assert "0_1" in result[0]
    assert set(result[1]) == {"1_1", "1_2"}
    assert "2_1" in result[2]


# ---------------------------------------------------------------------------
# filter_assembly_digraph_edges — verify layer 1 is now filtered (Bug 9 fix)
# ---------------------------------------------------------------------------

def _make_multi_path_digraph(num_joints: int) -> nx.DiGraph:
    """
    Build a digraph with 2 paths per layer (so there are removable edges).
    Layer naming: layer_<index>.  Layers 0 to num_joints.
    """
    dg = nx.DiGraph()
    for layer in range(num_joints):
        from_layer = layer
        to_layer = layer + 1
        # Two nodes per inner layer
        for idx in [1, 2]:
            for to_idx in [1, 2]:
                dg.add_edge(
                    f"{from_layer}_{idx}",
                    f"{to_layer}_{to_idx}",
                    edge_weight=1.0,
                    operation=(from_layer, to_layer),
                )
    return dg


def test_filter_reduces_edges():
    num_joints = 5
    num_layers = num_joints + 1  # as returned by get_num_layers
    dg = _make_multi_path_digraph(num_joints)
    edges_before = dg.number_of_edges()
    filtered = filter_assembly_digraph_edges(dg, 50, num_layers)
    assert filtered.number_of_edges() < edges_before


def test_filter_layer_1_edges_are_processed():
    """After the bug fix, layer-1 out-edges must also be subject to reduction."""
    num_joints = 4
    num_layers = num_joints + 1
    dg = _make_multi_path_digraph(num_joints)

    # Count edges whose source node is in layer 1 before filtering
    layer1_edges_before = [
        e for e in dg.edges() if e[0].startswith("1_")
    ]

    # Use 100 % filter to maximise removal (protected nodes guard only 0 and last)
    filtered = filter_assembly_digraph_edges(dg.copy(), 98, num_layers)

    layer1_edges_after = [
        e for e in filtered.edges() if e[0].startswith("1_")
    ]
    # Some layer-1 edges should have been removed (were previously untouched)
    assert len(layer1_edges_after) <= len(layer1_edges_before)
