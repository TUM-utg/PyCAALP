"""Tests for pycaalp/gapp/assembly_digraph.py — AssemblyDigraph class."""

import os
import pickle
import tempfile
import pytest
import networkx as nx

from pycaalp.gapp.assembly_digraph import AssemblyDigraph


EXAMPLE_JSON = "data/example/example_parts.json"
ASSEMBLY_1_JSON = "data/assembly_1/assembly_1_2_tech_parts.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def example_digraph():
    """AssemblyDigraph built from the tiny 3-part example assembly."""
    ad = AssemblyDigraph(file_name=EXAMPLE_JSON)
    ad.compute_assembly_digraph_complete()
    return ad


@pytest.fixture(scope="module")
def assembly1_digraph():
    """AssemblyDigraph built from the 13-joint assembly 1."""
    ad = AssemblyDigraph(
        file_name=ASSEMBLY_1_JSON,
        w_tech=1.0,
        w_hand=0.0,
        w_tol=0.0,
        w_mass=0.0,
    )
    ad.compute_assembly_digraph_complete()
    return ad


# ---------------------------------------------------------------------------
# __init__ — Bug 1 fixes
# ---------------------------------------------------------------------------

def test_init_with_valid_json():
    ad = AssemblyDigraph(file_name=EXAMPLE_JSON)
    assert ad.graph is not None


def test_init_file_name_none_raises():
    """Passing file_name=None without a graph must raise ValueError, not AttributeError."""
    with pytest.raises(ValueError, match="Provide either"):
        AssemblyDigraph(file_name=None)


def test_init_non_json_raises():
    with pytest.raises(ValueError, match=".json"):
        AssemblyDigraph(file_name="data/assembly_1/assembly_1_2_tech_parts.csv")


def test_init_non_string_raises():
    with pytest.raises(ValueError, match="string"):
        AssemblyDigraph(file_name=123)


def test_init_with_prebuilt_graph():
    """When a graph is passed directly (Bug 1: graph param was silently ignored)."""
    g = nx.Graph()
    g.add_node("P1", weight=1.0, handling=1)
    g.add_node("P2", weight=0.5, handling=2)
    g.add_edge(
        "P1", "P2",
        name="j1", technology="WELD", time=100, tolerance=1,
        handling=2, mass=1.0,
    )
    ad = AssemblyDigraph(graph=g)
    assert ad.graph.number_of_nodes() == 2
    assert ad.graph.number_of_edges() == 1


def test_init_weights_must_sum_to_one():
    with pytest.raises(AssertionError):
        AssemblyDigraph(
            file_name=EXAMPLE_JSON,
            w_tech=0.5, w_hand=0.5, w_tol=0.5, w_mass=0.5,
        )


def test_init_weights_must_be_non_negative():
    with pytest.raises(AssertionError):
        AssemblyDigraph(
            file_name=EXAMPLE_JSON,
            w_tech=-0.5, w_hand=0.5, w_tol=0.5, w_mass=0.5,
        )


# ---------------------------------------------------------------------------
# compute_assembly_digraph_complete
# ---------------------------------------------------------------------------

def test_digraph_has_nodes(example_digraph):
    assert example_digraph.assembly_digraph.number_of_nodes() > 0


def test_digraph_has_edges(example_digraph):
    assert example_digraph.assembly_digraph.number_of_edges() > 0


def test_digraph_start_node_exists(example_digraph):
    assert "0_1" in example_digraph.assembly_digraph.nodes()


def test_digraph_end_node_exists(example_digraph):
    num_joints = example_digraph.graph.number_of_edges()
    end_node = f"{num_joints}_1"
    assert end_node in example_digraph.assembly_digraph.nodes()


def test_digraph_is_dag(example_digraph):
    assert nx.is_directed_acyclic_graph(example_digraph.assembly_digraph)


def test_digraph_path_exists(example_digraph):
    num_joints = example_digraph.graph.number_of_edges()
    path = nx.shortest_path(
        example_digraph.assembly_digraph,
        source="0_1",
        target=f"{num_joints}_1",
    )
    assert len(path) == num_joints + 1


def test_sum_of_sh_path_weights_set(example_digraph):
    assert example_digraph.sum_of_sh_path_weights is not None
    assert example_digraph.sum_of_sh_path_weights > 0


def test_assembly1_node_count(assembly1_digraph):
    assert assembly1_digraph.assembly_digraph.number_of_nodes() == 343


def test_assembly1_edge_count(assembly1_digraph):
    assert assembly1_digraph.assembly_digraph.number_of_edges() == 1105


# ---------------------------------------------------------------------------
# edge weights
# ---------------------------------------------------------------------------

def test_edge_weights_are_non_negative(example_digraph):
    for _, _, data in example_digraph.assembly_digraph.edges(data=True):
        assert data["edge_weight"] >= 0.0


def test_edge_operations_in_main_graph(example_digraph):
    main_edges = set(example_digraph.graph.edges())
    main_edges_rev = {(b, a) for a, b in main_edges}
    for _, _, data in example_digraph.assembly_digraph.edges(data=True):
        op = data["operation"]
        assert op in main_edges or op in main_edges_rev


# ---------------------------------------------------------------------------
# save_class_to_pickle — Bug 2 fix
# ---------------------------------------------------------------------------

def test_save_dict_format_loads_as_dict(example_digraph):
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        fname = f.name
    try:
        example_digraph.pkl_save_format = "dict"
        example_digraph.save_class_to_pickle(fname)
        with open(fname, "rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, dict), (
            "pkl_save_format='dict' must save a dict, not a class object (Bug 2)"
        )
        assert "assembly_digraph" in loaded
        assert "main_graph" in loaded
    finally:
        os.unlink(fname)


def test_save_class_format_loads_as_class(example_digraph):
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        fname = f.name
    try:
        example_digraph.pkl_save_format = "class"
        example_digraph.save_class_to_pickle(fname)
        with open(fname, "rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, AssemblyDigraph)
    finally:
        os.unlink(fname)
        example_digraph.pkl_save_format = "dict"


def test_save_invalid_format_raises(example_digraph):
    example_digraph.pkl_save_format = "xml"
    with pytest.raises(ValueError):
        example_digraph.save_class_to_pickle("dummy.pkl")
    example_digraph.pkl_save_format = "dict"


# ---------------------------------------------------------------------------
# get_num_layers property
# ---------------------------------------------------------------------------

def test_get_num_layers(example_digraph):
    expected = example_digraph.graph.number_of_edges() + 1
    assert example_digraph.get_num_layers == expected


# ---------------------------------------------------------------------------
# edge reduction
# ---------------------------------------------------------------------------

def test_edge_reduction_reduces_digraph():
    ad = AssemblyDigraph(
        file_name=ASSEMBLY_1_JSON,
        w_tech=1.0, w_hand=0.0, w_tol=0.0, w_mass=0.0,
        reduction_percentage=50,
    )
    ad.compute_assembly_digraph_complete()
    assert ad.assembly_digraph.number_of_edges() < 1105
