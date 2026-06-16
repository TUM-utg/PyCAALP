"""Tests for pycaalp/gapp/read_write.py"""

import pytest
import networkx as nx
from pycaalp.gapp.read_write import read_graph_from_json


EXAMPLE_JSON = "data/example/example_parts.json"
ASSEMBLY_1_JSON = "data/assembly_1/assembly_1_2_tech_parts.json"


# ---------------------------------------------------------------------------
# read_graph_from_json
# ---------------------------------------------------------------------------

@pytest.fixture
def example_graph():
    return read_graph_from_json(EXAMPLE_JSON)


def test_read_graph_returns_nx_graph(example_graph):
    assert isinstance(example_graph, nx.Graph)


def test_read_graph_node_count(example_graph):
    # example_parts.json has 3 parts: P1, P2, P3
    assert example_graph.number_of_nodes() == 3


def test_read_graph_edge_count(example_graph):
    # example_parts.json has 2 joints
    assert example_graph.number_of_edges() == 2


def test_read_graph_node_attributes(example_graph):
    for node in example_graph.nodes():
        attrs = example_graph.nodes[node]
        assert "weight" in attrs
        assert "handling" in attrs


def test_read_graph_edge_attributes(example_graph):
    for u, v, attrs in example_graph.edges(data=True):
        assert "name" in attrs
        assert "technology" in attrs
        assert "time" in attrs
        assert "tolerance" in attrs
        assert "handling" in attrs
        assert "mass" in attrs


def test_read_graph_is_connected(example_graph):
    assert nx.is_connected(example_graph)


def test_read_graph_missing_file():
    with pytest.raises(FileNotFoundError):
        read_graph_from_json("data/nonexistent.json")


def test_read_graph_assembly1():
    g = read_graph_from_json(ASSEMBLY_1_JSON)
    assert g.number_of_edges() == 13
