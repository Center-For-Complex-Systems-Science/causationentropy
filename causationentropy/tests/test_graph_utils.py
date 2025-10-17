import unittest

import networkx as nx
import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from tigramite.toymodels import structural_causal_processes as scp

from causationentropy.graph.utils import (
    network_to_dataframe,
    networkx_to_pcmci,
    pcmci_network_to_dataframe,
    pcmci_to_networkx,
)


class TestGraphUtils(unittest.TestCase):
    def setUp(self):
        # Generate some sample data
        np.random.seed(42)
        data = np.random.randn(100, 3)
        # 0 -> 1 at lag 1
        data[:, 1] += 0.5 * np.roll(data[:, 0], 1)
        # 1 -> 2 at lag 2
        data[:, 2] += 0.5 * np.roll(data[:, 1], 2)

        self.dataframe = pp.DataFrame(
            data, datatime={0: np.arange(len(data))}, var_names=["X", "Y", "Z"]
        )
        pcmci = PCMCI(dataframe=self.dataframe, cond_ind_test=ParCorr(), verbosity=0)
        self.results = pcmci.run_pcmci(tau_max=2, pc_alpha=0.01)

    def test_pcmci_to_networkx(self):
        G = pcmci_to_networkx(self.results)

        self.assertIsInstance(G, nx.MultiDiGraph)
        self.assertEqual(G.number_of_nodes(), 3)

        # PCMCI finds 0 -> 1 at lag 1 and 1 -> 2 at lag 2
        # It may also find some contemporaneous links.
        # We check for the links we know should be there.

        # Check 0 -> 1 link
        edges = G.get_edge_data(0, 1)
        self.assertTrue(len(edges) > 0)
        self.assertTrue(
            any(d["link_type"] == "directed" and d["lag"] == 1 for d in edges.values())
        )

        # Check 1 -> 2 link
        edges = G.get_edge_data(1, 2)
        self.assertTrue(len(edges) > 0)
        self.assertTrue(
            any(d["link_type"] == "directed" and d["lag"] == 2 for d in edges.values())
        )

    def test_pcmci_to_networkx_2d_graph(self):
        results = {
            "graph": np.array([["", "-->"], ["", ""]]),
            "val_matrix": np.array([[0.0, 1.0], [0.0, 0.0]]),
            "p_matrix": np.array([[1.0, 0.01], [1.0, 1.0]]),
        }

        G = pcmci_to_networkx(results)

        self.assertIsInstance(G, nx.MultiDiGraph)
        self.assertEqual(G.number_of_nodes(), 2)
        self.assertTrue(G.has_edge(0, 1))
        self.assertFalse(G.has_edge(1, 0))

        edges = G.get_edge_data(0, 1)
        self.assertEqual(len(edges), 1)

        link_types = [d["link_type"] for d in edges.values()]
        self.assertIn("directed", link_types)

    def test_pcmci_to_networkx_binarize(self):
        G = pcmci_to_networkx(self.results, binarize=True, p_value=0.05)

        # Find the specific edges and check for significance
        edge_0_1_found = False
        for u, v, data in G.edges(data=True):
            if (
                u == 0
                and v == 1
                and data["link_type"] == "directed"
                and data["lag"] == 1
            ):
                self.assertTrue(data["significant"])
                edge_0_1_found = True
        self.assertTrue(edge_0_1_found)

        edge_1_2_found = False
        for u, v, data in G.edges(data=True):
            if (
                u == 1
                and v == 2
                and data["link_type"] == "directed"
                and data["lag"] == 2
            ):
                self.assertTrue(data["significant"])
                edge_1_2_found = True
        self.assertTrue(edge_1_2_found)

    def test_binarize_with_backward_arrow(self):
        """Test binarize=True with '<--' link type (lines 67-70)."""
        results = {
            "graph": np.array(
                [
                    [["", "", ""], ["", "", ""], ["", "", ""]],
                    [["<--", "", ""], ["", "", ""], ["", "", ""]],
                    [["", "", ""], ["", "", ""], ["", "", ""]],
                ]
            ),
            "val_matrix": np.ones((3, 3, 3)) * 0.6,
            "p_matrix": np.ones((3, 3, 3)) * 0.02,
        }

        G = pcmci_to_networkx(results, binarize=True, p_value=0.05)

        # '<--' at [1,0,0] means 0 -> 1
        self.assertTrue(G.has_edge(0, 1))
        edges = G.get_edge_data(0, 1)
        found = False
        for edge_data in edges.values():
            if edge_data["lag"] == 0 and edge_data["link_type"] == "directed":
                self.assertTrue(edge_data["significant"])
                found = True
        self.assertTrue(found)

    def test_binarize_with_undirected_and_conflicting(self):
        """Test binarize=True with 'o-o' and 'x-x' link types (lines 71-76)."""
        # Create graph matrix where undirected/conflicting edges appear at both [i,j] and [j,i]
        graph = np.full((3, 3, 2), "", dtype="<U3")
        # 'o-o' at [0,1,0] and [1,0,0]
        graph[0, 1, 0] = "o-o"
        graph[1, 0, 0] = "o-o"
        # 'x-x' at [1,2,1] and [2,1,1]
        graph[1, 2, 1] = "x-x"
        graph[2, 1, 1] = "x-x"

        results = {
            "graph": graph,
            "val_matrix": np.ones((3, 3, 2)) * 0.7,
            "p_matrix": np.ones((3, 3, 2)) * 0.03,
        }

        G = pcmci_to_networkx(results, binarize=True, p_value=0.05)

        # 'o-o' at [0,1,0] and [1,0,0] should create undirected edge
        self.assertTrue(G.has_edge(0, 1))
        self.assertTrue(G.has_edge(1, 0))

        # Check for undirected link type
        edges_0_1 = G.get_edge_data(0, 1)
        found_undirected = False
        for edge_data in edges_0_1.values():
            if edge_data["lag"] == 0 and edge_data["link_type"] == "undirected":
                self.assertTrue(edge_data["significant"])
                found_undirected = True
        self.assertTrue(found_undirected)

        # 'x-x' at [1,2,1] and [2,1,1] should create conflicting edge
        self.assertTrue(G.has_edge(1, 2))
        self.assertTrue(G.has_edge(2, 1))

        # Check for conflicting link type
        edges_1_2 = G.get_edge_data(1, 2)
        found_conflicting = False
        for edge_data in edges_1_2.values():
            if edge_data["lag"] == 1 and edge_data["link_type"] == "conflicting":
                self.assertTrue(edge_data["significant"])
                found_conflicting = True
        self.assertTrue(found_conflicting)

    def test_binarize_with_possible_directed(self):
        """Test binarize=True with '-?>' link type (lines 77-79)."""
        graph = np.full((2, 2, 1), "", dtype="<U3")
        graph[0, 1, 0] = "-?>"

        results = {
            "graph": graph,
            "val_matrix": np.ones((2, 2, 1)) * 0.4,
            "p_matrix": np.ones((2, 2, 1)) * 0.08,
        }

        G = pcmci_to_networkx(results, binarize=True, p_value=0.1)

        # '-?>' at [0,1,0] should create possible directed edge
        self.assertTrue(G.has_edge(0, 1))
        edges = G.get_edge_data(0, 1)
        found = False
        for edge_data in edges.values():
            if edge_data["lag"] == 0 and edge_data["link_type"] == "possible_directed":
                self.assertTrue(edge_data["significant"])
                found = True
        self.assertTrue(found)

    def test_non_binarize_with_possible_directed(self):
        """Test binarize=False with '-?>' link type (line 181)."""
        graph = np.full((3, 3, 2), "", dtype="<U3")
        graph[0, 1, 0] = "-?>"
        graph[1, 2, 1] = "-?>"

        results = {
            "graph": graph,
            "val_matrix": np.ones((3, 3, 2)) * 0.6,
            "p_matrix": np.ones((3, 3, 2)) * 0.04,
        }

        G = pcmci_to_networkx(results, binarize=False)

        # '-?>' at [0,1,0] should create possible directed edge
        self.assertTrue(G.has_edge(0, 1))
        edges_0_1 = G.get_edge_data(0, 1)
        found_lag_0 = False
        for edge_data in edges_0_1.values():
            if edge_data["lag"] == 0 and edge_data["link_type"] == "possible_directed":
                self.assertEqual(edge_data["val"], 0.6)
                self.assertEqual(edge_data["p_value"], 0.04)
                self.assertNotIn("significant", edge_data)
                found_lag_0 = True
        self.assertTrue(found_lag_0)

        # '-?>' at [1,2,1] should create possible directed edge
        self.assertTrue(G.has_edge(1, 2))
        edges_1_2 = G.get_edge_data(1, 2)
        found_lag_1 = False
        for edge_data in edges_1_2.values():
            if edge_data["lag"] == 1 and edge_data["link_type"] == "possible_directed":
                self.assertEqual(edge_data["val"], 0.6)
                self.assertEqual(edge_data["p_value"], 0.04)
                self.assertNotIn("significant", edge_data)
                found_lag_1 = True
        self.assertTrue(found_lag_1)

    def test_pcmci_to_networkx_tau_zero_has_scalar_edge_metrics(self):
        np.random.seed(123)
        data = np.random.randn(300, 3)
        data[:, 1] += 0.9 * data[:, 0]
        data[:, 2] += 0.8 * data[:, 1]

        dataframe = pp.DataFrame(
            data, datatime={0: np.arange(len(data))}, var_names=["X", "Y", "Z"]
        )
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
        results = pcmci.run_pcmci(tau_max=0, pc_alpha=0.01)

        G = pcmci_to_networkx(results)

        self.assertGreater(G.number_of_edges(), 0)

        # tau_max=0 currently stores edge metrics as length-1 arrays; enforce scalars
        for _, _, attrs in G.edges(data=True):
            self.assertNotIsInstance(attrs["val"], np.ndarray)
            self.assertNotIsInstance(attrs["p_value"], np.ndarray)

    def test_link_types(self):
        graph = np.full((3, 3, 3), "", dtype="<U3")
        graph[0, 1, 0] = "-->"
        graph[0, 1, 1] = "-?>"
        graph[1, 0, 0] = "<--"
        graph[0, 2, 1] = "o-o"
        graph[2, 0, 1] = "o-o"
        graph[1, 2, 2] = "x-x"
        graph[2, 1, 2] = "x-x"

        results = {
            "graph": graph,
            "val_matrix": np.ones((3, 3, 3)),
            "p_matrix": np.zeros((3, 3, 3)),
        }

        G = pcmci_to_networkx(results)

        # graph[0, 1, 0] = '-->'
        self.assertTrue(G.has_edge(0, 1))
        self.assertIn(
            "directed",
            [d["link_type"] for d in G.get_edge_data(0, 1).values() if d["lag"] == 0],
        )

        # graph[0, 1, 1] = '-?>'
        self.assertTrue(G.has_edge(0, 1))
        self.assertIn(
            "possible_directed",
            [d["link_type"] for d in G.get_edge_data(0, 1).values() if d["lag"] == 1],
        )

        # graph[1, 0, 0] = '<--'
        self.assertTrue(G.has_edge(0, 1))
        self.assertIn(
            "directed",
            [d["link_type"] for d in G.get_edge_data(0, 1).values() if d["lag"] == 0],
        )

        # graph[0, 2, 1] = 'o-o' and graph[2, 0, 1] = 'o-o'
        self.assertTrue(G.has_edge(0, 2))
        self.assertTrue(G.has_edge(2, 0))
        self.assertIn(
            "undirected",
            [d["link_type"] for d in G.get_edge_data(0, 2).values() if d["lag"] == 1],
        )
        self.assertIn(
            "undirected",
            [d["link_type"] for d in G.get_edge_data(2, 0).values() if d["lag"] == 1],
        )

        # graph[1, 2, 2] = 'x-x' and graph[2, 1, 2] = 'x-x'
        self.assertTrue(G.has_edge(1, 2))
        self.assertTrue(G.has_edge(2, 1))
        self.assertIn(
            "conflicting",
            [d["link_type"] for d in G.get_edge_data(1, 2).values() if d["lag"] == 2],
        )
        self.assertIn(
            "conflicting",
            [d["link_type"] for d in G.get_edge_data(2, 1).values() if d["lag"] == 2],
        )

    def test_unknown_link_type(self):
        """Test early validation catches unknown link types."""
        results = {
            "graph": np.array(
                [
                    [["", "unknown", ""], ["", "", ""], ["", "", ""]],
                    [["", "", ""], ["", "", ""], ["", "", ""]],
                    [["", "", ""], ["", "", ""], ["", "", ""]],
                ]
            ),
            "val_matrix": np.ones((3, 3, 3)),
            "p_matrix": np.zeros((3, 3, 3)),
        }
        with self.assertRaises(ValueError) as ctx:
            pcmci_to_networkx(results)
        self.assertIn("Unknown link type: unknown", str(ctx.exception))

    def test_invalid_graph_dimensions(self):
        """Test error handling for invalid graph dimensions."""
        # 1D graph (invalid)
        results = {
            "graph": np.array([""]),
            "val_matrix": np.zeros((2, 2, 1)),
            "p_matrix": np.ones((2, 2, 1)),
        }
        with self.assertRaises(ValueError) as ctx:
            pcmci_to_networkx(results)
        self.assertIn("Expected PCMCI graph with 2 or 3 dimensions", str(ctx.exception))

        # 4D graph (invalid)
        results = {
            "graph": np.zeros((2, 2, 2, 2), dtype="<U3"),
            "val_matrix": np.zeros((2, 2, 1)),
            "p_matrix": np.ones((2, 2, 1)),
        }
        with self.assertRaises(ValueError) as ctx:
            pcmci_to_networkx(results)
        self.assertIn("Expected PCMCI graph with 2 or 3 dimensions", str(ctx.exception))

    def test_invalid_val_matrix_dimensions(self):
        """Test error handling for invalid val_matrix dimensions."""
        # 1D val_matrix (invalid)
        results = {
            "graph": np.array([["", "-->"], ["", ""]]),
            "val_matrix": np.array([0.0]),
            "p_matrix": np.ones((2, 2)),
        }
        with self.assertRaises(ValueError) as ctx:
            pcmci_to_networkx(results)
        self.assertIn(
            "Expected value matrix with 2 or 3 dimensions", str(ctx.exception)
        )

        # 4D val_matrix (invalid)
        results = {
            "graph": np.array([["", "-->"], ["", ""]]),
            "val_matrix": np.zeros((2, 2, 2, 2)),
            "p_matrix": np.ones((2, 2)),
        }
        with self.assertRaises(ValueError) as ctx:
            pcmci_to_networkx(results)
        self.assertIn(
            "Expected value matrix with 2 or 3 dimensions", str(ctx.exception)
        )

    def test_invalid_p_matrix_dimensions(self):
        """Test error handling for invalid p_matrix dimensions."""
        # 1D p_matrix (invalid)
        results = {
            "graph": np.array([["", "-->"], ["", ""]]),
            "val_matrix": np.zeros((2, 2)),
            "p_matrix": np.array([1.0]),
        }
        with self.assertRaises(ValueError) as ctx:
            pcmci_to_networkx(results)
        self.assertIn(
            "Expected p-value matrix with 2 or 3 dimensions", str(ctx.exception)
        )

        # 4D p_matrix (invalid)
        results = {
            "graph": np.array([["", "-->"], ["", ""]]),
            "val_matrix": np.zeros((2, 2)),
            "p_matrix": np.ones((2, 2, 2, 2)),
        }
        with self.assertRaises(ValueError) as ctx:
            pcmci_to_networkx(results)
        self.assertIn(
            "Expected p-value matrix with 2 or 3 dimensions", str(ctx.exception)
        )

    def test_mismatched_matrix_shapes(self):
        """Test error handling for mismatched matrix shapes."""
        results = {
            "graph": np.array([["", "-->"], ["", ""]]),
            "val_matrix": np.zeros((3, 3)),  # Different shape
            "p_matrix": np.ones((2, 2)),
        }
        with self.assertRaises(ValueError) as ctx:
            pcmci_to_networkx(results)
        self.assertIn(
            "PCMCI graph, value, and p-value matrices must share shape",
            str(ctx.exception),
        )

    def test_networkx_to_pcmci_roundtrip(self):
        # 1. Create a MultiDiGraph
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, lag=1, link_type="directed", val=0.5, p_value=0.01)
        G.add_edge(1, 2, lag=2, link_type="directed", val=0.6, p_value=0.02)
        G.add_edge(0, 2, lag=0, link_type="undirected", val=0.7, p_value=0.03)
        G.add_edge(2, 0, lag=0, link_type="undirected", val=0.7, p_value=0.03)

        # 2. Convert to pcmci results
        results = networkx_to_pcmci(G)

        # 3. Assertions on the results dictionary
        self.assertEqual(results["graph"].shape, (3, 3, 3))
        self.assertEqual(results["graph"][0, 1, 1], "-->")
        self.assertEqual(results["graph"][1, 2, 2], "-->")
        self.assertEqual(results["graph"][0, 2, 0], "o-o")
        self.assertEqual(results["graph"][2, 0, 0], "o-o")

        self.assertEqual(results["val_matrix"][0, 1, 1], 0.5)
        self.assertEqual(results["p_matrix"][0, 1, 1], 0.01)

        # 4. Convert back to MultiDiGraph and check for consistency
        G2 = pcmci_to_networkx(results)

        self.assertEqual(G.number_of_nodes(), G2.number_of_nodes())
        self.assertEqual(G.number_of_edges(), G2.number_of_edges())

        # Check edges
        g1_edges = sorted(
            [str((u, v, d["link_type"], d["lag"])) for u, v, d in G.edges(data=True)]
        )
        g2_edges = sorted(
            [str((u, v, d["link_type"], d["lag"])) for u, v, d in G2.edges(data=True)]
        )
        self.assertEqual(g1_edges, g2_edges)

    def test_networkx_to_pcmci_unknown_semantic_link_type(self):
        """Test networkx_to_pcmci with unknown semantic link type (line 159)."""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, lag=1, link_type="invalid_type", val=0.5, p_value=0.01)

        with self.assertRaises(ValueError) as ctx:
            networkx_to_pcmci(G)
        self.assertIn("Unknown semantic link type: invalid_type", str(ctx.exception))

    def test_networkx_to_pcmci_possible_directed(self):
        """Test networkx_to_pcmci with possible_directed link type (lines 155-157)."""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, lag=1, link_type="possible_directed", val=0.4, p_value=0.08)

        results = networkx_to_pcmci(G)

        # Check that the graph has the correct symbol
        self.assertEqual(results["graph"][0, 1, 1], "-?>")
        self.assertEqual(results["val_matrix"][0, 1, 1], 0.4)
        self.assertEqual(results["p_matrix"][0, 1, 1], 0.08)


class TestNetworkToDataFrame(unittest.TestCase):
    """Test the network_to_dataframe function for CausationEntropy graphs."""

    def test_basic_conversion(self):
        """Test basic conversion of CausationEntropy graph to DataFrame."""
        G = nx.MultiDiGraph()
        G.add_node("X0")
        G.add_node("X1")
        G.add_node("X2")

        G.add_edge("X0", "X1", lag=1, cmi=0.5, p_value=0.01)
        G.add_edge("X1", "X2", lag=2, cmi=0.3, p_value=0.05)

        df = network_to_dataframe(G)

        # Check DataFrame properties
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)

        # Check base columns
        self.assertIn("Source", df.columns)
        self.assertIn("Sink", df.columns)
        self.assertIn("Lag", df.columns)
        self.assertIn("CMI", df.columns)
        self.assertIn("P_Value", df.columns)

        # Check values
        self.assertEqual(df.iloc[0]["Source"], "X0")
        self.assertEqual(df.iloc[0]["Sink"], "X1")
        self.assertEqual(df.iloc[0]["Lag"], 1)
        self.assertEqual(df.iloc[0]["CMI"], 0.5)
        self.assertEqual(df.iloc[0]["P_Value"], 0.01)

    def test_with_all_metadata(self):
        """Test conversion with all optional metadata parameters."""
        G = nx.MultiDiGraph()
        G.add_edge("X0", "X1", lag=1, cmi=0.5, p_value=0.01)

        df = network_to_dataframe(
            G,
            method="standard",
            information="gaussian",
            alpha_forward=0.05,
            alpha_backward=0.05,
            metric="euclidean",
            bandwidth="silverman",
            k_means=5,
            n_shuffles=200,
            max_lag=3,
        )

        # Check all columns present
        expected_cols = [
            "Source",
            "Sink",
            "Lag",
            "CMI",
            "P_Value",
            "Method",
            "Information",
            "Alpha_Forward",
            "Alpha_Backward",
            "Metric",
            "Bandwidth",
            "K_Means",
            "N_Shuffles",
            "Max_Lag",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

        # Check metadata values
        self.assertEqual(df.iloc[0]["Method"], "standard")
        self.assertEqual(df.iloc[0]["Information"], "gaussian")
        self.assertEqual(df.iloc[0]["Alpha_Forward"], 0.05)
        self.assertEqual(df.iloc[0]["Alpha_Backward"], 0.05)
        self.assertEqual(df.iloc[0]["Metric"], "euclidean")
        self.assertEqual(df.iloc[0]["Bandwidth"], "silverman")
        self.assertEqual(df.iloc[0]["K_Means"], 5)
        self.assertEqual(df.iloc[0]["N_Shuffles"], 200)
        self.assertEqual(df.iloc[0]["Max_Lag"], 3)

    def test_partial_metadata(self):
        """Test with only some metadata parameters."""
        G = nx.MultiDiGraph()
        G.add_edge("X0", "X1", lag=1, cmi=0.5, p_value=0.01)

        df = network_to_dataframe(G, method="alternative", information="knn", max_lag=5)

        # Check that only specified metadata columns are present
        self.assertIn("Method", df.columns)
        self.assertIn("Information", df.columns)
        self.assertIn("Max_Lag", df.columns)

        # Check that unspecified metadata columns are not present
        self.assertNotIn("Alpha_Forward", df.columns)
        self.assertNotIn("Metric", df.columns)
        self.assertNotIn("K_Means", df.columns)

    def test_empty_graph(self):
        """Test conversion of empty graph."""
        G = nx.MultiDiGraph()
        G.add_node("X0")
        G.add_node("X1")

        df = network_to_dataframe(G)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)
        # Should still have the base columns
        self.assertIn("Source", df.columns)
        self.assertIn("Sink", df.columns)

    def test_missing_edge_attributes(self):
        """Test handling of edges with missing attributes."""
        G = nx.MultiDiGraph()
        # Edge without lag
        G.add_edge("X0", "X1", cmi=0.5, p_value=0.01)
        # Edge without cmi
        G.add_edge("X1", "X2", lag=1, p_value=0.02)
        # Edge without p_value
        G.add_edge("X2", "X0", lag=2, cmi=0.3)

        df = network_to_dataframe(G)

        self.assertEqual(len(df), 3)
        # Missing lag should default to 0
        self.assertEqual(df.iloc[0]["Lag"], 0)
        # Missing cmi should be None (which pandas represents as NaN in numeric columns)
        self.assertTrue(pd.isna(df.iloc[1]["CMI"]))
        # Missing p_value should be None (which pandas represents as NaN in numeric columns)
        self.assertTrue(pd.isna(df.iloc[2]["P_Value"]))

    def test_column_ordering(self):
        """Test that columns are in the correct order."""
        G = nx.MultiDiGraph()
        G.add_edge("X0", "X1", lag=1, cmi=0.5, p_value=0.01)

        df = network_to_dataframe(
            G, method="standard", alpha_forward=0.05, n_shuffles=200
        )

        # Base columns should be first
        cols = list(df.columns)
        self.assertEqual(cols[0], "Source")
        self.assertEqual(cols[1], "Sink")
        self.assertEqual(cols[2], "Lag")
        self.assertEqual(cols[3], "CMI")
        self.assertEqual(cols[4], "P_Value")

        # Metadata should follow in specified order
        self.assertEqual(cols[5], "Method")
        self.assertEqual(cols[6], "Alpha_Forward")
        self.assertEqual(cols[7], "N_Shuffles")

    def test_multiple_edges_same_nodes(self):
        """Test handling of multiple edges between same nodes (different lags)."""
        G = nx.MultiDiGraph()
        G.add_edge("X0", "X1", lag=1, cmi=0.5, p_value=0.01)
        G.add_edge("X0", "X1", lag=2, cmi=0.3, p_value=0.05)
        G.add_edge("X0", "X1", lag=3, cmi=0.2, p_value=0.10)

        df = network_to_dataframe(G)

        self.assertEqual(len(df), 3)
        # All edges should have same source and sink
        self.assertTrue(all(df["Source"] == "X0"))
        self.assertTrue(all(df["Sink"] == "X1"))
        # But different lags
        self.assertEqual(set(df["Lag"]), {1, 2, 3})


class TestPCMCINetworkToDataFrame(unittest.TestCase):
    """Test the pcmci_network_to_dataframe function for PCMCI graphs."""

    def test_basic_directed_edges(self):
        """Test conversion of basic directed edges."""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, lag=1, val=0.5, p_value=0.01, link_type="directed")
        G.add_edge(1, 2, lag=2, val=0.3, p_value=0.05, link_type="directed")

        df = pcmci_network_to_dataframe(G)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)

        # Check base columns
        expected_cols = ["Source", "Sink", "Lag", "Val", "P_Value", "Link_Type"]
        for col in expected_cols:
            self.assertIn(col, df.columns)

        # Check values
        self.assertEqual(df.iloc[0]["Source"], 0)
        self.assertEqual(df.iloc[0]["Sink"], 1)
        self.assertEqual(df.iloc[0]["Lag"], 1)
        self.assertEqual(df.iloc[0]["Val"], 0.5)
        self.assertEqual(df.iloc[0]["P_Value"], 0.01)
        self.assertEqual(df.iloc[0]["Link_Type"], "directed")

    def test_undirected_edges_deduplication(self):
        """Test deduplication of undirected edges."""
        G = nx.MultiDiGraph()
        # Undirected edges appear twice (u->v and v->u)
        G.add_edge(0, 1, lag=0, val=0.7, p_value=0.03, link_type="undirected")
        G.add_edge(1, 0, lag=0, val=0.7, p_value=0.03, link_type="undirected")

        df = pcmci_network_to_dataframe(G)

        # Should only appear once in DataFrame
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Link_Type"], "undirected")
        # Nodes should be in canonical (sorted) order
        self.assertEqual(df.iloc[0]["Source"], 0)
        self.assertEqual(df.iloc[0]["Sink"], 1)

    def test_conflicting_edges_deduplication(self):
        """Test deduplication of conflicting edges."""
        G = nx.MultiDiGraph()
        # Conflicting edges appear twice
        G.add_edge(1, 2, lag=1, val=0.5, p_value=0.02, link_type="conflicting")
        G.add_edge(2, 1, lag=1, val=0.5, p_value=0.02, link_type="conflicting")

        df = pcmci_network_to_dataframe(G)

        # Should only appear once
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Link_Type"], "conflicting")

    def test_possible_directed_edges(self):
        """Test handling of possible directed edges."""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, lag=1, val=0.4, p_value=0.08, link_type="possible_directed")

        df = pcmci_network_to_dataframe(G)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Link_Type"], "possible_directed")

    def test_with_significant_flag(self):
        """Test conversion with binarized significance flag."""
        G = nx.MultiDiGraph()
        G.add_edge(
            0, 1, lag=1, val=0.5, p_value=0.01, link_type="directed", significant=True
        )
        G.add_edge(
            1, 2, lag=1, val=0.2, p_value=0.10, link_type="directed", significant=False
        )

        df = pcmci_network_to_dataframe(G)

        self.assertEqual(len(df), 2)
        self.assertIn("Significant", df.columns)
        self.assertTrue(df.iloc[0]["Significant"])
        self.assertFalse(df.iloc[1]["Significant"])

    def test_empty_graph(self):
        """Test conversion of empty graph."""
        G = nx.MultiDiGraph()
        G.add_node(0)
        G.add_node(1)

        df = pcmci_network_to_dataframe(G)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)
        # Should have correct columns even when empty
        expected_cols = [
            "Source",
            "Sink",
            "Lag",
            "Val",
            "P_Value",
            "Link_Type",
            "Significant",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_missing_val_attribute(self):
        """Test handling of missing val attribute (fallback to cmi)."""
        G = nx.MultiDiGraph()
        # Edge with cmi instead of val
        G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01, link_type="directed")

        df = pcmci_network_to_dataframe(G)

        self.assertEqual(len(df), 1)
        # Should use cmi value for Val column
        self.assertEqual(df.iloc[0]["Val"], 0.5)

    def test_none_and_empty_values(self):
        """Test handling of None and empty string values."""
        G = nx.MultiDiGraph()
        # Edge with None values
        G.add_edge(0, 1, lag=1, val=None, p_value="", link_type="directed")

        df = pcmci_network_to_dataframe(G)

        self.assertEqual(len(df), 1)
        self.assertIsNone(df.iloc[0]["Val"])
        self.assertIsNone(df.iloc[0]["P_Value"])

    def test_numpy_scalar_conversion(self):
        """Test conversion of numpy scalar types."""
        G = nx.MultiDiGraph()
        # Edge with numpy scalar values
        G.add_edge(
            0,
            1,
            lag=1,
            val=np.float64(0.5),
            p_value=np.float32(0.01),
            link_type="directed",
        )

        df = pcmci_network_to_dataframe(G)

        self.assertEqual(len(df), 1)
        # Values should be converted to Python float
        self.assertIsInstance(df.iloc[0]["Val"], float)
        self.assertIsInstance(df.iloc[0]["P_Value"], float)
        self.assertEqual(df.iloc[0]["Val"], 0.5)
        self.assertAlmostEqual(df.iloc[0]["P_Value"], 0.01, places=5)

    def test_mixed_link_types(self):
        """Test graph with all different link types."""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, lag=1, val=0.5, p_value=0.01, link_type="directed")
        G.add_edge(1, 2, lag=1, val=0.3, p_value=0.06, link_type="possible_directed")
        G.add_edge(0, 2, lag=0, val=0.7, p_value=0.03, link_type="undirected")
        G.add_edge(2, 0, lag=0, val=0.7, p_value=0.03, link_type="undirected")
        G.add_edge(1, 3, lag=2, val=0.4, p_value=0.04, link_type="conflicting")
        G.add_edge(3, 1, lag=2, val=0.4, p_value=0.04, link_type="conflicting")

        df = pcmci_network_to_dataframe(G)

        # Should have 4 rows (directed, possible_directed, undirected deduplicated, conflicting deduplicated)
        self.assertEqual(len(df), 4)
        link_types = set(df["Link_Type"])
        self.assertEqual(
            link_types, {"directed", "possible_directed", "undirected", "conflicting"}
        )

    def test_column_ordering(self):
        """Test that columns are in the correct order."""
        G = nx.MultiDiGraph()
        G.add_edge(
            0, 1, lag=1, val=0.5, p_value=0.01, link_type="directed", significant=True
        )

        df = pcmci_network_to_dataframe(G)

        cols = list(df.columns)
        self.assertEqual(cols[0], "Source")
        self.assertEqual(cols[1], "Sink")
        self.assertEqual(cols[2], "Lag")
        self.assertEqual(cols[3], "Val")
        self.assertEqual(cols[4], "P_Value")
        self.assertEqual(cols[5], "Link_Type")
        self.assertEqual(cols[6], "Significant")

    def test_string_node_names(self):
        """Test with string node names instead of integers."""
        G = nx.MultiDiGraph()
        G.add_edge("X0", "X1", lag=1, val=0.5, p_value=0.01, link_type="directed")
        G.add_edge("X1", "X2", lag=2, val=0.3, p_value=0.05, link_type="directed")

        df = pcmci_network_to_dataframe(G)

        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["Source"], "X0")
        self.assertEqual(df.iloc[0]["Sink"], "X1")

    def test_deduplication_with_different_lags(self):
        """Test that undirected edges at different lags are handled separately."""
        G = nx.MultiDiGraph()
        # Undirected at lag 0
        G.add_edge(0, 1, lag=0, val=0.7, p_value=0.03, link_type="undirected")
        G.add_edge(1, 0, lag=0, val=0.7, p_value=0.03, link_type="undirected")
        # Undirected at lag 1
        G.add_edge(0, 1, lag=1, val=0.6, p_value=0.04, link_type="undirected")
        G.add_edge(1, 0, lag=1, val=0.6, p_value=0.04, link_type="undirected")

        df = pcmci_network_to_dataframe(G)

        # Should have 2 rows (one for each lag)
        self.assertEqual(len(df), 2)
        self.assertEqual(set(df["Lag"]), {0, 1})


class TestGraphUtilsWithTigramiteGenerators(unittest.TestCase):
    """Integration tests using Tigramite's structural causal process generators."""

    @classmethod
    def setUpClass(cls):
        cls.tau_max = 2
        links, _ = scp.generate_structural_causal_process(
            N=3,
            L=3,
            dependency_funcs=["linear"],
            dependency_coeffs=[-0.8, -0.4, 0.4, 0.8],
            auto_coeffs=[0.3, 0.5, 0.7],
            contemp_fraction=0.0,
            max_lag=cls.tau_max,
            noise_seed=321,
            seed=123,
        )
        cls.links = links

        graph_matrix = scp.links_to_graph(links, tau_max=cls.tau_max)
        cls.graph_matrix = graph_matrix.copy()

        cls.val_matrix = np.zeros(graph_matrix.shape, dtype=float)
        cls.p_matrix = np.ones(graph_matrix.shape, dtype=float)
        cls.edge_strength = {}

        for child, dependencies in links.items():
            for (parent, minus_lag), coeff, _ in dependencies:
                lag = -minus_lag
                cls.val_matrix[parent, child, lag] = abs(coeff)
                cls.p_matrix[parent, child, lag] = 0.01
                cls.edge_strength[(parent, child, lag)] = abs(coeff)

        cls.results = {
            "graph": graph_matrix,
            "val_matrix": cls.val_matrix,
            "p_matrix": cls.p_matrix,
        }

        cls.generated_graph = pcmci_to_networkx(cls.results)
        cls.expected_edges = set(cls.edge_strength.keys())

    def test_pcmci_to_networkx_matches_generated_links(self):
        actual_edges = {
            (u, v, data["lag"])
            for u, v, data in self.generated_graph.edges(data=True)
            if data["link_type"] == "directed"
        }

        self.assertEqual(actual_edges, self.expected_edges)

        for u, v, data in self.generated_graph.edges(data=True):
            key = (u, v, data["lag"])
            self.assertEqual(data["link_type"], "directed")
            self.assertAlmostEqual(data["val"], self.edge_strength[key])
            self.assertAlmostEqual(data["p_value"], 0.01)

    def test_networkx_to_pcmci_roundtrip_matches_generated_matrices(self):
        roundtrip = networkx_to_pcmci(self.generated_graph)

        np.testing.assert_array_equal(roundtrip["graph"], self.graph_matrix)
        np.testing.assert_allclose(roundtrip["val_matrix"], self.val_matrix)
        np.testing.assert_allclose(roundtrip["p_matrix"], self.p_matrix)

    def test_pcmci_network_to_dataframe_matches_generated_graph(self):
        df = pcmci_network_to_dataframe(self.generated_graph)

        self.assertEqual(len(df), len(self.expected_edges))

        for _, row in df.iterrows():
            key = (row["Source"], row["Sink"], row["Lag"])
            self.assertIn(key, self.expected_edges)
            self.assertAlmostEqual(row["Val"], self.edge_strength[key])
            self.assertAlmostEqual(row["P_Value"], 0.01)
            self.assertEqual(row["Link_Type"], "directed")


if __name__ == "__main__":
    unittest.main()
