import unittest
import numpy as np
import tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import networkx as nx

from causationentropy.graph.utils import pcmci_to_networkx, networkx_to_pcmci


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
            data,
            datatime={0: np.arange(len(data))},
            var_names=["X", "Y", "Z"]
        )
        pcmci = PCMCI(
            dataframe=self.dataframe,
            cond_ind_test=ParCorr(),
            verbosity=0)
        self.results = pcmci.run_pcmci(tau_max=2, pc_alpha=0.01)

    def test_pcmci_to_networkx(self):
        G, all_strengths = pcmci_to_networkx(self.results)

        self.assertIsInstance(G, nx.MultiDiGraph)
        self.assertEqual(G.number_of_nodes(), 3)
        
        # PCMCI finds 0 -> 1 at lag 1 and 1 -> 2 at lag 2
        # It may also find some contemporaneous links.
        # We check for the links we know should be there.
        
        # Check 0 -> 1 link
        edges = G.get_edge_data(0, 1)
        self.assertTrue(len(edges) > 0)
        self.assertTrue(any(d['link_type'] == 'directed' and d['lag'] == 1 for d in edges.values()))

        # Check 1 -> 2 link
        edges = G.get_edge_data(1, 2)
        self.assertTrue(len(edges) > 0)
        self.assertTrue(any(d['link_type'] == 'directed' and d['lag'] == 2 for d in edges.values()))

    def test_pcmci_to_networkx_2d_graph(self):
        results = {
            'graph': np.array([
                ['', '-->'],
                ['', '']
            ]),
            'val_matrix': np.array([
                [0.0, 1.0],
                [0.0, 0.0]
            ]),
            'p_matrix': np.array([
                [1.0, 0.01],
                [1.0, 1.0]
            ])
        }

        G, all_strengths = pcmci_to_networkx(results)

        self.assertIsInstance(G, nx.MultiDiGraph)
        self.assertEqual(G.number_of_nodes(), 2)
        self.assertTrue(G.has_edge(0, 1))
        self.assertFalse(G.has_edge(1, 0))
        
        edges = G.get_edge_data(0, 1)
        self.assertEqual(len(edges), 1)

        link_types = [d['link_type'] for d in edges.values()]
        self.assertIn('directed', link_types)

    def test_pcmci_to_networkx_binarize(self):
        G, all_strengths = pcmci_to_networkx(self.results, binarize=True, p_value=0.05)

        # Find the specific edges and check for significance
        edge_0_1_found = False
        for u, v, data in G.edges(data=True):
            if u == 0 and v == 1 and data['link_type'] == 'directed' and data['lag'] == 1:
                self.assertTrue(data['significant'])
                edge_0_1_found = True
        self.assertTrue(edge_0_1_found)

        edge_1_2_found = False
        for u, v, data in G.edges(data=True):
            if u == 1 and v == 2 and data['link_type'] == 'directed' and data['lag'] == 2:
                self.assertTrue(data['significant'])
                edge_1_2_found = True
        self.assertTrue(edge_1_2_found)

    def test_pcmci_to_networkx_tau_zero_has_scalar_edge_metrics(self):
        np.random.seed(123)
        data = np.random.randn(300, 3)
        data[:, 1] += 0.9 * data[:, 0]
        data[:, 2] += 0.8 * data[:, 1]

        dataframe = pp.DataFrame(
            data,
            datatime={0: np.arange(len(data))},
            var_names=["X", "Y", "Z"]
        )
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=ParCorr(),
            verbosity=0
        )
        results = pcmci.run_pcmci(tau_max=0, pc_alpha=0.01)

        G, strengths = pcmci_to_networkx(results)

        self.assertGreater(G.number_of_edges(), 0)

        # tau_max=0 currently stores edge metrics as length-1 arrays; enforce scalars
        for _, _, attrs in G.edges(data=True):
            self.assertNotIsInstance(attrs['val'], np.ndarray)
            self.assertNotIsInstance(attrs['p_value'], np.ndarray)

        for strength in strengths:
            self.assertNotIsInstance(strength, np.ndarray)

    def test_link_types(self):
        graph = np.full((3, 3, 3), '', dtype='<U3')
        graph[0, 1, 0] = '-->'
        graph[0, 1, 1] = '-?>'
        graph[1, 0, 0] = '<--'
        graph[0, 2, 1] = 'o-o'
        graph[2, 0, 1] = 'o-o'
        graph[1, 2, 2] = 'x-x'
        graph[2, 1, 2] = 'x-x'

        results = {
            'graph': graph,
            'val_matrix': np.ones((3, 3, 3)),
            'p_matrix': np.zeros((3, 3, 3))
        }
        
        G, _ = pcmci_to_networkx(results)
        
        # graph[0, 1, 0] = '-->'
        self.assertTrue(G.has_edge(0, 1))
        self.assertIn('directed', [d['link_type'] for d in G.get_edge_data(0, 1).values() if d['lag'] == 0])

        # graph[0, 1, 1] = '-?>'
        self.assertTrue(G.has_edge(0, 1))
        self.assertIn('possible_directed', [d['link_type'] for d in G.get_edge_data(0, 1).values() if d['lag'] == 1])

        # graph[1, 0, 0] = '<--'
        self.assertTrue(G.has_edge(0, 1))
        self.assertIn('directed', [d['link_type'] for d in G.get_edge_data(0, 1).values() if d['lag'] == 0])

        # graph[0, 2, 1] = 'o-o' and graph[2, 0, 1] = 'o-o'
        self.assertTrue(G.has_edge(0, 2))
        self.assertTrue(G.has_edge(2, 0))
        self.assertIn('undirected', [d['link_type'] for d in G.get_edge_data(0, 2).values() if d['lag'] == 1])
        self.assertIn('undirected', [d['link_type'] for d in G.get_edge_data(2, 0).values() if d['lag'] == 1])

        # graph[1, 2, 2] = 'x-x' and graph[2, 1, 2] = 'x-x'
        self.assertTrue(G.has_edge(1, 2))
        self.assertTrue(G.has_edge(2, 1))
        self.assertIn('conflicting', [d['link_type'] for d in G.get_edge_data(1, 2).values() if d['lag'] == 2])
        self.assertIn('conflicting', [d['link_type'] for d in G.get_edge_data(2, 1).values() if d['lag'] == 2])

    def test_unknown_link_type(self):
        results = {
            'graph': np.array([
                [['', 'unknown', ''], ['', '', ''], ['', '', '']],
                [['', '', ''], ['', '', ''], ['', '', '']],
                [['', '', ''], ['', '', ''], ['', '', '']]
            ]),
            'val_matrix': np.ones((3, 3, 3)),
            'p_matrix': np.zeros((3, 3, 3))
        }
        with self.assertRaises(ValueError):
            pcmci_to_networkx(results)

    def test_networkx_to_pcmci_roundtrip(self):
        # 1. Create a MultiDiGraph
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, lag=1, link_type='directed', val=0.5, p_value=0.01)
        G.add_edge(1, 2, lag=2, link_type='directed', val=0.6, p_value=0.02)
        G.add_edge(0, 2, lag=0, link_type='undirected', val=0.7, p_value=0.03)
        G.add_edge(2, 0, lag=0, link_type='undirected', val=0.7, p_value=0.03)

        # 2. Convert to pcmci results
        results = networkx_to_pcmci(G)

        # 3. Assertions on the results dictionary
        self.assertEqual(results['graph'].shape, (3, 3, 3))
        self.assertEqual(results['graph'][0, 1, 1], '-->')
        self.assertEqual(results['graph'][1, 2, 2], '-->')
        self.assertEqual(results['graph'][0, 2, 0], 'o-o')
        self.assertEqual(results['graph'][2, 0, 0], 'o-o')
        
        self.assertEqual(results['val_matrix'][0, 1, 1], 0.5)
        self.assertEqual(results['p_matrix'][0, 1, 1], 0.01)

        # 4. Convert back to MultiDiGraph and check for consistency
        G2, _ = pcmci_to_networkx(results)

        self.assertEqual(G.number_of_nodes(), G2.number_of_nodes())
        self.assertEqual(G.number_of_edges(), G2.number_of_edges())

        # Check edges
        g1_edges = sorted([str((u, v, d['link_type'], d['lag'])) for u, v, d in G.edges(data=True)])
        g2_edges = sorted([str((u, v, d['link_type'], d['lag'])) for u, v, d in G2.edges(data=True)])
        self.assertEqual(g1_edges, g2_edges)

if __name__ == '__main__':
    unittest.main()
