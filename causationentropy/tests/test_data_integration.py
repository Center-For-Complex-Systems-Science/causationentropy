#!/usr/bin/env python3
"""
Integration test for all entropy methods in causal discovery.

This test runs discover_network() for every supported information type and method,
using synthetic data generators that match each entropy's distributional assumptions.
"""
import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from causationentropy.core.discovery import discover_network
from causationentropy.datasets.synthetic import (
    linear_stochastic_gaussian_process,
    poisson_coupled_oscillators,
    negative_binomial_coupled_oscillators, 
    hawkes_coupled_processes,
    von_mises_coupled_oscillators,
    laplace_coupled_oscillators
)


def create_test_graph(n_nodes=5, edge_prob=0.3, seed=42) -> nx.DiGraph:
    """Create a simple test graph for all methods."""
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed, directed=True)
    return G


def calculate_recovery_metrics(true_adj: np.ndarray, inferred_graph: nx.DiGraph) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score for network recovery."""
    n_nodes = true_adj.shape[0]
    inferred_adj = nx.to_numpy_array(inferred_graph, weight=None)
    
    # Convert to binary adjacency matrices
    true_edges = (true_adj > 0).astype(int)
    inferred_edges = (inferred_adj > 0).astype(int)
    
    # Calculate metrics
    tp = np.sum((true_edges == 1) & (inferred_edges == 1))
    fp = np.sum((true_edges == 0) & (inferred_edges == 1))
    fn = np.sum((true_edges == 1) & (inferred_edges == 0))
    tn = np.sum((true_edges == 0) & (inferred_edges == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_edges': int(np.sum(true_edges)),
        'inferred_edges': int(np.sum(inferred_edges))
    }


def test_method_information_combination(method: str, information: str, 
                                      test_graph: Optional[nx.DiGraph] = None,
                                      n_nodes: int = 5, T: int = 100) -> Dict:
    """Test a specific method-information combination."""
    
    print(f"Testing method='{method}', information='{information}'...")
    
    # Create test graph if not provided
    if test_graph is None:
        test_graph = create_test_graph(n_nodes)
    
    # Generate appropriate synthetic data
    try:
        if information == 'gaussian':
            # Use existing Gaussian generator with custom graph
            data = linear_stochastic_gaussian_process(rho=0.7, n=n_nodes, T=T, p=0.0, seed=42)
            # Get true adjacency from the graph structure used internally
            true_adj = nx.to_numpy_array(test_graph)
            
        elif information == 'poisson':
            data, true_adj = poisson_coupled_oscillators(
                n=n_nodes, T=T, G=test_graph, lambda_base=2.0, coupling_strength=0.3, seed=42
            )
            
        elif information == 'negative_binomial':
            data, true_adj = negative_binomial_coupled_oscillators(
                n=n_nodes, T=T, G=test_graph, r_base=5, p_nb=0.3, coupling_strength=0.2, seed=42
            )
            
        elif information == 'hawkes':
            data, true_adj = hawkes_coupled_processes(
                n=n_nodes, T=T, G=test_graph, mu_base=0.5, alpha=0.3, beta=1.0, 
                coupling_strength=0.2, dt=0.1, seed=42
            )
            
        elif information == 'von_mises':
            data, true_adj = von_mises_coupled_oscillators(
                n=n_nodes, T=T, G=test_graph, kappa_base=2.0, coupling_strength=0.5, 
                freq_base=0.1, seed=42
            )
            
        elif information == 'laplace':
            data, true_adj = laplace_coupled_oscillators(
                n=n_nodes, T=T, G=test_graph, scale_base=0.5, coupling_strength=0.1, seed=42
            )
            
        else:  # For kde, knn, geometric_knn, histogram - use Gaussian data
            data = linear_stochastic_gaussian_process(rho=0.7, n=n_nodes, T=T, p=0.0, seed=42)
            true_adj = nx.to_numpy_array(test_graph)
            
    except Exception as e:
        return {
            'method': method,
            'information': information,
            'status': 'data_generation_failed',
            'error': str(e),
            'metrics': {}
        }
    
    # Run causal discovery
    try:
        inferred_graph = discover_network(
            data=data,
            method=method,
            information=information,
            max_lag=2,  # Small lag for faster testing
            alpha_forward=0.1,  # More lenient for testing
            alpha_backward=0.1,
            n_shuffles=50  # Fewer shuffles for speed
        )
        
        # Calculate recovery metrics
        metrics = calculate_recovery_metrics(true_adj, inferred_graph)
        
        return {
            'method': method,
            'information': information,
            'status': 'success',
            'error': None,
            'metrics': metrics,
            'data_shape': data.shape,
            'true_graph_edges': int(np.sum(true_adj > 0))
        }
        
    except Exception as e:
        return {
            'method': method,
            'information': information,
            'status': 'discovery_failed',
            'error': str(e),
            'metrics': {},
            'data_shape': data.shape if 'data' in locals() else None
        }


def run_comprehensive_integration_test(n_nodes: int = 4, T: int = 80) -> pd.DataFrame:
    """Run integration test for all method-information combinations."""
    
    # All supported methods and information types
    methods = ['standard', 'alternative', 'information_lasso', 'lasso']
    information_types = [
        'gaussian', 'kde', 'knn', 'geometric_knn', 'histogram',
        'poisson', 'negative_binomial', 'hawkes', 'von_mises', 'laplace'
    ]
    
    # Create a common test graph for consistency
    test_graph = create_test_graph(n_nodes, edge_prob=0.4, seed=42)
    print(f"Using test graph with {test_graph.number_of_edges()} edges")
    
    results = []
    total_tests = len(methods) * len(information_types)
    test_count = 0
    
    for method in methods:
        for information in information_types:
            test_count += 1
            print(f"Progress: {test_count}/{total_tests}")
            
            result = test_method_information_combination(
                method=method,
                information=information,
                test_graph=test_graph,
                n_nodes=n_nodes,
                T=T
            )
            results.append(result)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Expand metrics into separate columns
    metrics_df = pd.json_normalize(df['metrics'])
    df = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)
    
    return df


def analyze_results(df: pd.DataFrame):
    """Analyze and print summary of integration test results."""
    print("\n" + "="*80)
    print("INTEGRATION TEST RESULTS SUMMARY")
    print("="*80)
    
    # Overall success rate
    success_rate = (df['status'] == 'success').mean()
    print(f"Overall Success Rate: {success_rate:.1%} ({np.sum(df['status'] == 'success')}/{len(df)} tests)")
    
    # Success by method
    print(f"\nSuccess Rate by Method:")
    method_success = df.groupby('method')['status'].apply(lambda x: (x == 'success').mean())
    for method, rate in method_success.items():
        count = np.sum((df['method'] == method) & (df['status'] == 'success'))
        total = np.sum(df['method'] == method)
        print(f"  {method:18s}: {rate:.1%} ({count}/{total})")
    
    # Success by information type
    print(f"\nSuccess Rate by Information Type:")
    info_success = df.groupby('information')['status'].apply(lambda x: (x == 'success').mean())
    for info, rate in info_success.items():
        count = np.sum((df['information'] == info) & (df['status'] == 'success'))
        total = np.sum(df['information'] == info)
        print(f"  {info:18s}: {rate:.1%} ({count}/{total})")
    
    # Performance metrics for successful tests
    successful_tests = df[df['status'] == 'success']
    if len(successful_tests) > 0:
        print(f"\nAverage Performance Metrics (successful tests only):")
        for metric in ['precision', 'recall', 'f1_score']:
            if metric in successful_tests.columns:
                mean_val = successful_tests[metric].mean()
                std_val = successful_tests[metric].std()
                print(f"  {metric:12s}: {mean_val:.3f} ± {std_val:.3f}")
    
    # Failure analysis
    failed_tests = df[df['status'] != 'success']
    if len(failed_tests) > 0:
        print(f"\nFailure Analysis:")
        failure_types = failed_tests['status'].value_counts()
        for failure_type, count in failure_types.items():
            print(f"  {failure_type:25s}: {count} tests")
            
        # Show some error examples
        print(f"\nSample Errors:")
        unique_errors = failed_tests['error'].dropna().unique()[:5]
        for i, error in enumerate(unique_errors[:3], 1):
            print(f"  {i}. {error}")
    
    print("\n" + "="*80)
    return df


def main():
    """Run the comprehensive integration test."""
    print("Starting Comprehensive Causal Discovery Integration Test")
    print("Testing all method-information combinations...\n")
    
    # Run the test with smaller parameters for speed
    results_df = run_comprehensive_integration_test(n_nodes=4, T=60)
    
    # Analyze and display results
    final_df = analyze_results(results_df)
    
    # Save results
    results_file = 'integration_test_results.csv'
    final_df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")
    
    return final_df


if __name__ == "__main__":
    results = main()