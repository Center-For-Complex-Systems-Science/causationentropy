import numpy as np
import networkx as nx
import sys

sys.path.insert(0, "/home/kslote/Desktop/causalentropy")

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from causationentropy import discover_network
from causationentropy.graph import pcmci_to_networkx

# Import functions from main script
exec(
    open("kuramoto_simplified.py")
    .read()
    .split(
        "# =============================================================================\n# MAIN"
    )[0]
)

# Configuration for ER_Density, param=0.50, trial=1
seed = 42 + (1 * 100)  # trial=1
p_edge = 0.5
rho = 0.7
n_nodes = 3
T = 500
dt = 0.05

print("=" * 80)
print(f"DEBUG: ER_Density param={p_edge} trial=1 (seed={seed})")
print("=" * 80)

# Generate graph
G = nx.erdos_renyi_graph(n_nodes, p_edge, seed=seed, directed=True)
true_adj = nx.to_numpy_array(G).astype(int)

print("\n[TRUE NETWORK]")
print(f"Adjacency matrix:\n{true_adj}")
print(
    f"Edges: {[(i,j) for i in range(n_nodes) for j in range(n_nodes) if true_adj[i,j]==1]}"
)
print(f"Total edges: {np.sum(true_adj)}")

# Simulate
theta, _ = simulate_kuramoto(
    G=G,
    T=T,
    dt=dt,
    rho=rho,
    seed=seed,
    omega_mean=0.0,
    omega_std=1.0,
    phase_noise_std=0.02,
    burn_in=50,
    normalize_by_indegree=False,
)

# Build basis
X, basis_meta, var_names = build_kuramoto_basis_lag1(theta, dt=dt)
print(f"\n[BASIS]")
print(f"Data shape: {X.shape}")
print(f"Variables: {var_names[:6]}... (first 6)")
print(f"Basis map (first 3): {basis_meta['basis_map'][:3]}")

# OCE Discovery
print("\n" + "=" * 80)
print("[OCE DISCOVERY]")
print("=" * 80)

import pandas as pd

X_df = pd.DataFrame(X, columns=var_names)

network = discover_network(
    data=X_df, max_lag=1, method="standard", information="gaussian"
)

print(f"\nOCE discovered {network.number_of_edges()} edges in expanded basis:")
for u, v, data in list(network.edges(data=True))[:10]:
    print(
        f"  {u} -> {v}, lag={data['lag']}, cmi={data['cmi']:.3f}, p={data['p_value']:.3f}"
    )

# Extract to node level
pred_adj_oce = extract_node_adjacency_from_basis_oce(
    network, basis_meta["n"], basis_meta["basis_map"], var_names
)

print(f"\n[OCE NODE-LEVEL ADJACENCY]")
print(f"Matrix:\n{pred_adj_oce}")
print(
    f"Edges: {[(i,j) for i in range(n_nodes) for j in range(n_nodes) if pred_adj_oce[i,j]==1]}"
)
print(f"Total edges: {np.sum(pred_adj_oce)}")

# Check which coupling terms predicted which velocities
print(f"\n[OCE COUPLING->VELOCITY EDGES]")
for i in range(n_nodes):
    velocity_var = f"v{i}"
    predictors = []
    for idx, (target, source) in enumerate(basis_meta["basis_map"]):
        coupling_var = var_names[n_nodes + idx]
        if network.has_edge(coupling_var, velocity_var):
            edges_data = network[coupling_var][velocity_var]
            for key, edata in edges_data.items():
                predictors.append((coupling_var, edata["lag"], edata["cmi"]))
    print(f"  v{i} predicted by: {predictors}")

# PCMCI Discovery
print("\n" + "=" * 80)
print("[PCMCI DISCOVERY]")
print("=" * 80)

T_eff = X.shape[0]
dataframe = pp.DataFrame(X, datatime={0: np.arange(T_eff)}, var_names=var_names)

pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
pcmci_res = pcmci.run_pcmci(tau_min=1, tau_max=1, pc_alpha=0.05)
graph_nx = pcmci_to_networkx(pcmci_res)

print(f"\nPCMCI discovered {graph_nx.number_of_edges()} edges in expanded basis:")
for u, v, data in list(graph_nx.edges(data=True))[:10]:
    print(
        f"  {u} -> {v}, lag={data['lag']}, val={data.get('val', 'N/A')}, p={data.get('p_value', 'N/A')}"
    )

# Extract to node level
pred_adj_pcmci = extract_node_adjacency_from_basis_pcmci(
    graph_nx, basis_meta["n"], basis_meta["basis_map"]
)

print(f"\n[PCMCI NODE-LEVEL ADJACENCY]")
print(f"Matrix:\n{pred_adj_pcmci}")
print(
    f"Edges: {[(i,j) for i in range(n_nodes) for j in range(n_nodes) if pred_adj_pcmci[i,j]==1]}"
)
print(f"Total edges: {np.sum(pred_adj_pcmci)}")

# Check which coupling terms predicted which velocities
print(f"\n[PCMCI COUPLING->VELOCITY EDGES]")
for i in range(n_nodes):
    velocity_idx = i
    predictors = []
    for idx, (target, source) in enumerate(basis_meta["basis_map"]):
        coupling_idx = n_nodes + idx
        if graph_nx.has_edge(coupling_idx, velocity_idx):
            edges_data = graph_nx[coupling_idx][velocity_idx]
            for key, edata in edges_data.items():
                predictors.append((f"s_{target}<-{source}", edata["lag"]))
    print(f"  v{i} predicted by: {predictors}")

# Comparison
print("\n" + "=" * 80)
print("[COMPARISON]")
print("=" * 80)

tp_oce = np.sum((pred_adj_oce == 1) & (true_adj == 1))
fp_oce = np.sum((pred_adj_oce == 1) & (true_adj == 0))
fn_oce = np.sum((pred_adj_oce == 0) & (true_adj == 1))
f1_oce = (
    2 * tp_oce / (2 * tp_oce + fp_oce + fn_oce)
    if (2 * tp_oce + fp_oce + fn_oce) > 0
    else 0
)

tp_pcmci = np.sum((pred_adj_pcmci == 1) & (true_adj == 1))
fp_pcmci = np.sum((pred_adj_pcmci == 1) & (true_adj == 0))
fn_pcmci = np.sum((pred_adj_pcmci == 0) & (true_adj == 1))
f1_pcmci = (
    2 * tp_pcmci / (2 * tp_pcmci + fp_pcmci + fn_pcmci)
    if (2 * tp_pcmci + fp_pcmci + fn_pcmci) > 0
    else 0
)

print(f"\nOCE:   TP={tp_oce}, FP={fp_oce}, FN={fn_oce}, F1={f1_oce:.3f}")
print(f"PCMCI: TP={tp_pcmci}, FP={fp_pcmci}, FN={fn_pcmci}, F1={f1_pcmci:.3f}")

# Show differences
diff_oce_only = pred_adj_oce - pred_adj_pcmci
diff_pcmci_only = pred_adj_pcmci - pred_adj_oce

if np.any(diff_oce_only > 0):
    print(f"\nEdges found by OCE but NOT PCMCI:")
    for i in range(n_nodes):
        for j in range(n_nodes):
            if diff_oce_only[i, j] > 0:
                print(f"  {i} -> {j}")

if np.any(diff_pcmci_only > 0):
    print(f"\nEdges found by PCMCI but NOT OCE:")
    for i in range(n_nodes):
        for j in range(n_nodes):
            if diff_pcmci_only[i, j] > 0:
                print(f"  {i} -> {j}")
                # Check what coupling terms led to this
                for idx, (target, source) in enumerate(basis_meta["basis_map"]):
                    if (target == j and source == i) or (target == i and source == j):
                        coupling_var = var_names[n_nodes + idx]
                        print(f"    Via coupling term: {coupling_var}")
                        # Check if OCE found this coupling
                        if network.has_edge(coupling_var, f"v{j}"):
                            print(f"      OCE had {coupling_var} -> v{j}: YES")
                        else:
                            print(f"      OCE had {coupling_var} -> v{j}: NO")
