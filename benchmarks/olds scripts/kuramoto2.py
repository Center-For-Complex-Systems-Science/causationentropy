import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import warnings

# Tigramite imports
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# from tigramite.independence_tests.cmiknn import CMIknn  # optional, slower but more nonlinear

# User Library Imports
from causationentropy import discover_network
from causationentropy.graph import pcmci_to_networkx
from causationentropy.core.linalg import companion_matrix

warnings.filterwarnings("ignore")


# =============================================================================
# KURAMOTO SIMULATION + CORRECT-BASIS TRANSFORM
# =============================================================================


def wrap_to_pi(x):
    """Wrap angles to [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def simulate_kuramoto(
    G,
    T,
    dt=0.05,
    rho=0.7,
    seed=0,
    omega_mean=0.0,
    omega_std=1.0,
    phase_noise_std=0.02,
    burn_in=50,
    normalize_by_indegree=True,
):
    """
    Simulate directed Kuramoto on graph G.

    Uses coupling into i:
        dtheta_i = omega_i + rho * sum_j A_{ji} sin(theta_j - theta_i) / indeg_i

    Returns:
        theta: (T, n) phases wrapped to [-pi, pi]
        meta: dict
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes).astype(float)  # A_ij = i -> j

    omega = rng.normal(loc=omega_mean, scale=omega_std, size=n)
    theta = rng.uniform(low=-np.pi, high=np.pi, size=n)

    if normalize_by_indegree:
        indeg = A.sum(axis=0)  # indeg_i = sum_j A_{j,i} (incoming)
        indeg_safe = np.where(indeg > 0, indeg, 1.0)
    else:
        indeg_safe = np.ones(n)

    total_steps = burn_in + T
    out = np.zeros((T, n), dtype=float)

    for step in range(total_steps):
        # coupling_i = sum_j A_{j,i} sin(theta_j - theta_i)
        coupling = (A.T * np.sin(theta[None, :] - theta[:, None])).sum(axis=1)
        dtheta = omega + (rho * coupling / indeg_safe)

        noise = rng.normal(loc=0.0, scale=phase_noise_std, size=n)
        theta = theta + dt * dtheta + noise
        theta = wrap_to_pi(theta)

        if step >= burn_in:
            out[step - burn_in] = theta

    return out, {"omega": omega, "dt": dt, "rho": rho}


def phase_velocity(theta, dt):
    """
    theta: (T, n) wrapped angles
    returns v: (T-1, n) with wrapped differences / dt
    """
    dtheta = wrap_to_pi(theta[1:] - theta[:-1])
    return dtheta / dt


def build_kuramoto_basis(theta, dt):
    """
    Build expanded data matrix:
      targets: v_i(t) for t=1..T-1
      predictors: s_{i<-j}(t-1) = sin(theta_j(t-1) - theta_i(t-1))

    Returns:
      X: (T-1, n + n*(n-1)) with columns [v_0..v_{n-1}, s_{0<-1}..]
      meta: mapping info
      var_names: list of names aligned with columns of X
    """
    T, n = theta.shape
    v = phase_velocity(theta, dt)  # (T-1, n), aligned to t=1..T-1

    th = theta[:-1]  # predictors at t-1, shape (T-1, n)
    S = np.zeros((T - 1, n * (n - 1)), dtype=float)

    basis_map = []  # local index -> (i, j)
    names = [f"v{i}" for i in range(n)]

    col = 0
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            S[:, col] = np.sin(th[:, j] - th[:, i])
            basis_map.append((i, j))  # s_{i<-j}
            names.append(f"s_{i}<-{j}")
            col += 1

    X = np.concatenate([v, S], axis=1)
    meta = {
        "n": n,
        "dt": dt,
        "pred_offset": n,
        "basis_map": basis_map,  # local idx -> (i, j)
        "p": X.shape[1],
    }
    return X, meta, names


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_PARAMS = {
    "n_nodes": 10,
    "T": 600,  # increased a bit since we lose one step to velocity (T-1)
    "n_trials": 5,
    "alpha": 0.05,
    "tau_max": 1,  # we are explicitly using lag-1 basis: s(t-1) -> v(t)
    # Kuramoto sim params
    "dt": 0.05,
    "omega_std": 1.0,
    "phase_noise_std": 0.02,
    "burn_in": 50,
    "normalize_by_indegree": False,  # often helps recovery by preserving heterogeneity
}

EXPERIMENTS = {
    "ER_Density": {
        "type": "Erdos-Renyi",
        "vary_param": "p_edge",
        "label": "Edge Probability (p)",
        "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "defaults": {"rho": 0.7},
    },
    "Coupling_Strength": {
        "type": "Erdos-Renyi",
        "vary_param": "rho",
        "label": "Coupling Strength (rho)",
        "values": [0.1, 0.3, 0.5, 0.7, 0.9],
        "defaults": {"p_edge": 0.3},
    },
    "ScaleFree_Hubs": {
        "type": "Scale-Free",
        "vary_param": "m_attachment",
        "label": "Attachment Edges (m)",
        "values": [1, 2, 3, 4],
        "defaults": {"rho": 0.7},
    },
    "SmallWorld_Rewiring": {
        "type": "Small-World",
        "vary_param": "p_rewire",
        "label": "Rewiring Probability (p)",
        "values": [0.0, 0.1, 0.3, 0.5, 0.8, 1.0],
        "defaults": {"rho": 0.7, "k_neighbors": 2},
    },
}


# =============================================================================
# HELPERS
# =============================================================================


def recover_node_adj_from_expanded_adj(
    adj_exp, basis_meta, tau_max=1, threshold=0.0, mode="binary"
):
    """
    Map expanded-variable adjacency back to original node adjacency.

    Expanded variables are ordered as:
      [v0..v(n-1), s_{0<-1}, s_{0<-2}, ..., s_{i<-j}, ...]  (i!=j)

    Recovery rule:
      if adj_exp[s_{i<-j} at lag1 -> v_i at lag0] is present/significant,
      then original edge j -> i is present.

    Parameters
    ----------
    adj_exp : (M, M) np.ndarray
        Adjacency over expanded variables. Can be:
          - (p, p)   : variables only
          - (p*(tau_max+1), p*(tau_max+1)) : lag-augmented/companion-style
    basis_meta : dict
        From build_kuramoto_basis(): contains
          - n
          - pred_offset (= n)
          - p
          - basis_map: list where basis_map[k] = (i, j) for s_{i<-j}
    tau_max : int
        Lag used in discovery. If adj is lag-augmented, we use edges from lag=1 block to lag=0 block.
    threshold : float
        Consider an edge present if abs(weight) > threshold (or weight != 0 if threshold=0).
    mode : {"binary", "weighted"}
        - "binary": returns {0,1} adjacency
        - "weighted": returns weights aggregated into node edges (max abs over evidence)

    Returns
    -------
    A_node : (n, n) np.ndarray
        Recovered node adjacency where A_node[j, i] corresponds to j -> i.
    """
    n = basis_meta["n"]
    p = basis_meta["p"]
    pred_offset = basis_meta["pred_offset"]  # == n
    basis_map = basis_meta["basis_map"]  # index k -> (i, j)

    adj_exp = np.asarray(adj_exp)
    M = adj_exp.shape[0]
    assert adj_exp.shape[0] == adj_exp.shape[1], "adj_exp must be square"

    # Decide whether this is lag-augmented
    lag_augmented = M == p * (tau_max + 1)

    # Initialize recovered node adjacency
    A_node = np.zeros((n, n), dtype=float)

    def edge_weight(src, tgt):
        w = adj_exp[src, tgt]
        return w

    # Indices of v_i in the expanded variable list (lag0 within a block)
    v_idx = np.arange(n)  # 0..n-1

    for k, (i, j) in enumerate(basis_map):
        s_idx = pred_offset + k  # column index of s_{i<-j} in the expanded list

        if lag_augmented:
            # Typical block layout: [lag0 vars | lag1 vars | ...]
            # We want: s at lag1  -> v at lag0
            src = 1 * p + s_idx  # lag1 block
            tgt = 0 * p + i  # v_i at lag0
        else:
            # If not lag-augmented, we assume the discovery already encoded lag in the adjacency
            # (or you're using a non-lag matrix). Use direct s_idx -> v_i.
            src = s_idx
            tgt = i

        w = edge_weight(src, tgt)

        present = (abs(w) > threshold) if threshold > 0 else (w != 0)

        if present:
            if mode == "binary":
                A_node[j, i] = 1.0
            elif mode == "weighted":
                # Aggregate evidence (keep strongest absolute effect for j->i)
                if abs(w) > abs(A_node[j, i]):
                    A_node[j, i] = w
            else:
                raise ValueError("mode must be 'binary' or 'weighted'")

    return A_node.astype(int) if mode == "binary" else A_node


def generate_graph_topology(topo_type, n_nodes, params, seed):
    if topo_type == "Erdos-Renyi":
        p = params.get("p_edge", 0.2)
        return nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)

    if topo_type == "Scale-Free":
        m = params.get("m_attachment", 1)
        G_und = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
        return nx.DiGraph(G_und)  # becomes bidirected

    if topo_type == "Small-World":
        k = params.get("k_neighbors", 2)
        p = params.get("p_rewire", 0.1)
        G_und = nx.connected_watts_strogatz_graph(n_nodes, k, p, tries=100, seed=seed)
        return nx.DiGraph(G_und)  # becomes bidirected

    raise ValueError(f"Unknown topology: {topo_type}")


def compute_metrics(predicted_adj, true_adj, time_taken):
    B = (predicted_adj != 0).astype(int)
    A = (true_adj != 0).astype(int)

    tp = np.sum((B == 1) & (A == 1))
    fp = np.sum((B == 1) & (A == 0))
    tn = np.sum((B == 0) & (A == 0))
    fn = np.sum((B == 0) & (A == 1))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
    shd = fp + fn

    return {
        "TPR": tpr,
        "FPR": fpr,
        "Precision": precision,
        "F1": f1,
        "SHD": shd,
        "Time": time_taken,
    }


# =============================================================================
# MAIN
# =============================================================================

results = []

print(
    f"Starting Kuramoto-basis Experiments with {GLOBAL_PARAMS['n_nodes']} nodes "
    f"over {GLOBAL_PARAMS['n_trials']} trials..."
)

total_iters = sum(
    len(cfg["values"]) * GLOBAL_PARAMS["n_trials"] for cfg in EXPERIMENTS.values()
)
pbar = tqdm(total=total_iters, desc="Total progress")

for exp_name, config in EXPERIMENTS.items():
    print(f"\n--- Running Experiment: {exp_name} ---")

    param_name = config["vary_param"]
    param_values = config["values"]
    defaults = config["defaults"]

    for val in param_values:
        current_params = defaults.copy()
        current_params[param_name] = val

        for trial in range(GLOBAL_PARAMS["n_trials"]):
            seed = 42 + (trial * 100)

            # 1) Graph + ground truth
            G = generate_graph_topology(
                config["type"], GLOBAL_PARAMS["n_nodes"], current_params, seed
            )
            true_adj = nx.to_numpy_array(G).astype(int)

            # 2) Simulate phases
            theta, _ = simulate_kuramoto(
                G=G,
                T=GLOBAL_PARAMS["T"],
                dt=GLOBAL_PARAMS["dt"],
                rho=current_params.get("rho", 0.7),
                seed=seed,
                omega_mean=0.0,
                omega_std=GLOBAL_PARAMS["omega_std"],
                phase_noise_std=GLOBAL_PARAMS["phase_noise_std"],
                burn_in=GLOBAL_PARAMS["burn_in"],
                normalize_by_indegree=GLOBAL_PARAMS["normalize_by_indegree"],
            )

            # 3) Build correct basis data matrix X = [v | sin-differences]
            X, basis_meta, var_names = build_kuramoto_basis(
                theta, dt=GLOBAL_PARAMS["dt"]
            )
            T_eff = X.shape[0]

            dataframe = pp.DataFrame(
                X, datatime={0: np.arange(T_eff)}, var_names=var_names
            )

            # -------------------------------------------------------
            # METHOD 1: OCE on expanded variables, then map back
            # -------------------------------------------------------
            oce_estimators = [
                "gaussian"
            ]  # add "knn"/"kde" if you want, but start simple

            for est in oce_estimators:
                start_time = time.time()
                network = discover_network(
                    data=X, max_lag=GLOBAL_PARAMS["tau_max"], information=est
                )
                # map expanded graph -> node adjacency
                # pred_adj = recovered_node_adj_from_expanded_graph(network, basis_meta)
                pred_adj_exp = nx.to_numpy_array(network)  # learned on expanded vars
                pred_adj_nodes = recover_node_adj_from_expanded_adj(
                    pred_adj_exp, basis_meta, tau_max=GLOBAL_PARAMS["tau_max"]
                )

                true_adj_nodes = nx.to_numpy_array(G).astype(int)
                metrics = compute_metrics(
                    pred_adj_nodes, true_adj_nodes, time.time() - start_time
                )

                # metrics = compute_metrics(pred_adj, true_adj, time.time() - start_time)
                metrics.update(
                    {
                        "Experiment": exp_name,
                        "Parameter": val,
                        "Method": f"OCE ({est})",
                        "Trial": trial,
                    }
                )
                results.append(metrics)

            # -------------------------------------------------------
            # METHOD 2: PCMCI on expanded variables, then map back
            # -------------------------------------------------------
            tigramite_configs = [
                ("PCMCI (ParCorr)", ParCorr()),
                # ("PCMCI (CMIknn)", CMIknn()),  # optional, slower but can help on nonlinearities
            ]

            for method_label, ind_test in tigramite_configs:
                start_time = time.time()
                pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ind_test, verbosity=0)
                pcmci_res = pcmci.run_pcmci(
                    tau_max=GLOBAL_PARAMS["tau_max"], pc_alpha=GLOBAL_PARAMS["alpha"]
                )

                graph_nx = pcmci_to_networkx(pcmci_res)
                # pred_adj = recovered_node_adj_from_expanded_graph(graph_nx, basis_meta)
                pred_adj_exp = nx.to_numpy_array(network)  # learned on expanded vars
                pred_adj_nodes = recover_node_adj_from_expanded_adj(
                    pred_adj_exp, basis_meta, tau_max=GLOBAL_PARAMS["tau_max"]
                )

                true_adj_nodes = nx.to_numpy_array(G).astype(int)
                metrics = compute_metrics(
                    pred_adj_nodes, true_adj_nodes, time.time() - start_time
                )

                # metrics = compute_metrics(pred_adj, true_adj, time.time() - start_time)
                metrics.update(
                    {
                        "Experiment": exp_name,
                        "Parameter": val,
                        "Method": method_label,
                        "Trial": trial,
                    }
                )
                results.append(metrics)

            pbar.update(1)

pbar.close()

# =============================================================================
# SAVE + PLOTS + SUMMARY
# =============================================================================

df_results = pd.DataFrame(results)
df_results.to_csv("causal_discovery_kuramoto_basis_results.csv", index=False)
print(
    "\nSimulation complete. Results saved to causal_discovery_kuramoto_basis_results.csv"
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
experiments_list = list(EXPERIMENTS.keys())

fig, axes = plt.subplots(
    len(experiments_list), 2, figsize=(14, 5 * len(experiments_list))
)
if len(experiments_list) == 1:
    axes = np.array([axes])

for idx, exp_name in enumerate(experiments_list):
    exp_data = df_results[df_results["Experiment"] == exp_name]
    config = EXPERIMENTS[exp_name]

    ax_f1 = axes[idx, 0]
    sns.lineplot(
        data=exp_data,
        x="Parameter",
        y="F1",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        ax=ax_f1,
        ci=68,
    )
    ax_f1.set_title(f"{exp_name}: F1 Score")
    ax_f1.set_xlabel(config["label"])
    ax_f1.set_ylabel("F1")

    ax_tpr = axes[idx, 1]
    sns.lineplot(
        data=exp_data,
        x="Parameter",
        y="TPR",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        ax=ax_tpr,
        ci=68,
    )
    ax_tpr.set_title(f"{exp_name}: True Positive Rate")
    ax_tpr.set_xlabel(config["label"])
    ax_tpr.set_ylabel("TPR")

    if idx == 0:
        ax_f1.legend(bbox_to_anchor=(1.05, 1.2), loc="upper left", ncol=3)
        ax_tpr.get_legend().remove()
    else:
        ax_f1.get_legend().remove()
        ax_tpr.get_legend().remove()

plt.tight_layout()
plt.savefig(
    "kuramoto_basis_comprehensive_causal_analysis.png", dpi=300, bbox_inches="tight"
)
plt.show()

print("\n" + "=" * 80)
print("AGGREGATED PERFORMANCE BY EXPERIMENT (KURAMOTO BASIS)")
print("=" * 80)

summary = (
    df_results.groupby(["Experiment", "Method"])[["F1", "TPR", "FPR", "Time"]]
    .mean()
    .reset_index()
)
summary.to_csv("kuramoto_basis_summary.csv", index=False)

for exp_name in experiments_list:
    print(f"\nExperiment: {exp_name}")
    print("-" * 80)
    exp_summary = summary[summary["Experiment"] == exp_name].sort_values(
        "F1", ascending=False
    )
    print(
        exp_summary[["Method", "F1", "TPR", "FPR", "Time"]].to_string(
            index=False, float_format="%.3f"
        )
    )
