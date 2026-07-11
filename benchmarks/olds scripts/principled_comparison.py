"""
Principled Causal Discovery Benchmark
======================================

This benchmark addresses the fundamental methodological issue in kuramoto.py and rossler.py:
those benchmarks construct features that explicitly encode the causal structure, reducing
causal discovery to variable selection.

This benchmark tests THREE conditions:

1. RAW OBSERVATIONS (principled test):
   - Input: θ(t) for Kuramoto, (x,y,z)(t) for Rössler, X(t) for VAR
   - No knowledge of functional form
   - Uses nonlinear CI tests (GPDC) to handle nonlinearity
   - TRUE TEST of causal discovery capability

2. ENGINEERED BASIS (oracle condition):
   - Input: Correct functional form sin(θ_j - θ_i) or (s_j - s_i)
   - Shows upper bound when functional form is known
   - Equivalent to the original benchmarks

3. WRONG BASIS (negative control):
   - Input: Incorrect functional form (polynomial instead of sine/diffusive)
   - Shows what happens when model assumptions are violated

SYSTEMS TESTED:
- Kuramoto: Nonlinear phase oscillators
- VAR: Linear vector autoregression (basis shouldn't help)
- Rössler: Chaotic 3D oscillators (future work)

"""

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import warnings
import sys
import os
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, accuracy_score

# Tigramite imports
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.gpdc import GPDC

# User Library Imports
from causationentropy import discover_network
from causationentropy.graph import pcmci_to_networkx

warnings.filterwarnings("ignore")


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# =============================================================================
# SIMULATION: KURAMOTO, VAR, RÖSSLER
# =============================================================================


def simulate_kuramoto(
    G, T, dt=0.05, rho=0.7, seed=0, omega_std=1.0, phase_noise_std=0.02, burn_in=50
):
    """Simulate Kuramoto model - returns phase trajectories."""
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes).astype(float)

    omega = rng.normal(loc=0.0, scale=omega_std, size=n)
    theta = rng.uniform(low=-np.pi, high=np.pi, size=n)

    total_steps = burn_in + T
    out = np.zeros((T, n), dtype=float)

    for step in range(total_steps):
        coupling = (A.T * np.sin(theta[None, :] - theta[:, None])).sum(axis=1)
        dtheta = omega + rho * coupling
        noise = rng.normal(loc=0.0, scale=phase_noise_std, size=n)
        theta = theta + dt * dtheta + noise
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        if step >= burn_in:
            out[step - burn_in] = theta

    return out


def simulate_var(
    G, T, dt=1.0, coupling_strength=0.3, seed=0, noise_std=0.5, burn_in=50
):
    """Simulate linear VAR(1) system: X(t+1) = A @ X(t) + noise"""
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    A_graph = nx.to_numpy_array(G, nodelist=nodes).astype(float)

    # VAR coefficient matrix: scaled adjacency + diagonal stability
    A = coupling_strength * A_graph
    np.fill_diagonal(A, 0.7)  # Autoregressive term for stability

    # Check stability
    eigvals = np.linalg.eigvals(A)
    if np.max(np.abs(eigvals)) >= 1.0:
        # Rescale to ensure stability
        A = A / (np.max(np.abs(eigvals)) + 0.1)

    total_steps = burn_in + T
    X = rng.normal(0, 1, size=n)
    out = np.zeros((T, n), dtype=float)

    for step in range(total_steps):
        X = A @ X + rng.normal(0, noise_std, size=n)
        if step >= burn_in:
            out[step - burn_in] = X

    return out


# =============================================================================
# DATA PREPARATION FUNCTIONS
# =============================================================================


def prepare_raw_observations(data):
    """
    RAW OBSERVATIONS: No feature engineering.
    Input: (T, n) array
    Output: Same array, variable names
    """
    T, n = data.shape
    var_names = [f"X{i}" for i in range(n)]
    return data, var_names


def prepare_kuramoto_oracle_basis(theta, dt):
    """
    ORACLE BASIS for Kuramoto: Uses correct functional form sin(θ_j - θ_i).
    This is what kuramoto.py does - encoding ground truth structure.
    """
    T, n = theta.shape

    # Compute velocities
    dtheta = (theta[1:] - theta[:-1]) / dt
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi  # wrap

    # Coupling terms: sin(θ_j(t) - θ_i(t))
    theta_prev = theta[:-2]  # align with lag-1 structure
    dtheta_next = dtheta[1:]  # targets at t+1

    S_cols = []
    basis_map = []
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            S_cols.append(np.sin(theta_prev[:, j] - theta_prev[:, i]))
            basis_map.append((i, j))

    S = np.stack(S_cols, axis=1)
    X = np.concatenate([dtheta_next, S], axis=1)

    var_names = [f"v{i}" for i in range(n)] + [f"sin_{i}<-{j}" for (i, j) in basis_map]
    meta = {"n": n, "basis_map": basis_map, "pred_offset": n}
    return X, var_names, meta


def prepare_kuramoto_wrong_basis(theta, dt):
    """
    WRONG BASIS for Kuramoto: Uses polynomial instead of sine.
    This tests what happens when model assumptions are wrong.
    """
    T, n = theta.shape

    dtheta = (theta[1:] - theta[:-1]) / dt
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

    theta_prev = theta[:-2]
    dtheta_next = dtheta[1:]

    # WRONG: Use polynomial differences instead of sine
    P_cols = []
    basis_map = []
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            diff = theta_prev[:, j] - theta_prev[:, i]
            # Polynomial basis: diff, diff^2
            P_cols.append(diff)
            P_cols.append(diff**2)
            basis_map.append((i, j))

    P = np.stack(P_cols, axis=1)
    X = np.concatenate([dtheta_next, P], axis=1)

    var_names = [f"v{i}" for i in range(n)]
    for i, j in basis_map:
        var_names.append(f"diff_{i}<-{j}")
        var_names.append(f"diff2_{i}<-{j}")

    meta = {"n": n, "basis_map": basis_map, "pred_offset": n}
    return X, var_names, meta


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================


def extract_adjacency_from_raw(graph_nx, n):
    """
    Extract adjacency from raw observation graph.
    graph_nx has n nodes (0 to n-1), edges indicate direct causation.
    """
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j and graph_nx.has_edge(j, i):  # j -> i
                A[j, i] = 1
    return A


def extract_adjacency_from_basis(graph_nx, n, basis_map, var_names):
    """
    Extract adjacency from basis-expanded graph.
    Works for both OCE (string nodes) and PCMCI (integer nodes).
    """
    A = np.zeros((n, n), dtype=int)

    # Determine if graph uses string or integer nodes
    sample_node = list(graph_nx.nodes())[0] if len(graph_nx.nodes()) > 0 else None
    is_string_nodes = isinstance(sample_node, str)

    if is_string_nodes:
        # OCE case: string variable names
        velocity_vars = [f"v{i}" for i in range(n)]
        for i in range(n):
            velocity_var = velocity_vars[i]
            for idx, (target, source) in enumerate(basis_map):
                # Find coupling variable - could be sin_, diff_, c_, etc.
                coupling_candidates = [
                    v for v in var_names if f"_{target}<-{source}" in v
                ]
                for coupling_var in coupling_candidates:
                    if graph_nx.has_edge(coupling_var, velocity_var):
                        edges = graph_nx[coupling_var][velocity_var]
                        for _, edata in edges.items():
                            if edata.get("lag", 0) <= 1:
                                if target == i:
                                    A[source, i] = 1
                                break
    else:
        # PCMCI case: integer node indices
        for i in range(n):
            velocity_idx = i
            for idx, (target, source) in enumerate(basis_map):
                coupling_idx = n + idx
                if graph_nx.has_edge(coupling_idx, velocity_idx):
                    if target == i:
                        A[source, i] = 1

    return A


# =============================================================================
# METRICS
# =============================================================================


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
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "TPR": tpr,
        "FPR": fpr,
        "Precision": precision,
        "F1": f1,
        "SHD": shd,
        "Accuracy": accuracy,
        "Time": time_taken,
    }


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_PARAMS = {
    "n_nodes": 3,
    "T": 500,  # Shorter for faster testing with GPDC (slow)
    "n_trials": 5,  # Reduced for faster testing
    "alpha": 0.05,
    "tau_max": 1,
    "dt": 0.05,
    "burn_in": 50,
}

EXPERIMENTS = {
    "Kuramoto_EdgeDensity": {
        "system": "kuramoto",
        "vary_param": "p_edge",
        "label": "Edge Probability",
        "values": [0.2, 0.4, 0.6],
        "defaults": {"rho": 0.7},
    },
    "VAR_EdgeDensity": {
        "system": "var",
        "vary_param": "p_edge",
        "label": "Edge Probability",
        "values": [0.2, 0.4, 0.6],
        "defaults": {"coupling_strength": 0.3},
    },
}

CONDITIONS = {
    "Raw_OCE": {
        "description": "Raw observations + OCE (Gaussian CI)",
        "method": "oce",
        "basis": "raw",
        "ci_test": "gaussian",
    },
    "Raw_PCMCI_ParCorr": {
        "description": "Raw observations + PCMCI (Partial Correlation)",
        "method": "pcmci",
        "basis": "raw",
        "ci_test": "parcorr",
    },
    "Raw_PCMCI_GPDC": {
        "description": "Raw observations + PCMCI (Gaussian Process)",
        "method": "pcmci",
        "basis": "raw",
        "ci_test": "gpdc",
    },
    "Oracle_OCE": {
        "description": "Oracle basis + OCE (Gaussian CI)",
        "method": "oce",
        "basis": "oracle",
        "ci_test": "gaussian",
    },
    "Oracle_PCMCI": {
        "description": "Oracle basis + PCMCI (Partial Correlation)",
        "method": "pcmci",
        "basis": "oracle",
        "ci_test": "parcorr",
    },
    "Wrong_OCE": {
        "description": "Wrong basis + OCE (Gaussian CI)",
        "method": "oce",
        "basis": "wrong",
        "ci_test": "gaussian",
    },
}


# =============================================================================
# MAIN BENCHMARK
# =============================================================================


def run_benchmark():
    os.makedirs("figs_principled", exist_ok=True)
    results = []

    print("=" * 80)
    print("PRINCIPLED CAUSAL DISCOVERY BENCHMARK")
    print("=" * 80)
    print(f"\nTesting {len(EXPERIMENTS)} systems × {len(CONDITIONS)} conditions")
    print(
        f"Nodes: {GLOBAL_PARAMS['n_nodes']}, Trials: {GLOBAL_PARAMS['n_trials']}, T: {GLOBAL_PARAMS['T']}\n"
    )

    total_runs = sum(
        len(exp_cfg["values"]) * GLOBAL_PARAMS["n_trials"] * len(CONDITIONS)
        for exp_cfg in EXPERIMENTS.values()
    )
    pbar = tqdm(total=total_runs, desc="Overall progress")

    for exp_name, exp_cfg in EXPERIMENTS.items():
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"{'='*80}")

        system = exp_cfg["system"]
        param_name = exp_cfg["vary_param"]
        param_values = exp_cfg["values"]
        defaults = exp_cfg["defaults"]

        for param_val in param_values:
            current_params = defaults.copy()
            current_params[param_name] = param_val

            for trial in range(GLOBAL_PARAMS["n_trials"]):
                seed = 42 + trial * 100

                # Generate graph
                if param_name == "p_edge":
                    p = param_val
                    G = nx.erdos_renyi_graph(
                        GLOBAL_PARAMS["n_nodes"], p, seed=seed, directed=True
                    )
                else:
                    p = 0.3
                    G = nx.erdos_renyi_graph(
                        GLOBAL_PARAMS["n_nodes"], p, seed=seed, directed=True
                    )

                true_adj = nx.to_numpy_array(G).astype(int)

                # Simulate system
                if system == "kuramoto":
                    data = simulate_kuramoto(
                        G,
                        GLOBAL_PARAMS["T"],
                        dt=GLOBAL_PARAMS["dt"],
                        rho=current_params.get("rho", 0.7),
                        seed=seed,
                        burn_in=GLOBAL_PARAMS["burn_in"],
                    )
                elif system == "var":
                    data = simulate_var(
                        G,
                        GLOBAL_PARAMS["T"],
                        dt=1.0,
                        coupling_strength=current_params.get("coupling_strength", 0.3),
                        seed=seed,
                        burn_in=GLOBAL_PARAMS["burn_in"],
                    )
                else:
                    raise ValueError(f"Unknown system: {system}")

                # Test each condition
                for cond_name, cond_cfg in CONDITIONS.items():
                    # Skip wrong basis for VAR (doesn't make sense)
                    if system == "var" and cond_cfg["basis"] == "wrong":
                        pbar.update(1)
                        continue

                    # Skip oracle basis for VAR (VAR is already linear, oracle doesn't apply)
                    if system == "var" and cond_cfg["basis"] == "oracle":
                        pbar.update(1)
                        continue

                    try:
                        start_time = time.time()

                        # Prepare data based on basis type
                        if cond_cfg["basis"] == "raw":
                            X, var_names = prepare_raw_observations(data)
                            meta = {"n": GLOBAL_PARAMS["n_nodes"]}
                        elif cond_cfg["basis"] == "oracle" and system == "kuramoto":
                            X, var_names, meta = prepare_kuramoto_oracle_basis(
                                data, GLOBAL_PARAMS["dt"]
                            )
                        elif cond_cfg["basis"] == "wrong" and system == "kuramoto":
                            X, var_names, meta = prepare_kuramoto_wrong_basis(
                                data, GLOBAL_PARAMS["dt"]
                            )
                        else:
                            pbar.update(1)
                            continue

                        # Run causal discovery
                        if cond_cfg["method"] == "oce":
                            X_df = pd.DataFrame(X, columns=var_names)
                            with suppress_stdout():
                                graph_nx = discover_network(
                                    data=X_df,
                                    max_lag=GLOBAL_PARAMS["tau_max"],
                                    method="standard",
                                    information=cond_cfg["ci_test"],
                                )
                        elif cond_cfg["method"] == "pcmci":
                            dataframe = pp.DataFrame(
                                X,
                                datatime={0: np.arange(X.shape[0])},
                                var_names=var_names,
                            )

                            if cond_cfg["ci_test"] == "parcorr":
                                ci_test = ParCorr()
                            elif cond_cfg["ci_test"] == "gpdc":
                                ci_test = GPDC(significance="analytic")
                            else:
                                raise ValueError(
                                    f"Unknown CI test: {cond_cfg['ci_test']}"
                                )

                            pcmci = PCMCI(
                                dataframe=dataframe, cond_ind_test=ci_test, verbosity=0
                            )
                            pcmci_res = pcmci.run_pcmci(
                                tau_min=1,
                                tau_max=GLOBAL_PARAMS["tau_max"],
                                pc_alpha=GLOBAL_PARAMS["alpha"],
                            )
                            graph_nx = pcmci_to_networkx(pcmci_res)

                        # Extract adjacency
                        if cond_cfg["basis"] == "raw":
                            pred_adj = extract_adjacency_from_raw(graph_nx, meta["n"])
                        else:
                            pred_adj = extract_adjacency_from_basis(
                                graph_nx, meta["n"], meta["basis_map"], var_names
                            )

                        elapsed = time.time() - start_time

                        # Compute metrics
                        metrics = compute_metrics(pred_adj, true_adj, elapsed)
                        metrics.update(
                            {
                                "System": system,
                                "Experiment": exp_name,
                                "Parameter": param_val,
                                "Condition": cond_name,
                                "Basis": cond_cfg["basis"],
                                "Method": cond_cfg["method"],
                                "CI_Test": cond_cfg["ci_test"],
                                "Trial": trial,
                            }
                        )
                        results.append(metrics)

                    except Exception as e:
                        print(f"\nERROR in {cond_name}: {e}")
                        # Log failure
                        results.append(
                            {
                                "System": system,
                                "Experiment": exp_name,
                                "Parameter": param_val,
                                "Condition": cond_name,
                                "Basis": cond_cfg["basis"],
                                "Method": cond_cfg["method"],
                                "CI_Test": cond_cfg["ci_test"],
                                "Trial": trial,
                                "F1": 0.0,
                                "TPR": 0.0,
                                "FPR": 1.0,
                                "Precision": 0.0,
                                "SHD": true_adj.sum(),
                                "Accuracy": 0.0,
                                "Time": 0.0,
                            }
                        )

                    pbar.update(1)

    pbar.close()
    return pd.DataFrame(results)


# =============================================================================
# ANALYSIS & PLOTTING
# =============================================================================


def analyze_results(df_results):
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Overall summary by condition
    summary = df_results.groupby(["System", "Condition"])[
        ["F1", "TPR", "FPR", "Accuracy", "Time"]
    ].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    print("\n" + "-" * 80)
    print("OVERALL PERFORMANCE BY CONDITION")
    print("-" * 80)
    print(summary.to_string(index=False))

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    for system in df_results["System"].unique():
        sys_data = df_results[df_results["System"] == system]
        print(f"\n{system.upper()}:")

        raw_f1 = sys_data[sys_data["Basis"] == "raw"]["F1"].mean()
        oracle_f1 = sys_data[sys_data["Basis"] == "oracle"]["F1"].mean()
        wrong_f1 = (
            sys_data[sys_data["Basis"] == "wrong"]["F1"].mean()
            if "wrong" in sys_data["Basis"].values
            else None
        )

        print(f"  Raw observations: F1 = {raw_f1:.3f}")
        if not np.isnan(oracle_f1):
            print(
                f"  Oracle basis:     F1 = {oracle_f1:.3f} (+{oracle_f1 - raw_f1:.3f} vs raw)"
            )
        if wrong_f1 is not None and not np.isnan(wrong_f1):
            print(
                f"  Wrong basis:      F1 = {wrong_f1:.3f} ({wrong_f1 - raw_f1:.3f} vs raw)"
            )

        if not np.isnan(oracle_f1) and oracle_f1 > raw_f1 + 0.1:
            print(f"  → Oracle basis provides {(oracle_f1 - raw_f1):.1%} improvement")
            print(f"  → This indicates that knowing the functional form is CRITICAL")
        else:
            print(f"  → Oracle basis provides minimal benefit")
            print(f"  → Methods can handle nonlinearity without feature engineering")


def plot_results(df_results):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

    systems = df_results["System"].unique()

    for system in systems:
        sys_data = df_results[df_results["System"] == system]
        experiments = sys_data["Experiment"].unique()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for exp_idx, exp_name in enumerate(experiments):
            exp_data = sys_data[sys_data["Experiment"] == exp_name]

            # F1 Score
            ax = axes[exp_idx * 2]
            sns.lineplot(
                data=exp_data,
                x="Parameter",
                y="F1",
                hue="Condition",
                style="Basis",
                markers=True,
                dashes=False,
                ax=ax,
                ci=68,
            )
            ax.set_title(f"{exp_name}: F1 Score")
            ax.set_xlabel("Parameter Value")
            ax.set_ylabel("F1")
            ax.legend(fontsize=7, loc="best")

            # TPR
            ax = axes[exp_idx * 2 + 1]
            sns.lineplot(
                data=exp_data,
                x="Parameter",
                y="TPR",
                hue="Condition",
                style="Basis",
                markers=True,
                dashes=False,
                ax=ax,
                ci=68,
            )
            ax.set_title(f"{exp_name}: True Positive Rate")
            ax.set_xlabel("Parameter Value")
            ax.set_ylabel("TPR")
            ax.legend(fontsize=7, loc="best")

        plt.tight_layout()
        plt.savefig(
            f"figs_principled/{system}_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"\nPlot saved: figs_principled/{system}_comparison.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PRINCIPLED CAUSAL DISCOVERY BENCHMARK")
    print("Testing causal discovery WITHOUT pre-encoding the answer")
    print("=" * 80 + "\n")

    df_results = run_benchmark()

    # Save results
    df_results.to_csv("principled_results.csv", index=False)
    print("\n✓ Results saved to principled_results.csv")

    # Analyze
    analyze_results(df_results)

    # Plot
    plot_results(df_results)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
