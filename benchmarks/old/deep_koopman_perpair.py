"""
Per-Pair Deep Koopman for Causal Discovery on Rössler Oscillators.

Faithful implementation of the Kausal paper approach:
  "Koopman-based approach for causal discovery" (Nat. Comm. Phys., 2025)

Key difference from deep_koopman_kausal.py:
  Trains SEPARATE MLP encoders for each candidate cause-effect pair,
  rather than reusing a shared global encoder. Each pair gets its own
  learned observables, preventing indirect causal paths from leaking
  through a shared representation.

For each candidate pair (j→i):
  - Marginal model: MLP(x_i) → M features, Koopman K predicts next features
  - Joint model:    MLP([x_i, x_j]) → M features, Koopman K predicts next features
  Compare prediction quality on held-out test set via bootstrap test.

Methods compared:
  1. Kausal (per-pair): Per-pair MLPs + Koopman K, bootstrap test
  2. Linear Granger:    Pairwise DMD in raw state space (no MLP)
  3. PCMCI (raw):       Tigramite PCMCI on derivative basis
  4. oCSE (raw):        oCSE on derivative basis
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

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from causationentropy import discover_network
from causationentropy.graph import pcmci_to_networkx
from causationentropy.datasets.synthetic import (
    generate_graph_topology,
    simulate_rossler,
    prepare_rossler_data_for_causal_discovery,
)

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# PER-PAIR KOOPMAN MODEL
# =============================================================================


class PairKoopmanModel(nn.Module):
    """Koopman model for a single causal pair test.

    Marginal (input_dim=3): encoder sees only node i's state.
    Joint (input_dim=6): encoder sees [node i, node j].
    Decoder always predicts node i's 3D state.
    """

    def __init__(self, input_dim, M=32, target_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, M),
        )
        self.K = nn.Parameter(torch.randn(M, M) * 0.01)
        self.decoder = nn.Sequential(
            nn.Linear(M, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, target_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def predict_next(self, x_t):
        z_t = self.encode(x_t)
        z_next = z_t @ self.K.T
        return self.decode(z_next), z_t, z_next


def train_pair_model(
    x_t,
    x_tp1_enc,
    target_t,
    target_tp1,
    input_dim,
    M=32,
    target_dim=3,
    epochs=500,
    lr=1e-3,
    batch_size=128,
    lambda_pred=1.0,
    lambda_lin=0.5,
    lambda_recon=0.5,
    weight_decay=1e-4,
):
    """Train one per-pair Koopman model.

    Args:
        x_t:         (T, input_dim) encoder input at time t
        x_tp1_enc:   (T, input_dim) encoder input at t+1 (linearity loss)
        target_t:    (T, target_dim) reconstruction target (x_i at t)
        target_tp1:  (T, target_dim) prediction target (x_i at t+1)
    """
    ds = TensorDataset(
        torch.tensor(x_t, dtype=torch.float32),
        torch.tensor(x_tp1_enc, dtype=torch.float32),
        torch.tensor(target_t, dtype=torch.float32),
        torch.tensor(target_tp1, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = PairKoopmanModel(input_dim, M, target_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        for b_xt, b_xtp1, b_tgt_t, b_tgt_tp1 in loader:
            b_xt = b_xt.to(device)
            b_xtp1 = b_xtp1.to(device)
            b_tgt_t = b_tgt_t.to(device)
            b_tgt_tp1 = b_tgt_tp1.to(device)

            z_t = model.encode(b_xt)
            z_next = z_t @ model.K.T

            # Reconstruction: decoder(encoder(x_t)) ≈ x_i(t)
            L_recon = torch.mean((model.decode(z_t) - b_tgt_t) ** 2)

            # Prediction: decoder(K · encoder(x_t)) ≈ x_i(t+1)
            L_pred = torch.mean((model.decode(z_next) - b_tgt_tp1) ** 2)

            # Linearity: K · encoder(x_t) ≈ encoder(x_{t+1})
            z_tp1 = model.encode(b_xtp1)
            L_lin = torch.mean((z_next - z_tp1.detach()) ** 2)

            loss = lambda_recon * L_recon + lambda_pred * L_pred + lambda_lin * L_lin

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


def prediction_errors(model, x_t, target_tp1):
    """Per-timestep prediction MSE on test data."""
    model.eval()
    with torch.no_grad():
        pred, _, _ = model.predict_next(
            torch.tensor(x_t, dtype=torch.float32).to(device)
        )
        pred = pred.cpu().numpy()
    return np.sum((target_tp1 - pred) ** 2, axis=1)


# =============================================================================
# PER-PAIR KAUSAL TEST
# =============================================================================


def kausal_perpair_test(
    traj_nodes_norm,
    n_nodes,
    M=32,
    epochs=500,
    lr=1e-3,
    batch_size=128,
    lambda_pred=1.0,
    lambda_lin=0.5,
    lambda_recon=0.5,
    n_bootstrap=200,
    p_crit=0.05,
    train_frac=0.8,
):
    """Per-pair Koopman Granger causality test (Kausal paper, faithful).

    For each pair (j→i):
      1. Marginal: train MLP_i on x_i → predict x_i(t+1)
      2. Joint:    train MLP_ji on [x_i, x_j] → predict x_i(t+1)
      3. Compare held-out prediction errors via bootstrap test.
    """
    T = traj_nodes_norm[0].shape[0]
    split = int((T - 1) * train_frac)

    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    pvals = np.ones((n_nodes, n_nodes))
    delta_means = np.zeros((n_nodes, n_nodes))

    # --- Train marginal models (one per target node) ---
    marg_errors_test = []
    for i in range(n_nodes):
        xi = traj_nodes_norm[i]
        xi_t, xi_tp1 = xi[:-1], xi[1:]

        model = train_pair_model(
            x_t=xi_t[:split],
            x_tp1_enc=xi_tp1[:split],
            target_t=xi_t[:split],
            target_tp1=xi_tp1[:split],
            input_dim=3,
            M=M,
            target_dim=3,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            lambda_pred=lambda_pred,
            lambda_lin=lambda_lin,
            lambda_recon=lambda_recon,
        )
        errs = prediction_errors(model, xi_t[split:], xi_tp1[split:])
        marg_errors_test.append(errs)

    # --- Train joint models & compare ---
    for i in range(n_nodes):
        xi = traj_nodes_norm[i]
        xi_t, xi_tp1 = xi[:-1], xi[1:]

        for j in range(n_nodes):
            if i == j:
                continue

            xj = traj_nodes_norm[j]
            xj_t, xj_tp1 = xj[:-1], xj[1:]

            joint_t = np.concatenate([xi_t, xj_t], axis=1)
            joint_tp1 = np.concatenate([xi_tp1, xj_tp1], axis=1)

            model = train_pair_model(
                x_t=joint_t[:split],
                x_tp1_enc=joint_tp1[:split],
                target_t=xi_t[:split],
                target_tp1=xi_tp1[:split],
                input_dim=6,
                M=M,
                target_dim=3,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                lambda_pred=lambda_pred,
                lambda_lin=lambda_lin,
                lambda_recon=lambda_recon,
            )
            errs_joint = prediction_errors(model, joint_t[split:], xi_tp1[split:])
            errs_marg = marg_errors_test[i]

            delta = errs_marg - errs_joint
            T_eff = len(delta)

            rng = np.random.default_rng(42 + i * n_nodes + j)
            boot = np.array(
                [
                    delta[rng.choice(T_eff, size=T_eff, replace=True)].mean()
                    for _ in range(n_bootstrap)
                ]
            )
            p_val = (np.sum(boot <= 0) + 1) / (n_bootstrap + 1)

            delta_means[j, i] = delta.mean()
            pvals[j, i] = p_val
            if p_val < p_crit:
                adj[j, i] = 1

    return adj, pvals, delta_means


# =============================================================================
# LINEAR GRANGER (DMD in raw state space, no MLP)
# =============================================================================


def linear_granger_test(
    traj_nodes_norm,
    n_nodes,
    n_bootstrap=200,
    p_crit=0.05,
    train_frac=0.8,
):
    """Pairwise Granger causality via DMD in raw state space."""
    T = traj_nodes_norm[0].shape[0]
    split = int((T - 1) * train_frac)

    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    pvals = np.ones((n_nodes, n_nodes))

    marg_errors = []
    for i in range(n_nodes):
        xi = traj_nodes_norm[i]
        xi_t, xi_tp1 = xi[:-1], xi[1:]

        X_tr, X_te = xi_t[:split].T, xi_t[split:].T
        Y_tr, Y_te = xi_tp1[:split].T, xi_tp1[split:].T

        K = Y_tr @ np.linalg.pinv(X_tr)
        marg_errors.append(np.sum((Y_te - K @ X_te) ** 2, axis=0))

    for i in range(n_nodes):
        xi = traj_nodes_norm[i]
        xi_t, xi_tp1 = xi[:-1], xi[1:]

        for j in range(n_nodes):
            if i == j:
                continue

            xj = traj_nodes_norm[j]
            xj_t = xj[:-1]

            joint_t = np.concatenate([xi_t, xj_t], axis=1)

            X_tr, X_te = joint_t[:split].T, joint_t[split:].T
            Y_tr, Y_te = xi_tp1[:split].T, xi_tp1[split:].T

            K = Y_tr @ np.linalg.pinv(X_tr)
            errs_joint = np.sum((Y_te - K @ X_te) ** 2, axis=0)

            delta = marg_errors[i] - errs_joint
            T_eff = len(delta)
            rng = np.random.default_rng(42 + i * n_nodes + j)
            boot = np.array(
                [
                    delta[rng.choice(T_eff, size=T_eff, replace=True)].mean()
                    for _ in range(n_bootstrap)
                ]
            )
            p_val = (np.sum(boot <= 0) + 1) / (n_bootstrap + 1)

            pvals[j, i] = p_val
            if p_val < p_crit:
                adj[j, i] = 1

    return adj, pvals


# =============================================================================
# BASELINE HELPERS
# =============================================================================


def extract_node_graph_pcmci(graph_nx, n, basis_map, coupling_on="x"):
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n))
    target_start = {"x": 0, "y": n, "z": 2 * n}[coupling_on]
    coupling_start = 3 * n
    for i in range(n):
        target_node_idx = target_start + i
        for idx, (target, source) in enumerate(basis_map):
            coupling_node_idx = coupling_start + idx
            if graph_nx.has_edge(coupling_node_idx, target_node_idx):
                edges = graph_nx[coupling_node_idx][target_node_idx]
                for _, edata in edges.items():
                    if target == i:
                        G.add_edge(source, i)
    return G


def extract_node_graph_oce(graph_nx, n, basis_map, var_names, coupling_on="x"):
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n))
    target_vars = [f"d{coupling_on}{i}" for i in range(n)]
    base = 3 * n
    for i in range(n):
        target_var = target_vars[i]
        for idx, (target, source) in enumerate(basis_map):
            coupling_var = var_names[base + idx]
            if graph_nx.has_edge(coupling_var, target_var):
                edges = graph_nx[coupling_var][target_var]
                for _, edata in edges.items():
                    if target == i:
                        G.add_edge(source, i)
    return G


def collapse_to_binary_adjacency(node_graph, n):
    A = np.zeros((n, n), dtype=int)
    for u, v, _ in node_graph.edges(data=True):
        A[u, v] = 1
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

    total = tp + fp + tn + fn
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * (precision * tpr) / (precision + tpr)
        if (precision + tpr) > 0
        else 0.0
    )
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TPR": tpr,
        "FPR": fpr,
        "Precision": precision,
        "F1": f1,
        "Accuracy": accuracy,
        "Time": time_taken,
    }


# =============================================================================
# HELPERS
# =============================================================================


def extract_node_trajectories(traj_sub, n_nodes):
    """(T, n, 3) → list of (T, 3) per-node arrays."""
    return [traj_sub[:, i, :] for i in range(n_nodes)]


def normalize_node_trajectories(traj_nodes):
    """Independently normalize each node's trajectory to zero-mean unit-var."""
    normed = []
    for xi in traj_nodes:
        mu = xi.mean(axis=0)
        sigma = xi.std(axis=0)
        sigma[sigma < 1e-8] = 1.0
        normed.append((xi - mu) / sigma)
    return normed


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_PARAMS = {
    "n_nodes": 5,
    "T": 5000,
    "n_trials": 3,
    "dt": 0.02,
    "subsample": 5,
    "burn_in": 500,
    # Rössler
    "a": 0.2,
    "b": 0.2,
    "c": 5.7,
    "rho_default": 0.4,
    "noise_std": 0.0,
    "init_scale": 1.0,
    "coupling_on": "x",
    "normalize_by_indegree": False,
    # PCMCI / oCSE
    "alpha": 0.05,
    "tau_max": 1,
    # Per-pair Koopman
    "M": 32,
    "epochs_perpair": 500,
    "lr": 1e-3,
    "batch_size": 128,
    "lambda_pred": 1.0,
    "lambda_lin": 0.5,
    "lambda_recon": 0.5,
    "train_frac": 0.8,
    # Bootstrap test
    "n_bootstrap": 200,
    "p_crit": 0.05,
}

EXPERIMENTS = {
    "ER_Density": {
        "type": "Erdos-Renyi",
        "vary_param": "p_edge",
        "label": "Edge Probability (p)",
        "values": [0.2, 0.4, 0.6],
        "defaults": {"rho": 0.2},
    },
}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = []

    n_nodes = GLOBAL_PARAMS["n_nodes"]
    M = GLOBAL_PARAMS["M"]
    n_pairs = n_nodes * (n_nodes - 1)

    print("=" * 80)
    print("Per-Pair Koopman Causal Discovery — Rössler Oscillators")
    print(f"Device: {device}")
    print(f"{n_nodes} nodes, M={M} features/model, {n_pairs} pair tests per trial")
    print(f"{GLOBAL_PARAMS['n_trials']} trials per condition")
    print("=" * 80)

    total_iters = sum(
        len(cfg["values"]) * GLOBAL_PARAMS["n_trials"]
        for cfg in EXPERIMENTS.values()
    )
    pbar = tqdm(total=total_iters, desc="Total progress")
    os.makedirs("benchmarks/results", exist_ok=True)

    for exp_name, config in EXPERIMENTS.items():
        print(f"\n--- Experiment: {exp_name} ---")

        param_name = config["vary_param"]
        param_values = config["values"]
        defaults = config["defaults"]

        for val in param_values:
            current_params = defaults.copy()
            current_params[param_name] = val

            for trial in range(GLOBAL_PARAMS["n_trials"]):
                seed = 42 + (trial * 100)
                print(f"\n  {config['label']}={val}, Trial {trial}, seed={seed}")

                torch.manual_seed(seed)
                np.random.seed(seed)

                # Generate graph and simulate
                G = generate_graph_topology(
                    config["type"], n_nodes, current_params, seed
                )
                true_adj = nx.to_numpy_array(G).astype(int)

                traj, _ = simulate_rossler(
                    G=G,
                    T=GLOBAL_PARAMS["T"],
                    dt=GLOBAL_PARAMS["dt"],
                    rho=current_params.get("rho", GLOBAL_PARAMS["rho_default"]),
                    seed=seed,
                    a=GLOBAL_PARAMS["a"],
                    b=GLOBAL_PARAMS["b"],
                    c=GLOBAL_PARAMS["c"],
                    init_scale=GLOBAL_PARAMS["init_scale"],
                    noise_std=GLOBAL_PARAMS["noise_std"],
                    burn_in=GLOBAL_PARAMS["burn_in"],
                    normalize_by_indegree=GLOBAL_PARAMS["normalize_by_indegree"],
                    coupling_on=GLOBAL_PARAMS["coupling_on"],
                )

                k = GLOBAL_PARAMS["subsample"]
                traj_sub = traj[::k] if k > 1 else traj
                dt_eff = GLOBAL_PARAMS["dt"] * k

                # Prepare per-node normalized trajectories
                traj_nodes = extract_node_trajectories(traj_sub, n_nodes)
                traj_nodes_norm = normalize_node_trajectories(traj_nodes)

                n_true = int(true_adj.sum())
                n_possible = n_nodes * (n_nodes - 1)
                print(f"  True edges: {n_true}/{n_possible}")

                # ===========================================================
                # METHOD 1: Kausal (per-pair) — THE MAIN METHOD
                # ===========================================================
                print(f"  Training {n_nodes + n_pairs} per-pair models (M={M})...")
                start_time = time.time()

                pred_adj_pp, pvals_pp, deltas_pp = kausal_perpair_test(
                    traj_nodes_norm,
                    n_nodes,
                    M=GLOBAL_PARAMS["M"],
                    epochs=GLOBAL_PARAMS["epochs_perpair"],
                    lr=GLOBAL_PARAMS["lr"],
                    batch_size=GLOBAL_PARAMS["batch_size"],
                    lambda_pred=GLOBAL_PARAMS["lambda_pred"],
                    lambda_lin=GLOBAL_PARAMS["lambda_lin"],
                    lambda_recon=GLOBAL_PARAMS["lambda_recon"],
                    n_bootstrap=GLOBAL_PARAMS["n_bootstrap"],
                    p_crit=GLOBAL_PARAMS["p_crit"],
                    train_frac=GLOBAL_PARAMS["train_frac"],
                )
                pp_time = time.time() - start_time

                m_pp = compute_metrics(pred_adj_pp, true_adj, pp_time)
                m_pp.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "Kausal (per-pair)",
                    "Trial": trial,
                })
                results.append(m_pp)
                print(
                    f"  Per-pair   — F1: {m_pp['F1']:.3f}, "
                    f"TP: {m_pp['TP']}, FP: {m_pp['FP']}, "
                    f"TPR: {m_pp['TPR']:.2f}, FPR: {m_pp['FPR']:.2f}, "
                    f"Acc: {m_pp['Accuracy']:.3f}  ({pp_time:.0f}s)"
                )

                # ===========================================================
                # METHOD 2: Linear Granger (DMD, no MLP)
                # ===========================================================
                print("  Running Linear Granger (DMD)...")
                start_time = time.time()

                pred_adj_lg, pvals_lg = linear_granger_test(
                    traj_nodes_norm,
                    n_nodes,
                    n_bootstrap=GLOBAL_PARAMS["n_bootstrap"],
                    p_crit=GLOBAL_PARAMS["p_crit"],
                    train_frac=GLOBAL_PARAMS["train_frac"],
                )
                lg_time = time.time() - start_time

                m_lg = compute_metrics(pred_adj_lg, true_adj, lg_time)
                m_lg.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "Linear Granger",
                    "Trial": trial,
                })
                results.append(m_lg)
                print(
                    f"  Lin Grang  — F1: {m_lg['F1']:.3f}, "
                    f"TP: {m_lg['TP']}, FP: {m_lg['FP']}, "
                    f"TPR: {m_lg['TPR']:.2f}, FPR: {m_lg['FPR']:.2f}, "
                    f"Acc: {m_lg['Accuracy']:.3f}  ({lg_time:.1f}s)"
                )

                # ===========================================================
                # METHOD 3: Raw PCMCI baseline
                # ===========================================================
                print("  Running PCMCI baseline (raw)...")
                start_time = time.time()

                X_basis, basis_meta, var_names = (
                    prepare_rossler_data_for_causal_discovery(
                        traj_sub,
                        dt=dt_eff,
                        coupling_on=GLOBAL_PARAMS["coupling_on"],
                    )
                )
                T_eff = X_basis.shape[0]
                dataframe = pp.DataFrame(
                    X_basis,
                    datatime={0: np.arange(T_eff)},
                    var_names=var_names,
                )
                pcmci = PCMCI(
                    dataframe=dataframe,
                    cond_ind_test=ParCorr(),
                    verbosity=0,
                )
                pcmci_res = pcmci.run_pcmci(
                    tau_min=1,
                    tau_max=GLOBAL_PARAMS["tau_max"],
                    pc_alpha=GLOBAL_PARAMS["alpha"],
                )
                graph_nx = pcmci_to_networkx(pcmci_res)
                pcmci_node_graph = extract_node_graph_pcmci(
                    graph_nx,
                    basis_meta["n"],
                    basis_meta["basis_map"],
                    coupling_on=GLOBAL_PARAMS["coupling_on"],
                )
                pred_adj_pcmci = collapse_to_binary_adjacency(
                    pcmci_node_graph, n_nodes
                )
                pcmci_time = time.time() - start_time

                m_pcmci = compute_metrics(pred_adj_pcmci, true_adj, pcmci_time)
                m_pcmci.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "PCMCI (raw)",
                    "Trial": trial,
                })
                results.append(m_pcmci)
                print(
                    f"  PCMCI raw  — F1: {m_pcmci['F1']:.3f}, "
                    f"TP: {m_pcmci['TP']}, FP: {m_pcmci['FP']}, "
                    f"TPR: {m_pcmci['TPR']:.2f}, FPR: {m_pcmci['FPR']:.2f}, "
                    f"Acc: {m_pcmci['Accuracy']:.3f}  ({pcmci_time:.1f}s)"
                )

                # ===========================================================
                # METHOD 4: Raw oCSE baseline
                # ===========================================================
                print("  Running oCSE baseline (raw)...")
                start_time = time.time()

                X_df = pd.DataFrame(X_basis, columns=var_names)
                with suppress_stdout():
                    oce_network = discover_network(
                        data=X_df,
                        max_lag=GLOBAL_PARAMS["tau_max"],
                        method="standard",
                        information="gaussian",
                        alpha_forward=GLOBAL_PARAMS["alpha"],
                        alpha_backward=GLOBAL_PARAMS["alpha"],
                    )

                oce_node_graph = extract_node_graph_oce(
                    oce_network,
                    n_nodes,
                    basis_meta["basis_map"],
                    var_names,
                    coupling_on=GLOBAL_PARAMS["coupling_on"],
                )
                pred_adj_oce = collapse_to_binary_adjacency(
                    oce_node_graph, n_nodes
                )
                oce_time = time.time() - start_time

                m_oce = compute_metrics(pred_adj_oce, true_adj, oce_time)
                m_oce.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "oCSE (raw)",
                    "Trial": trial,
                })
                results.append(m_oce)
                print(
                    f"  oCSE raw   — F1: {m_oce['F1']:.3f}, "
                    f"TP: {m_oce['TP']}, FP: {m_oce['FP']}, "
                    f"TPR: {m_oce['TPR']:.2f}, FPR: {m_oce['FPR']:.2f}, "
                    f"Acc: {m_oce['Accuracy']:.3f}  ({oce_time:.1f}s)"
                )

                pbar.update(1)

    pbar.close()

    # =================================================================
    # SAVE
    # =================================================================

    df_results = pd.DataFrame(results)
    csv_path = "benchmarks/results/deep_koopman_perpair_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # =================================================================
    # SUMMARY
    # =================================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    summary = (
        df_results.groupby(["Experiment", "Method"])[
            ["F1", "TP", "FP", "FN", "Accuracy", "TPR", "FPR", "Precision", "Time"]
        ]
        .mean()
        .reset_index()
        .sort_values(["Experiment", "F1"], ascending=[True, False])
    )

    for exp_name in EXPERIMENTS:
        print(f"\nExperiment: {exp_name}")
        print("-" * 80)
        exp_summary = summary[summary["Experiment"] == exp_name]
        print(
            exp_summary[
                ["Method", "F1", "TP", "FP", "Accuracy", "TPR", "FPR", "Time"]
            ].to_string(index=False, float_format="%.3f")
        )

    # Reference: shared-encoder Kausal from previous benchmark
    prev_csv = "benchmarks/results/deep_koopman_kausal_results.csv"
    if os.path.exists(prev_csv):
        df_prev = pd.read_csv(prev_csv)
        kausal_shared = df_prev[df_prev["Method"] == "Koopman + Kausal"]
        if len(kausal_shared) > 0:
            print("\n--- Comparison: Shared-encoder Kausal (from previous run) ---")
            print(
                f"  Shared Kausal avg: "
                f"F1={kausal_shared['F1'].mean():.3f}, "
                f"TPR={kausal_shared['TPR'].mean():.3f}, "
                f"FPR={kausal_shared['FPR'].mean():.3f}, "
                f"Acc={kausal_shared['Accuracy'].mean():.3f}"
            )

    # =================================================================
    # PLOTTING
    # =================================================================

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for exp_name, config in EXPERIMENTS.items():
        exp_data = df_results[df_results["Experiment"] == exp_name]

        for idx, (metric, label) in enumerate(
            [("F1", "F1 Score"), ("TPR", "True Positive Rate"), ("FPR", "False Positive Rate")]
        ):
            sns.lineplot(
                data=exp_data,
                x="Parameter",
                y=metric,
                hue="Method",
                style="Method",
                markers=True,
                dashes=False,
                ax=axes[idx],
                errorbar=("ci", 68),
            )
            axes[idx].set_title(f"{exp_name}: {label}")
            axes[idx].set_xlabel(config["label"])
            axes[idx].set_ylabel(metric)

    plt.tight_layout()
    plot_path = "benchmarks/results/deep_koopman_perpair_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}")
