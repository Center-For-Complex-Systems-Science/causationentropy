"""
Causal Effects Analysis
=======================
Converts the discovered causal network into a time-unrolled DAG (pgmpy),
applies the backdoor criterion to find minimal adjustment sets, and
computes the partial correlation to determine the sign and significance
of the causal effect: Pro-Reg TikTok → Background Checks.
"""

import matplotlib
matplotlib.use("Agg")

from itertools import product

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pingouin as pg
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference.CausalInference import CausalInference


# ── 1. Build lagged DAG from causal relations ────────────────────────

def build_lagged_dag(variables, max_lag, relations):
    """
    Construct a time-unrolled DAG from a list of lagged causal relations.

    Each relation (source, lag_offset, target) means source_{t-lag} -> target_t.
    The DAG is unrolled across all valid time shifts up to max_lag.
    """
    nodes = [
        f"{v}_t" if lag == 0 else f"{v}_t-{lag}"
        for v, lag in product(variables, range(max_lag + 1))
    ]

    edges = []
    for source, lag_offset, target in relations:
        for shift in range(max_lag - lag_offset + 1):
            src_lag = lag_offset + shift
            tgt_lag = shift

            src = f"{source}_t-{src_lag}"
            tgt = f"{target}_t" if tgt_lag == 0 else f"{target}_t-{tgt_lag}"
            edges.append((src, tgt))

    model = DiscreteBayesianNetwork()
    model.add_nodes_from(nodes)
    model.add_edges_from(edges)
    return model


# ── 2. Visualize the unrolled DAG ────────────────────────────────────

def plot_lagged_dag(model, figsize=(14, 12), save_path=None):
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())

    plt.figure(figsize=figsize)
    nx.draw_circular(
        G,
        with_labels=True,
        node_size=2000,
        node_color="lightblue",
        arrows=True,
        font_size=8,
    )
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


# ── 3. Backdoor adjustment ───────────────────────────────────────────

def get_minimal_adjustment_set(model, X, Y):
    ci = CausalInference(model)
    return ci.get_minimal_adjustment_set(X=X, Y=Y)


# ── 4. Load data (same pipeline as tiktok_analysis.py) ───────────────

def load_merged_data():
    df = pd.read_csv("data/all_dailies_reg.csv")
    proquest_df = pd.read_csv("data/proquest.csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.floor("D")
    proquest_df["PubDate"] = pd.to_datetime(proquest_df["PubDate"])
    proquest_df["date"] = proquest_df["PubDate"].dt.floor("D")

    articles_per_day = (
        proquest_df.groupby("date").size().reset_index(name="Media Articles")
    )

    merged = (
        df.merge(articles_per_day, on="date", how="left")
          .drop(columns=["PubDate"], errors="ignore")
          .rename(columns={
              "bg_checks": "Background Checks",
              "anti_reg": "Anti-Reg",
              "pro_reg": "Pro-Reg",
          })
    )
    merged["Media Articles"] = merged["Media Articles"].fillna(0)
    merged = merged.set_index("date").sort_index().asfreq("D")

    decomp = seasonal_decompose(
        merged["Background Checks"], model="additive", period=365
    )
    merged["Deseasonalized Background Checks"] = decomp.observed - decomp.seasonal

    return merged


# ── 5. Main analysis ─────────────────────────────────────────────────

if __name__ == "__main__":

    # Variable mapping: full name -> short DAG label
    NAME_MAP = {
        "Anti-Reg": "A",
        "Pro-Reg": "P",
        "Media Articles": "M",
        "Deseasonalized Background Checks": "D",
    }
    LABEL_MAP = {v: k for k, v in NAME_MAP.items()}

    variables = sorted(NAME_MAP.values())
    max_lag = 7

    # ── Relations from discovered network (oCSE, alpha=0.005, 200 shuffles) ──
    #
    # Full edge list from discover_network():
    #   D -> M  lag=7  cmi=0.023352
    #   D -> M  lag=1  cmi=0.005634
    #   A -> P  lag=1  cmi=0.019138
    #   A -> M  lag=1  cmi=0.013525
    #   P -> D  lag=7  cmi=0.002685
    #   P -> A  lag=1  cmi=0.055341
    #   P -> M  lag=1  cmi=0.064096
    #   M -> D  lag=7  cmi=0.019163
    #   M -> P  lag=4  cmi=0.018752
    #   M -> P  lag=1  cmi=0.014069
    #
    # No self-loops were discovered.

    relations = [
        ("D", 7, "M"),
        ("D", 1, "M"),
        ("A", 1, "P"),
        ("A", 1, "M"),
        ("P", 7, "D"),
        ("P", 1, "A"),
        ("P", 1, "M"),
        ("M", 7, "D"),
        ("M", 4, "P"),
        ("M", 1, "P"),
    ]

    # Build time-unrolled DAG
    model = build_lagged_dag(variables, max_lag, relations)
    plot_lagged_dag(model, save_path="results/lagged_dag.pdf")

    print(f"DAG: {len(model.nodes())} nodes, {len(model.edges())} edges")
    print()

    # ── Backdoor analysis: Pro-Reg -> Background Checks ──────────────

    # Treatment: P at various lags.  Outcome: D_t
    treatment_var = "P"
    outcome_var = "D"
    outcome_node = "D_t"

    print("=" * 70)
    print("BACKDOOR ANALYSIS: Pro-Reg -> Deseasonalized Background Checks")
    print("=" * 70)

    for lag in range(1, max_lag + 1):
        treatment_node = f"{treatment_var}_t-{lag}"
        if treatment_node not in model.nodes():
            continue

        try:
            adj_set = get_minimal_adjustment_set(model, treatment_node, outcome_node)
            print(f"\n  {treatment_node} -> {outcome_node}")
            print(f"    Minimal adjustment set: {adj_set}")

            # Map short DAG names back to full names + lags for partial corr
            adj_decoded = []
            for node in adj_set:
                parts = node.split("_t")
                short = parts[0]
                node_lag = 0 if parts[1] == "" else int(parts[1].replace("-", ""))
                adj_decoded.append((LABEL_MAP[short], node_lag))
            print(f"    Decoded: {adj_decoded}")

        except ValueError as e:
            print(f"\n  {treatment_node} -> {outcome_node}")
            print(f"    No valid adjustment set: {e}")

    # ── Partial correlations ─────────────────────────────────────────
    #
    # The discovered network tells us the parents of D (Background Checks):
    #   - Pro-Reg at lag 7   (P → D, cmi=0.002685)
    #   - Media Articles at lag 7  (M → D, cmi=0.019163)
    #
    # To test the causal effect of Pro-Reg on D, we condition on the
    # OTHER parents of D (i.e. Media Articles at lag 7). This matches
    # how oCSE found the edge: I(P_lag7; D | M_lag7) was significant.
    #
    # The backdoor criterion gives an empty set for P_t-7 → D_t, which
    # yields a non-significant unconditional correlation. But the effect
    # is suppressed by Media — conditioning on Media reveals it.

    merged = load_merged_data()
    Y_col = "Deseasonalized Background Checks"

    # ── Discovered parents of D ──
    # From oCSE: (source, lag, cmi)
    parents_of_D = [
        ("Pro-Reg", 7, 0.002685),
        ("Media Articles", 7, 0.019163),
    ]

    print()
    print("=" * 70)
    print("CAUSAL EFFECT: Pro-Reg -> Background Checks")
    print("  conditioning on other parents of D (from discovered network)")
    print("=" * 70)

    # Build lagged dataframe
    df_test = pd.DataFrame(index=merged.index)
    df_test[Y_col] = merged[Y_col]

    for var, lag, _ in parents_of_D:
        col = f"{var}_lag{lag}"
        df_test[col] = merged[var].shift(lag)

    df_test = df_test.dropna()

    X_col = "Pro-Reg_lag7"
    covars = [f"{var}_lag{lag}" for var, lag, _ in parents_of_D
              if not (var == "Pro-Reg" and lag == 7)]

    print(f"\n  Treatment:  Pro-Reg (lag 7)")
    print(f"  Outcome:    {Y_col}")
    print(f"  Condition:  {covars}")
    print(f"  n = {len(df_test)}")

    # Partial correlation (Pearson)
    result_pcorr = pg.partial_corr(
        data=df_test, x=X_col, y=Y_col, covar=covars,
    )
    p_col = "p-val" if "p-val" in result_pcorr.columns else "p_val"

    # Spearman (rank-based, robust to the power-law distribution)
    from scipy.stats import spearmanr
    import statsmodels.api as sm

    # Residualize for Spearman
    M_design = sm.add_constant(df_test[covars])
    D_resid = sm.OLS(df_test[Y_col], M_design).fit().resid
    P_resid = sm.OLS(df_test[X_col], M_design).fit().resid
    r_spearman, p_spearman = spearmanr(P_resid, D_resid)

    print()
    print("  Partial correlation (Pearson):")
    print(f"    r = {result_pcorr['r'].values[0]:+.4f}")
    print(f"    p = {result_pcorr[p_col].values[0]:.6f}")
    print(f"    n = {int(result_pcorr['n'].values[0])}")
    print()
    print("  Rank correlation on residuals (Spearman):")
    print(f"    r = {r_spearman:+.4f}")
    print(f"    p = {p_spearman:.6f}")

    r_val = result_pcorr["r"].values[0]
    p_val = result_pcorr[p_col].values[0]

    print()
    direction = "POSITIVE" if r_val > 0 else "NEGATIVE"
    if p_val < 0.05:
        print(f"  RESULT: {direction} causal effect")
        print(f"    Pearson  r={r_val:+.4f}  (p={p_val:.6f})")
        print(f"    Spearman r={r_spearman:+.4f}  (p={p_spearman:.6f})")
        print()
        print(f"  Pro-Reg TikTok content {'increases' if r_val > 0 else 'decreases'} "
              f"background checks 7 days later,")
        print(f"  after conditioning on Media Articles (lag 7).")
    else:
        print(f"  RESULT: No significant causal effect (r={r_val:+.4f}, p={p_val:.6f})")

    # ── Also test all edges into D with their co-parents ──

    print()
    print("=" * 70)
    print("ALL EDGES INTO D: partial correlations conditioned on co-parents")
    print("=" * 70)

    for target_var, target_lag, target_cmi in parents_of_D:
        X_col_i = f"{target_var}_lag{target_lag}"
        covars_i = [f"{var}_lag{lag}" for var, lag, _ in parents_of_D
                    if not (var == target_var and lag == target_lag)]

        result_i = pg.partial_corr(
            data=df_test, x=X_col_i, y=Y_col, covar=covars_i,
        )
        r_i = result_i["r"].values[0]
        p_i = result_i[p_col].values[0] if p_col in result_i.columns else result_i["p_val"].values[0]
        sig_i = "*" if p_i < 0.05 else ""

        # Spearman on residuals
        if covars_i:
            M_i = sm.add_constant(df_test[covars_i])
            D_res_i = sm.OLS(df_test[Y_col], M_i).fit().resid
            X_res_i = sm.OLS(df_test[X_col_i], M_i).fit().resid
        else:
            D_res_i = df_test[Y_col]
            X_res_i = df_test[X_col_i]
        r_sp_i, p_sp_i = spearmanr(X_res_i, D_res_i)
        sig_sp_i = "*" if p_sp_i < 0.05 else ""

        print(f"\n  {target_var} (lag {target_lag})  ->  D")
        print(f"    CMI = {target_cmi:.6f}")
        print(f"    Condition on: {covars_i if covars_i else '(none)'}")
        print(f"    Pearson:  r={r_i:+.4f}, p={p_i:.6f}  {sig_i}")
        print(f"    Spearman: r={r_sp_i:+.4f}, p={p_sp_i:.6f}  {sig_sp_i}")

    # ── Also check reverse: D -> P (does background checks cause Pro-Reg?) ──

    print()
    print("=" * 70)
    print("REVERSE CHECK: Background Checks -> Pro-Reg")
    print("=" * 70)

    for lag in range(1, max_lag + 1):
        rev_treatment = f"{outcome_var}_t-{lag}"
        rev_outcome = f"{treatment_var}_t"
        if rev_treatment not in model.nodes():
            continue
        try:
            adj_rev = get_minimal_adjustment_set(model, rev_treatment, rev_outcome)
            print(f"\n  {rev_treatment} -> {rev_outcome}")
            print(f"    Minimal adjustment set: {adj_rev}")
        except ValueError as e:
            print(f"\n  {rev_treatment} -> {rev_outcome}")
            print(f"    No valid adjustment set: {e}")
