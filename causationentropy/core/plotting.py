import itertools
import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch

from causationentropy.core.stats import auc


def _communities_seed_order(G: nx.Graph) -> List:
    # Greedy modularity communities is in NetworkX stdlib
    try:
        comms = list(
            nx.algorithms.community.greedy_modularity_communities(G.to_undirected())
        )
    except Exception as e:
        print(f"Error finding communities: {e}")
        comms = [set(G.nodes())]
    # Order communities by size, and nodes inside by degree (descending)
    order = []
    for c in sorted(comms, key=lambda s: -len(s)):
        sub = list(c)
        sub.sort(key=lambda n: -G.degree(n))
        order.extend(sub)
    # De-duplicate if overlaps happened (shouldn’t, but safe)
    seen = set()
    uniq = []
    for n in order:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    # Add any missed nodes
    for n in G.nodes():
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq


def _circular_positions(order: List, radius: float = 1.0) -> Dict:
    N = len(order)
    pos = {}
    for i, n in enumerate(order):
        theta = 2 * math.pi * i / N
        pos[n] = np.array([radius * math.cos(theta), radius * math.sin(theta)])
    return pos


def _edge_crossings_for_order(G: nx.Graph, order: List) -> int:
    # Map node -> index on circle
    idx = {n: i for i, n in enumerate(order)}
    # Convert MultiDiGraph edges to undirected simple pairs for crossings
    edges = []
    for u, v in G.edges():
        if u == v:
            continue
        a, b = idx[u], idx[v]
        if a > b:
            a, b = b, a
        edges.append((a, b))
    # Count chord crossings on circle: chords (a,b) and (c,d) cross
    # iff a < c < b < d or c < a < d < b (indices on the circle)
    crossings = 0
    for (a, b), (c, d) in itertools.combinations(edges, 2):
        if (a < c < b < d) or (c < a < d < b):
            crossings += 1
    return crossings


def _edge_length_variance(G: nx.Graph, order: List) -> float:
    idx = {n: i for i, n in enumerate(order)}
    N = len(order)
    # chord length on unit circle: L = 2*sin(pi*delta/N)
    lengths = []
    for u, v in G.edges():
        if u == v:
            continue
        du = abs(idx[u] - idx[v])
        delta = min(du, N - du)
        L = 2 * math.sin(math.pi * delta / N)
        lengths.append(L)
    if not lengths:
        return 0.0
    arr = np.array(lengths, dtype=float)
    return float(np.var(arr) / (np.mean(arr) + 1e-9))


def _connected_angle_penalty(G: nx.Graph, order: List, min_frac: float = 0.0) -> float:
    # Encourage connected nodes to not be nearly adjacent (tiny angles),
    # which reduces hairballing near labels.
    idx = {n: i for i, n in enumerate(order)}
    N = len(order)
    s = 0.0
    m = 0
    for u, v in G.edges():
        if u == v:
            continue
        du = abs(idx[u] - idx[v])
        delta = min(du, N - du) / N  # fractional arc
        # Penalize very small delta (nonlinear)
        s += 1.0 / (delta + 1e-6)
        m += 1
    return s / max(m, 1)


def _label_collision_proxy(order: List, min_sep: int = 0) -> int:
    # Simple proxy: penalize runs of nodes being too close is not needed on a unit circle,
    # but if you later map label boxes, replace this with precise bbox overlap checks.
    # Here we keep 0 to avoid overfitting; left as hook.
    return 0


def _objective(G: nx.Graph, order: List, w=(1.0, 0.2, 0.05, 0.0)) -> float:
    # Lower is better
    crossings = _edge_crossings_for_order(G, order)
    var_len = _edge_length_variance(G, order)
    ang_pen = _connected_angle_penalty(G, order)
    coll = _label_collision_proxy(order)
    return w[0] * crossings + w[1] * var_len + w[2] * ang_pen + w[3] * coll


def optimize_circular_order(
    G: nx.Graph,
    seed_order: List = None,
    max_iters: int = 3000,
    block_moves: bool = True,
    rng: int = 7,
) -> List:
    random.seed(rng)
    if seed_order is None:
        seed_order = _communities_seed_order(G)
    best = seed_order[:]
    best_score = _objective(G, best)
    temp0 = 1.0
    for t in range(1, max_iters + 1):
        cur = best[:]
        N = len(cur)
        # Propose a move: swap two nodes or reverse a block
        if block_moves and N >= 6 and random.random() < 0.5:
            i, j = sorted(random.sample(range(N), 2))
            # reverse a block [i, j]
            cur[i : j + 1] = reversed(cur[i : j + 1])
        else:
            i, j = random.sample(range(N), 2)
            cur[i], cur[j] = cur[j], cur[i]

        score = _objective(G, cur)
        if score < best_score:
            best, best_score = cur, score
        else:
            # Simulated annealing accept
            T = temp0 * (1.0 - t / (max_iters + 1))
            if (
                T > 1e-3
                and math.exp((best_score - score) / max(T, 1e-9)) > random.random()
            ):
                best, best_score = cur, score
    return best


def roc_curve(TPRs, FPRs):
    """
    Plot Receiver Operating Characteristic (ROC) curve.

    This function creates a ROC curve visualization, which is a graphical plot
    that illustrates the diagnostic ability of a binary classifier system.
    The ROC curve plots the True Positive Rate against the False Positive Rate
    at various threshold settings.

    The ROC curve is defined by the parametric equations:

    .. math::

        \\text{TPR}(t) = \\frac{\\text{TP}(t)}{\\text{TP}(t) + \\text{FN}(t)} = \\frac{\\text{TP}(t)}{P}

        \\text{FPR}(t) = \\frac{\\text{FP}(t)}{\\text{FP}(t) + \\text{TN}(t)} = \\frac{\\text{FP}(t)}{N}

    where t is the classification threshold, P is the total number of positives,
    and N is the total number of negatives.

    Parameters
    ----------
    TPRs : array-like
        True Positive Rates (Sensitivity, Recall) for different thresholds.
        Values should be in [0, 1].
    FPRs : array-like
        False Positive Rates (1 - Specificity) for different thresholds.
        Values should be in [0, 1].

    Notes
    -----
    **ROC Curve Interpretation:**
    - Perfect classifier: Curve passes through (0, 1) - high TPR, zero FPR
    - Random classifier: Diagonal line from (0, 0) to (1, 1)
    - Useless classifier: Curve below the diagonal

    **Key Points:**
    - (0, 0): No false positives, but also no true positives (very conservative)
    - (1, 1): All positives detected, but all negatives misclassified (very liberal)
    - (0, 1): Perfect classification (ideal classifier)

    **AUC (Area Under Curve):**
    - AUC = 1.0: Perfect classifier
    - AUC = 0.5: Random classifier
    - AUC < 0.5: Worse than random (can be inverted)

    **Applications:**
    - Medical diagnosis evaluation
    - Network reconstruction assessment
    - Causal discovery method comparison
    - Binary classification performance analysis

    The function automatically computes and displays the AUC value on the plot
    using the trapezoidal integration rule.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from causationentropy.core.plotting import roc_curve
    >>>
    >>> # Perfect classifier example
    >>> tpr_perfect = [0, 1, 1]
    >>> fpr_perfect = [0, 0, 1]
    >>>
    >>> plt.figure(figsize=(8, 6))
    >>> roc_curve(tpr_perfect, fpr_perfect)
    >>> plt.legend(['Perfect Classifier'])
    >>> plt.show()
    >>>
    >>> # Random classifier comparison
    >>> tpr_random = [0, 0.5, 1]
    >>> fpr_random = [0, 0.5, 1]
    >>> roc_curve(tpr_random, fpr_random)
    >>> plt.legend(['Random Classifier'])

    See Also
    --------
    causationentropy.core.stats.auc : Compute area under ROC curve
    causationentropy.core.stats.Compute_TPR_FPR : Compute TPR and FPR from confusion matrix
    """
    plt.plot(FPRs, TPRs)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Add diagonal reference line for random classifier
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random Classifier")

    AUC = auc(TPRs, FPRs)
    plt.text(
        0.4,
        0.1,
        f"AUC = {AUC:.4f}",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


def plot_causal_network(
    G: nx.MultiDiGraph,
    pos: Dict = None,
    seed: int = 7,
    figsize: Tuple[float, float] = (14, 14),
    dpi: int = None,
    node_size: int = 8000,
    node_color: str = "white",
    node_linewidth: float = 5.0,
    edge_width_range: Tuple[float, float] = (1.0, 8.0),
    arrowsize: int = 30,
    stats_fontsize: int = 16,
    label_fontsize: int = 16,
    legend_fontsize: int = 10,
    colorbar_fontsize: int = 14,
    title_fontsize: int = 24,
    colormaps: List[str] = None,
    colorblind_safe: bool = False,
    show_colorbar: bool = True,
    use_pvalue_alpha: bool = True,
    pvalue_threshold: float = 0.05,
    show_edge_labels: bool = False,
    show_statistics: bool = True,
    title: str = "Discovered Causal Network",
    save_path: str = None,
    file_format: str = "png",
    transparent: bool = False,
    show_plot: bool = True,
):
    """
    Plot a causal network from a MultiDiGraph object with production-quality styling.

    This function visualizes a causal network, accounting for edge attributes
    like lag, p-value, and conditional mutual information (CMI). It is designed
    to produce publication-quality plots with high readability, customizable
    styling, and multiple output options.

    Parameters
    ----------
    G : nx.MultiDiGraph
        The causal network graph to plot. Expected to have 'lag', 'cmi', and
        optionally 'p_value' as edge attributes.
    pos : dict, optional
        A dictionary with nodes as keys and positions as values. If not
        provided, an optimized circular layout will be computed.
    seed : int, default=7
        Seed for the random number generator used in layout optimization.
    figsize : tuple of float, default=(14, 14)
        Figure size in inches (width, height).
    dpi : int, optional
        Resolution in dots per inch. If None, uses matplotlib's default (~100).
        Use 300+ for publication quality.
    node_size : int, default=8000
        Size of nodes in the plot.
    node_color : str, default='white'
        Color of nodes.
    node_linewidth : float, default=5.0
        Width of node borders.
    edge_width_range : tuple of float, default=(1.0, 8.0)
        Range for edge width scaling based on CMI values (min_width, max_width).
    arrowsize : int, default=30
        Size of arrow heads on directed edges.
    label_fontsize : int, default=16
        Font size for node labels.
    title_fontsize : int, default=24
        Font size for the plot title.
    colormaps : list of str, optional
        List of matplotlib colormap names to use for different lags.
        If None, defaults to ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds'].
        If colorblind_safe=True, this is overridden with accessible palettes.
    colorblind_safe : bool, default=False
        If True, use colorblind-safe color palettes (viridis, plasma, cividis, etc.).
    show_colorbar : bool, default=True
        If True, display a colorbar showing the CMI scale for each lag.
    use_pvalue_alpha : bool, default=True
        If True, use p-value to set edge transparency (more significant = more opaque).
        Requires 'p_value' in edge attributes.
    pvalue_threshold : float, default=0.05
        Significance threshold for filtering edges. Edges with p > threshold
        are drawn with reduced opacity.
    show_edge_labels : bool, default=False
        If True, display CMI values as edge labels.
    show_statistics : bool, default=True
        If True, display network statistics (nodes, edges, max lag) in a text box.
    title : str, default='Discovered Causal Network'
        Title for the plot.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    file_format : str, default='png'
        File format for saving ('png', 'pdf', 'svg', 'eps', etc.).
    transparent : bool, default=False
        If True, save figure with transparent background.
    show_plot : bool, default=True
        If True, display the plot using plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.

    Notes
    -----
    **Layout Optimization:**
        Nodes are arranged in a circular layout. If `pos` is not provided, the
        node order is optimized to minimize edge crossings and improve readability
        using simulated annealing.

    **Edge Rendering:**
        - Edges for lag 1 are drawn as straight lines (connectionstyle radius=0).
        - Edges for lags > 1 are drawn as arcs outside the circle, with arc
          radius increasing for higher lags to prevent overlap.
        - Edge color and width are scaled by the CMI value within each lag group.
        - Edge transparency (alpha) is modulated by p-value if use_pvalue_alpha=True.

    **Color Accessibility:**
        When colorblind_safe=True, the function uses perceptually uniform and
        colorblind-accessible colormaps from matplotlib (viridis, plasma, cividis,
        inferno, magma). This ensures the plot is accessible to readers with
        color vision deficiencies.

    **P-value Visualization:**
        If edge attributes include 'p_value' and use_pvalue_alpha=True:
        - Edges with p < pvalue_threshold are drawn with alpha=1.0 (fully opaque)
        - Edges with p >= pvalue_threshold are drawn with alpha=0.3 (translucent)
        This provides a visual indication of statistical significance.

    **Publication Quality:**
        - DPI parameter allows control of resolution (use 300-600 for journals)
        - Large fonts and thick lines ensure readability when scaled
        - Optional statistics box provides at-a-glance network properties
        - Multiple export formats supported (PNG, PDF, SVG, EPS)

    Examples
    --------
    Basic usage with default settings:

    >>> import networkx as nx
    >>> from causationentropy.core.plotting import plot_causal_network
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge('X1', 'X2', lag=1, cmi=0.5, p_value=0.01)
    >>> G.add_edge('X2', 'X3', lag=1, cmi=0.3, p_value=0.03)
    >>> fig, ax = plot_causal_network(G)

    Colorblind-safe plot with custom styling:

    >>> fig, ax = plot_causal_network(
    ...     G,
    ...     colorblind_safe=True,
    ...     figsize=(16, 16),
    ...     node_size=10000,
    ...     show_edge_labels=True
    ... )

    Save high-resolution plot for publication:

    >>> fig, ax = plot_causal_network(
    ...     G,
    ...     dpi=600,
    ...     save_path='causal_network.pdf',
    ...     file_format='pdf',
    ...     show_plot=False
    ... )

    Customize colors and disable statistics:

    >>> custom_cmaps = ['YlOrRd', 'PuBu', 'BuGn']
    >>> fig, ax = plot_causal_network(
    ...     G,
    ...     colormaps=custom_cmaps,
    ...     show_statistics=False,
    ...     title='Custom Causal Network'
    ... )

    See Also
    --------
    optimize_circular_order : Optimize node ordering for circular layout
    causationentropy.core.discovery.discover_network : Discover causal networks
    """
    if not G or G.number_of_nodes() == 0:
        print("Graph is empty, nothing to plot.")
        return None, None

    # Set colorblind-safe palettes if requested
    if colorblind_safe:
        colormaps = ["viridis", "plasma", "cividis", "inferno", "magma"]
    elif colormaps is None:
        colormaps = ["Blues", "Greens", "Oranges", "Purples", "Reds"]

    # If no positions are provided, compute optimized circular positions
    if pos is None:
        order = optimize_circular_order(G, rng=seed)
        pos = _circular_positions(order, radius=1.0)

    # Collect edge data grouped by lag
    edge_data = defaultdict(list)
    self_loops = defaultdict(list)

    for u, v, k, estimated_data in G.edges(keys=True, data=True):
        if u == v:
            self_loops[estimated_data.get("lag", 0)].append((u, v, estimated_data))
            continue
        lag = estimated_data.get("lag", 0)
        cmi = max(0.0, float(estimated_data.get("cmi", 0.0)))
        p_value = estimated_data.get("p_value", None)
        edge_data[lag].append((u, v, cmi, p_value))

    # Create figure with specified DPI
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color=node_color,
        edgecolors="black",
        linewidths=node_linewidth,
        ax=ax,
    )

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=label_fontsize, font_weight="bold", ax=ax)

    sorted_lags = sorted(edge_data.keys())
    higher_lags = [lag for lag in sorted_lags if lag > 1]

    # Track global CMI range for colorbar
    all_cmis = []
    for edges in edge_data.values():
        all_cmis.extend([e[2] for e in edges])
    global_max_cmi = max(all_cmis) if all_cmis else 1.0

    # Draw edges for each lag
    for i, lag in enumerate(sorted_lags):
        edges = edge_data[lag]

        cmis = np.array([e[2] for e in edges])
        p_values = np.array([e[3] if e[3] is not None else 1.0 for e in edges])

        # Normalize CMI for this lag
        max_cmi = cmis.max() if cmis.max() > 0 else 1.0
        norm_cmis = cmis / max_cmi

        # Scale edge widths by CMI
        widths = (
            edge_width_range[0]
            + (edge_width_range[1] - edge_width_range[0]) * norm_cmis
        )

        # Get colors from colormap
        cmap = plt.cm.get_cmap(colormaps[i % len(colormaps)])
        colors = cmap(norm_cmis)

        # Modulate alpha by p-value if requested
        if use_pvalue_alpha:
            alphas = np.where(p_values < pvalue_threshold, 1.0, 0.3)
            # Apply alpha to RGBA colors
            colors[:, 3] = alphas

        # Calculate arc radius for higher lags
        if lag == 1:
            rad = 0.0
        else:
            try:
                lag_index = higher_lags.index(lag)
                rad = 0.1 * (lag_index + 1)
            except ValueError:
                rad = 0.1

        # Draw edges
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(e[0], e[1]) for e in edges],
            edge_color=colors,
            width=widths,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=arrowsize,
            connectionstyle=f"arc3,rad={rad}",
            node_size=node_size,
            ax=ax,
        )

        # Draw edge labels if requested
        if show_edge_labels:
            edge_labels = {(e[0], e[1]): f"{e[2]:.3f}" for e in edges}
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, font_size=10, ax=ax
            )

    # Create legend for lag groups
    legend_elements = []
    for i, lag in enumerate(sorted_lags):
        colormap = plt.cm.get_cmap(colormaps[i % len(colormaps)])
        color = colormap(0.7)
        legend_elements.append(
            Patch(facecolor=color, edgecolor="black", label=f"Lag {lag}")
        )

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=legend_fontsize,
        title="Lag Groups",
        title_fontsize=legend_fontsize,
        framealpha=0.9,
        prop={"weight": "bold", "size": legend_fontsize},
    )

    # Add colorbar showing CMI scale if requested
    if show_colorbar and sorted_lags:
        # Create colorbar for the first lag as representative
        cmap = plt.cm.get_cmap(colormaps[0])
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=Normalize(vmin=0, vmax=global_max_cmi)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(
            "Conditional Mutual Information (CMI)",
            rotation=270,
            labelpad=25,
            fontsize=colorbar_fontsize,
            fontweight="bold",
        )

        font_props = FontProperties(weight="bold", size=colorbar_fontsize / 2)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontproperties(font_props)

    # Add network statistics box if requested
    if show_statistics:
        stats_text = (
            f"Nodes: {G.number_of_nodes()}\n"
            f"Edges: {G.number_of_edges()}\n"
            f"Max Lag: {max(sorted_lags) if sorted_lags else 0}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=stats_fontsize,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="black",
                alpha=0.9,
                linewidth=2,
            ),
            fontweight="bold",
        )

    ax.set_axis_off()
    ax.set_title(title, fontsize=title_fontsize, pad=20, fontweight="bold")

    plt.margins(0.1)
    plt.tight_layout()

    # Save figure if path provided
    if save_path is not None:
        plt.savefig(
            save_path,
            format=file_format,
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
        )
        print(f"Figure saved to: {save_path}")

    # Show plot if requested
    if show_plot:
        plt.show()

    return fig, ax
