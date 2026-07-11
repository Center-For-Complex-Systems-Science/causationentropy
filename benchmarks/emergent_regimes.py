"""
Emergent-regime generators for coupled phase oscillators
========================================================

Stage 1 of matching the experiments to the abstract: actually PRODUCE the
emergent dynamical regimes the abstract names -- chimera, traveling waves,
metastability, cyclops states -- with a KNOWN ground-truth coupling network, so
that causal-network reconstruction (oCSE / PCMCI / Granger / VARLiNGAM) can be
benchmarked on them in a later stage.

The base model is a general Kuramoto-Sakaguchi system

    dtheta_i/dt = omega_i + K * sum_j Wn_ij * [ sin(theta_j - theta_i - alpha)
                                  + a2 * sin(2(theta_j - theta_i) - beta) ]

where Wn is the (row-normalized) coupling matrix.  The GROUND-TRUTH causal graph
for reconstruction is the binary support of W:  edge j -> i  iff  W[i, j] > 0.

This file only generates regimes and verifies they emerged (space-time plots,
global + local order parameters).  Causal discovery is wired in separately.

Run:
    ./venv/bin/python benchmarks/emergent_regimes.py
Produces one diagnostic PNG per regime in benchmarks/results/emergent/.
"""
import os
import numpy as np


# =============================================================================
# GENERAL KURAMOTO-SAKAGUCHI SIMULATOR (RK4)
# =============================================================================

def _rhs(theta, omega, Wn, alpha, K, a2, beta):
    # diff[i, j] = theta_j - theta_i
    diff = theta[None, :] - theta[:, None]
    coupling = np.sin(diff - alpha)
    if a2:
        coupling = coupling + a2 * np.sin(2.0 * diff - beta)
    return omega + K * np.sum(Wn * coupling, axis=1)


def simulate_phase_oscillators(
    W, omega, *, alpha=0.0, K=1.0, a2=0.0, beta=0.0,
    T=6000, dt=0.05, burn_in=2000, theta0=None, noise_std=0.0,
    row_normalize=True, seed=0, store_every=1,
):
    """
    Integrate the general Kuramoto-Sakaguchi model with RK4.

    Parameters
    ----------
    W : (n, n) array        coupling weights; W[i, j] = influence of j on i.
    omega : (n,) array      natural frequencies.
    alpha : float           phase lag (Sakaguchi); ~pi/2 enables chimera.
    K : float               global coupling strength.
    a2, beta : float        second-harmonic amplitude / lag (cyclops states).
    row_normalize : bool    divide each row of W by its sum (mean-field scaling).

    Returns
    -------
    theta : (n_store, n) array of phases (after burn-in), wrapped to [-pi, pi).
    info : dict
    """
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    Wn = W.astype(float).copy()
    if row_normalize:
        rs = Wn.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Wn = Wn / rs

    if theta0 is None:
        theta = rng.uniform(-np.pi, np.pi, size=n)
    else:
        theta = np.asarray(theta0, dtype=float).copy()

    sd = noise_std * np.sqrt(dt)
    out = []
    n_steps = burn_in + T
    for step in range(n_steps):
        k1 = _rhs(theta, omega, Wn, alpha, K, a2, beta)
        k2 = _rhs(theta + 0.5 * dt * k1, omega, Wn, alpha, K, a2, beta)
        k3 = _rhs(theta + 0.5 * dt * k2, omega, Wn, alpha, K, a2, beta)
        k4 = _rhs(theta + dt * k3, omega, Wn, alpha, K, a2, beta)
        theta = theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if sd:
            theta = theta + sd * rng.standard_normal(n)
        if step >= burn_in and (step - burn_in) % store_every == 0:
            out.append(theta.copy())
    theta_hist = np.array(out)
    theta_hist = (theta_hist + np.pi) % (2 * np.pi) - np.pi
    info = {"n": n, "dt": dt * store_every, "alpha": alpha, "K": K,
            "a2": a2, "beta": beta, "row_normalize": row_normalize}
    return theta_hist, info


# =============================================================================
# COUPLING-GRAPH BUILDERS  (ground truth = binary support of W)
# =============================================================================

def ring_nonlocal(n, radius):
    """Ring of n nodes, each bidirectionally coupled to its `radius` nearest
    neighbors on each side. Used for chimera and traveling waves."""
    W = np.zeros((n, n))
    for i in range(n):
        for d in range(1, radius + 1):
            W[i, (i + d) % n] = 1.0
            W[i, (i - d) % n] = 1.0
    return W


def all_to_all(n):
    """Complete graph (no self loop). Cyclops states live here -- note the
    ground-truth graph is then trivially complete (FPR is degenerate)."""
    W = np.ones((n, n))
    np.fill_diagonal(W, 0.0)
    return W


def modular_sbm(n_comm, comm_size, p_in, p_out, seed=0):
    """Stochastic block model with `n_comm` communities. Metastability arises
    from heterogeneous frequencies on a modular network."""
    rng = np.random.default_rng(seed)
    n = n_comm * comm_size
    comm = np.repeat(np.arange(n_comm), comm_size)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = p_in if comm[i] == comm[j] else p_out
            if rng.random() < p:
                W[i, j] = 1.0
    return W, comm


# =============================================================================
# REGIME PRESETS  (parameters from the chimera / cyclops literature)
# =============================================================================

def regime_chimera(seed=0, n=48, radius=None):
    """Abrams-Strogatz nonlocal ring of identical oscillators, phase lag near
    pi/2 -> coexisting coherent + incoherent domains. n=48 gives a clear split.
    The classic range is ~0.35 N, but a SMALLER radius still yields a chimera
    while making the ground-truth graph sparse -- better for causal discovery
    (more true negatives, fewer edges -> faster oCSE)."""
    if radius is None:
        radius = round(0.35 * n)               # nonlocal coupling range ~0.35 N
    W = ring_nonlocal(n, radius)
    omega = np.zeros(n)                       # identical oscillators
    # split initial condition: half coherent, half random -> seeds the chimera
    rng = np.random.default_rng(seed)
    theta0 = np.zeros(n)
    half = n // 2
    theta0[half:] = rng.uniform(-np.pi, np.pi, size=n - half)
    theta0[:half] = 0.01 * rng.standard_normal(half)
    return dict(W=W, omega=omega, alpha=np.pi / 2 - 0.10, K=1.0,
                theta0=theta0, T=4000, dt=0.05, burn_in=6000, noise_std=0.02,
                row_normalize=True, seed=seed, name="chimera")


def regime_traveling_wave(seed=0, n=36):
    """Ring with short-range coupling, identical oscillators, small phase lag ->
    a q-twisted state that travels. Works down to small n (~20)."""
    radius = 2
    W = ring_nonlocal(n, radius)
    omega = np.zeros(n)
    q = 2                                      # winding number of the twist
    theta0 = 2 * np.pi * q * np.arange(n) / n
    theta0 = theta0 + 0.05 * np.random.default_rng(seed).standard_normal(n)
    # noise=0.08: the wave is robust (|Z_q|>0.98) and noise supplies the only
    # time-variation in the coupling features that causal discovery can use.
    return dict(W=W, omega=omega, alpha=0.1, K=1.0, theta0=theta0,
                T=4000, dt=0.05, burn_in=2000, noise_std=0.08,
                row_normalize=True, seed=seed, name="traveling_wave")


def regime_metastable(seed=0):
    """Modular network + heterogeneous frequencies + intermediate coupling ->
    communities transiently (de)synchronize; fluctuating global order param."""
    W, comm = modular_sbm(n_comm=3, comm_size=8, p_in=0.8, p_out=0.08, seed=seed)
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, 0.5, size=n)       # mild heterogeneity -> communities
                                               # sync internally but drift relative
    return dict(W=W, omega=omega, alpha=0.0, K=1.0, theta0=None,
                T=4000, dt=0.05, burn_in=2000, noise_std=0.02,
                row_normalize=True, seed=seed, name="metastable", comm=comm)


def regime_cyclops(seed=0):
    """Biharmonic repulsive Kuramoto-Sakaguchi (Munyayev et al., PRL 2023):
        dtheta_k/dt = w + (1/N) sum_j [ e1 sin(d - a1) + e2 sin(2d - a2) ],
        d = theta_j - theta_k,  e1=1, a1=1.7 (repulsive), e2=0.08, a2=-0.3.
    Identical oscillators, ODD N (=11) required -> two coherent clusters plus a
    solitary 'eye'. NOTE: all-to-all coupling => ground-truth graph is complete,
    so FPR is degenerate here; kept for dynamics, flagged for the causal stage."""
    n = 11                                     # odd, per the paper
    W = all_to_all(n)
    rng = np.random.default_rng(seed)
    omega = np.zeros(n)                        # identical
    theta0 = rng.uniform(-np.pi, np.pi, size=n)
    # noise=0.02 is the ceiling: above it the solitary 'eye' splits and the
    # [5,5,1] structure breaks. Just enough to give non-constant features.
    return dict(W=W, omega=omega, alpha=1.7, K=1.0, a2=0.08, beta=-0.3,
                theta0=theta0, T=4000, dt=0.05, burn_in=8000, noise_std=0.02,
                row_normalize=True, seed=seed, name="cyclops")


REGIMES = {
    "chimera": regime_chimera,
    "traveling_wave": regime_traveling_wave,
    "metastable": regime_metastable,
    "cyclops": regime_cyclops,
}


# =============================================================================
# DIAGNOSTICS  (did the regime actually emerge?)
# =============================================================================

def global_order_parameter(theta):
    """R1(t) = | mean_j exp(i theta_j) |. The standard Kuramoto order parameter.
    NOTE: it is identically ~0 for twisted (traveling-wave) states and constant
    for rigidly-rotating cluster states (cyclops) -- it is blind to both."""
    return np.abs(np.exp(1j * theta).mean(axis=1))


def daido_order_parameter(theta, m):
    """m-th Daido order parameter R_m(t) = | mean_j exp(i m theta_j) |.
    R_2 reveals two-cluster / splay states (e.g. cyclops) that R_1 misses."""
    return np.abs(np.exp(1j * m * theta).mean(axis=1))


def twisted_order_parameter(theta, q):
    """Generalized order parameter for a q-twisted state on a ring:
    Z_q(t) = mean_j exp(i (theta_j - 2 pi q j / n)).  |Z_q| ~ 1 for a clean
    traveling wave of winding number q; arg(Z_q) drifts as the wave travels."""
    n = theta.shape[1]
    j = np.arange(n)
    return np.exp(1j * (theta - 2 * np.pi * q * j / n)).mean(axis=1)


def best_winding_number(theta, q_max=6):
    """Winding number q that maximizes the time-averaged |Z_q| (>0)."""
    best_q, best_val = 0, -1.0
    for q in range(1, q_max + 1):
        val = np.abs(twisted_order_parameter(theta, q)).mean()
        if val > best_val:
            best_q, best_val = q, val
    return best_q, best_val


def local_order_parameter(theta_snapshot, window):
    """Spatial local order parameter on a ring (window = +/- nodes)."""
    n = theta_snapshot.shape[0]
    z = np.exp(1j * theta_snapshot)
    R = np.empty(n)
    for i in range(n):
        idx = [(i + d) % n for d in range(-window, window + 1)]
        R[i] = np.abs(z[idx].mean())
    return R


def emergence_report(name, theta, info, comm=None):
    """Heuristic check + summary numbers for a regime."""
    R = global_order_parameter(theta)
    last = theta[-1]
    msg = {"regime": name, "R_mean": float(R.mean()), "R_std": float(R.std())}
    if name in ("chimera",):
        Rloc = local_order_parameter(last, window=max(2, theta.shape[1] // 20))
        msg["local_R_min"] = float(Rloc.min())
        msg["local_R_max"] = float(Rloc.max())
        msg["chimera_like"] = bool(Rloc.min() < 0.6 and Rloc.max() > 0.9)
    if name == "metastable":
        msg["metastable_like"] = bool(R.std() > 0.05 and R.mean() < 0.95)
    if name == "traveling_wave":
        # phase should increase ~linearly around the ring (a twist)
        unwrapped = np.unwrap(last)
        msg["twist_total"] = float(unwrapped[-1] - unwrapped[0])
        q, zval = best_winding_number(theta)
        msg["winding_q"] = int(q)
        msg["|Z_q|_mean"] = float(zval)               # ~1 for a clean wave
        ang = np.unwrap(np.angle(twisted_order_parameter(theta, q)))
        msg["wave_speed"] = float((ang[-1] - ang[0]) / ((len(ang) - 1) * info["dt"]))
    if name == "cyclops":
        sizes = _phase_cluster_sizes(last, tol=0.25)
        msg["cluster_sizes"] = sizes
        # cyclops = two big clusters + one solitary 'eye'
        big = [s for s in sizes if s >= 2]
        solo = [s for s in sizes if s == 1]
        msg["cyclops_like"] = bool(len(big) == 2 and len(solo) >= 1)
        R2 = daido_order_parameter(theta, 2)
        msg["R2_mean"] = float(R2.mean())             # 2nd-harmonic sees the splay
        uw = np.unwrap(theta, axis=0)
        freqs = (uw[-1] - uw[0]) / ((theta.shape[0] - 1) * info["dt"])
        msg["locked_freq"] = float(freqs.mean())
        msg["freq_spread"] = float(np.ptp(freqs))     # ~0 => rigid rotation
    return msg


def _phase_cluster_sizes(phases, tol=0.25):
    """Greedy 1-D clustering of phases on the circle; returns sorted cluster
    sizes (largest first)."""
    z = np.exp(1j * phases)
    assigned = -np.ones(len(phases), dtype=int)
    centers, sizes = [], []
    for k in range(len(phases)):
        placed = False
        for c, ctr in enumerate(centers):
            if abs(np.angle(z[k] / np.exp(1j * ctr))) < tol:
                assigned[k] = c
                placed = True
                break
        if not placed:
            centers.append(phases[k])
            assigned[k] = len(centers) - 1
    sizes = [int((assigned == c).sum()) for c in range(len(centers))]
    return sorted(sizes, reverse=True)


# =============================================================================
# MAIN: simulate every regime and save a diagnostic figure
# =============================================================================

def diagnostic_figure(name, theta, info, comm, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = theta.shape[1]
    t = np.arange(theta.shape[0]) * info["dt"]
    R = global_order_parameter(theta)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # (a) space-time of sin(theta)
    im = axes[0].imshow(np.sin(theta).T, aspect="auto", cmap="twilight",
                        extent=[t[0], t[-1], 0, n], origin="lower")
    axes[0].set_xlabel("time"); axes[0].set_ylabel("oscillator index")
    axes[0].set_title(f"{name}: sin(theta) space-time")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    # (b) regime-appropriate order parameter
    #     R1 is blind to twisted (traveling-wave) and rigidly-rotating (cyclops)
    #     states, so use the generalized / Daido order parameter there.
    axes[1].set_xlabel("time"); axes[1].set_ylim(0, 1.05)
    if name == "traveling_wave":
        q, _ = best_winding_number(theta)
        Zq = twisted_order_parameter(theta, q)
        ang = np.unwrap(np.angle(Zq))
        speed = (ang[-1] - ang[0]) / ((len(ang) - 1) * info["dt"])
        axes[1].plot(t, np.abs(Zq), "g-", lw=1.2, label=rf"$|Z_{q}|$ (twisted)")
        axes[1].plot(t, R, "0.6", lw=0.8, label=r"$R_1$ (standard, $\approx$0)")
        axes[1].set_ylabel("order parameter")
        axes[1].set_title(rf"$|Z_{q}|$={np.abs(Zq).mean():.2f}, wave speed={speed:.3f} rad/s")
        axes[1].legend(loc="center right", fontsize=9)
    elif name == "cyclops":
        R2 = daido_order_parameter(theta, 2)
        axes[1].plot(t, R2, "m-", lw=1.2, label=r"$R_2$ (2-cluster splay)")
        axes[1].plot(t, R, "0.6", lw=0.8, label=r"$R_1$ (standard)")
        axes[1].set_ylabel("order parameter")
        axes[1].set_title(rf"$R_2$={R2.mean():.2f} (rigid rotation: $R$ constant)")
        axes[1].legend(loc="center right", fontsize=9)
    else:
        axes[1].plot(t, R, lw=0.8)
        axes[1].set_ylabel(r"global order parameter $R_1(t)$")
        axes[1].set_title(f"R_mean={R.mean():.2f}  R_std={R.std():.3f}")
    # (c) final snapshot
    axes[2].plot(np.arange(n), theta[-1], ".", ms=6)
    axes[2].set_xlabel("oscillator index"); axes[2].set_ylabel("phase (final)")
    axes[2].set_title("phase snapshot")
    if name == "chimera":
        Rloc = local_order_parameter(theta[-1], window=max(2, n // 20))
        ax2b = axes[2].twinx()
        ax2b.plot(np.arange(n), Rloc, "r-", lw=1.5, alpha=0.7)
        ax2b.set_ylabel("local R", color="r"); ax2b.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "results", "emergent")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Generating emergent regimes (physics verification stage)")
    print("=" * 70)
    for name, builder in REGIMES.items():
        cfg = builder(seed=0)
        comm = cfg.pop("comm", None)
        cfg.pop("name")
        theta, info = simulate_phase_oscillators(**cfg)
        rep = emergence_report(name, theta, info, comm)
        png = os.path.join(out_dir, f"{name}.png")
        diagnostic_figure(name, theta, info, comm, png)
        print(f"\n[{name}]")
        for k, v in rep.items():
            print(f"    {k}: {v}")
        print(f"    saved {png}")


if __name__ == "__main__":
    main()
