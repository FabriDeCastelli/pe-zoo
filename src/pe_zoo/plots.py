"""Diagnostic plots of window encodings: how the spectrum and the eigenvectors move as the topology does.

Every function takes the encodings of one series, `pe` [windows, nodes, 2k] as in `PositionalEncodings.array`, and the
time of each window (its last step) as an array of numbers or `datetime64`. They return the matplotlib figure and never
show or save it. Eigenvector signs are arbitrary and differ from window to window, so the time plots show absolute
values and the similarities are sign-invariant. Needs the `plot` extra.
"""
import numpy as np


def _time_varying(pe) -> np.ndarray:
    pe = np.asarray(pe, dtype=np.float64)
    if pe.ndim != 3:
        raise ValueError("these plots need one encoding per window; a fixed topology has a single [nodes, 2k] array")
    return pe


def _shown(times, start, end) -> np.ndarray:
    times = np.asarray(times)
    return (times >= start) & (times < end)


def orthonormal_bases(pe) -> np.ndarray:
    """[windows, nodes, k] orthonormal bases of the eigenvector half of each window's encoding."""
    pe = _time_varying(pe)
    return np.linalg.qr(pe[..., :pe.shape[-1] // 2])[0]


def _similarity(q_a, q_b) -> np.ndarray:
    """Mean cosine of the principal angles between the subspaces spanned by the columns of q_a and q_b, pairwise along axis 0."""
    return np.linalg.svd(np.einsum("wnp,wnq->wpq", q_a, q_b), compute_uv=False).mean(-1)


def window_similarity(pe) -> np.ndarray:
    """[windows, windows]: 1 when two windows span the same eigenvector subspace, whatever the signs and the basis."""
    q = orthonormal_bases(pe)
    w = len(q)
    rows, cols = np.indices((w, w))
    return _similarity(q[rows.ravel()], q[cols.ravel()]).reshape(w, w)


def lag_similarity(pe, lags) -> np.ndarray:
    """Mean subspace similarity between the windows `lag` steps apart."""
    q = orthonormal_bases(pe)
    return np.array([_similarity(q[:len(q) - lag], q[lag:]).mean() for lag in lags])


def edge_jaccard(g, lags) -> np.ndarray:
    """Mean Jaccard similarity of the (undirected) edge sets of steps `lag` apart: the topology's own memory."""
    n = g.num_nodes
    edges = [np.unique(np.minimum(*e) * n + np.maximum(*e)) for e in (g.edges_at(s)[0] for s in range(g.num_steps))]

    def jaccard(a, b):
        inter = len(np.intersect1d(a, b, assume_unique=True))
        return inter / max(len(a) + len(b) - inter, 1)

    return np.array([np.mean([jaccard(edges[s], edges[s + lag]) for s in range(len(edges) - lag)]) for lag in lags])


def plot_eigenvalues(times, pe, *, title: str = ""):
    """The k smallest eigenvalues of each window over time: how the spectrum drifts as the topology changes."""
    import matplotlib.pyplot as plt

    pe = _time_varying(pe)
    k = pe.shape[-1] // 2
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for j in range(k):
        ax.plot(times, pe[:, 0, k + j], lw=1.0, label=f"$\\lambda_{{{j + 1}}}$")
    ax.set_xlabel("window (last step)")
    ax.set_ylabel("eigenvalue")
    ax.legend(ncol=k, fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_components(times, pe, nodes, *, components: int = 3, title: str = ""):
    """|entry| of the first eigenvectors at chosen nodes over time (the sign is arbitrary, so only the magnitude is shown)."""
    import matplotlib.pyplot as plt

    pe = _time_varying(pe)
    fig, axes = plt.subplots(components, 1, sharex=True, figsize=(11, 1.9 * components + 0.8), squeeze=False)
    for j, ax in enumerate(axes[:, 0]):
        for node in nodes:
            ax.plot(times, np.abs(pe[:, node, j]), lw=0.9, label=f"node {node}")
        ax.set_ylabel(f"$|u_{{{j + 1}}}|$")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(ncol=len(nodes), fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("window (last step)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_window(window_pe, *, title: str = ""):
    """One window's encoding [nodes, 2k]: the eigenvector half as nodes by eigenvector, and the eigenvalues beside it."""
    import matplotlib.pyplot as plt

    window_pe = np.asarray(window_pe, dtype=np.float64)
    k = window_pe.shape[-1] // 2
    u = window_pe[:, :k]
    fig, (left, right) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]}, figsize=(9, 4.6))
    bound = np.abs(u).max()
    image = left.imshow(u, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound, interpolation="nearest")
    left.set_xlabel("eigenvector")
    left.set_ylabel(f"node ({100 * np.mean(np.abs(u).sum(1) == 0):.0f}% all zero)")
    left.set_xticks(range(k), [str(j + 1) for j in range(k)])
    fig.colorbar(image, ax=left, label="entry (sign is arbitrary)")
    right.barh(range(k), window_pe[0, k:])
    right.invert_yaxis()
    right.set_yticks(range(k), [f"$\\lambda_{{{j + 1}}}$" for j in range(k)])
    right.set_xlabel("eigenvalue")
    right.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_similarity_matrix(times, pe, *, start, end, title: str = ""):
    """Subspace similarity between the eigenvectors of every pair of windows in [start, end): blocks are periods of a stable topology."""
    import matplotlib.pyplot as plt

    shown = _shown(times, start, end)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    image = ax.imshow(window_similarity(np.asarray(pe)[shown]), cmap="viridis", vmin=0, vmax=1, origin="lower")
    ticks = np.linspace(0, shown.sum() - 1, min(shown.sum(), 8)).astype(int)
    labels = [str(np.asarray(times)[shown][i]) for i in ticks]
    ax.set_xticks(ticks, labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(ticks, labels, fontsize=7)
    fig.colorbar(image, label="mean cosine of principal angles")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_lag_curves(lags, curves: dict[str, np.ndarray], *, title: str = ""):
    """Curves against lag in steps, e.g. {"encoding": lag_similarity(pe, lags), "topology": edge_jaccard(g, lags)}."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 3.6))
    for label, values in curves.items():
        ax.plot(lags, values, label=label, lw=1.4, ls="--" if label == "topology" else "-",
                color="black" if label == "topology" else None)
    ax.set_xlabel("lag (steps)")
    ax.set_ylabel("similarity")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    return fig
