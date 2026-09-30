"""Plot helpers on synthetic encodings with a known structure."""
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from pe_zoo import plots  # noqa: E402

W, N, K = 24, 30, 3


def synthetic(seed=0):
    """Eigenvector halves that alternate between two fixed subspaces, with random signs and mixing of the basis."""
    rng = np.random.default_rng(seed)
    bases = [np.linalg.qr(rng.normal(size=(N, K)))[0] for _ in range(2)]
    u = np.stack([bases[(t // 6) % 2] @ np.linalg.qr(rng.normal(size=(K, K)))[0] for t in range(W)])
    lam = np.tile(np.linspace(0, 0.5, K), (W, N, 1))
    return np.concatenate([u, lam], axis=-1)


def test_subspace_similarity_ignores_sign_and_basis_and_sees_the_subspace():
    pe = synthetic()
    s = plots.window_similarity(pe)
    assert s.shape == (W, W) and np.allclose(np.diag(s), 1)
    assert np.allclose(s[:6, :6], 1) and np.allclose(s[:6, 12:18], 1)            # same subspace, any signs and basis
    assert s[0, 6] < 0.7                                                          # the other subspace
    flipped = pe.copy()
    flipped[3, :, :K] *= -1
    assert np.allclose(plots.window_similarity(flipped), s)


def test_lag_similarity_is_periodic():
    lags = np.array([0, 6, 12])
    s = plots.lag_similarity(synthetic(), lags)
    assert s[0] == pytest.approx(1) and s[1] < 0.7 and s[2] == pytest.approx(1)


def test_edge_jaccard():
    class Graph:
        num_nodes, num_steps = 5, 4
        steps = [np.array([[0, 1], [1, 2]]), np.array([[1, 0], [2, 1]]), np.array([[0, 3], [3, 4]]), np.array([[0, 1], [1, 2]])]

        def edges_at(self, step):
            return self.steps[step], None

    j = plots.edge_jaccard(Graph(), [0, 1, 3])
    assert j[0] == 1 and j[2] == 1                                                # reversed edges are the same edge
    assert 0 < j[1] < 1


def test_figures():
    pe, times = synthetic(), np.arange(W)
    lags = np.arange(1, 10)
    assert len(plots.plot_eigenvalues(times, pe).axes) == 1
    assert len(plots.plot_components(times, pe, [0, 5], components=3).axes) == 3
    assert len(plots.plot_window(pe[0]).axes) == 3                                 # heatmap, colorbar, eigenvalues
    assert len(plots.plot_similarity_matrix(times, pe, start=4, end=20).axes) == 2
    assert len(plots.plot_lag_curves(lags, {"encoding": plots.lag_similarity(pe, lags), "topology": np.ones(9)}).axes) == 1


def test_fixed_topology_has_no_time_plots():
    with pytest.raises(ValueError, match="fixed topology"):
        plots.plot_eigenvalues(np.arange(3), np.zeros((30, 6)))
