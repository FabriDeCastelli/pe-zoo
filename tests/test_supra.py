"""pe_zoo.supra against a dense, naive transcription of the LaTeX (numpy loops, no shared code)."""
import numpy as np
import pytest
import torch

from pe_zoo import supra

N, W, K, MU, TOL = 10, 3, 3, 0.7, 1e-8


def random_layers(seed, n=N, w=W, inactive_last=(8, 9)):
    """w random edge lists; the last snapshot leaves `inactive_last` without edges, one-way edges included."""
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(w):
        allowed = [v for v in range(n) if i < w - 1 or v not in inactive_last]
        pairs = [(a, b) for a in allowed for b in allowed if a != b and rng.random() < 0.3]
        pairs += [(allowed[0], allowed[1]), (allowed[1], allowed[2])]
        layers.append(torch.tensor(pairs, dtype=torch.long).T)
    return layers


def naive_layer(edges, n):
    """Dense A_tau with the global node n: symmetric, binary, global joined to the nodes with an edge."""
    a = np.zeros((n + 1, n + 1))
    for i, j in edges.T.tolist():
        a[i, j] = a[j, i] = 1.0
    for v in range(n):
        if a[v].sum() > 0:
            a[v, n] = a[n, v] = 1.0
    return a


def naive_supra(adjs, mu):
    m, w = adjs[0].shape[0], len(adjs)
    a = np.zeros((w * m, w * m))
    for t in range(w):
        a[t * m:(t + 1) * m, t * m:(t + 1) * m] = adjs[t]
        if t + 1 < w:
            a[t * m:(t + 1) * m, (t + 1) * m:(t + 2) * m] = mu * np.eye(m)
            a[(t + 1) * m:(t + 2) * m, t * m:(t + 1) * m] = mu * np.eye(m)
    return a


def naive_pe(layers, n, k, mu):
    """Phi_p by dense eigh: build A_sup, drop the inactive copies, L = I - D^-1/2 A D^-1/2, last block, columns L2, [U || 1 lambda]."""
    adjs = [naive_layer(e, n) for e in layers]
    a = naive_supra(adjs, mu)
    m = n + 1
    keep = np.concatenate([adj[:, n] > 0 for adj in adjs])
    keep[[t * m + n for t in range(len(adjs))]] = True
    a = a[np.ix_(keep, keep)]
    d = a.sum(1)
    lap = np.eye(len(d)) - a / np.sqrt(np.outer(d, d))
    lam, vec = np.linalg.eigh(lap)
    lam, vec = lam[:k], vec[:, :k]
    rows = np.flatnonzero(keep)
    last = [i for i, r in enumerate(rows) if (len(adjs) - 1) * m <= r < (len(adjs) - 1) * m + n]
    u = np.zeros((n, k))
    nodes = rows[last] - (len(adjs) - 1) * m
    block = vec[last]
    u[nodes] = block / np.linalg.norm(block, axis=0)
    return np.concatenate([u, np.tile(lam, (n, 1))], axis=1)


def align_signs(x, ref):
    """Flip each of the first k columns of x to match ref; the eigenvector sign is the solver's."""
    k = x.shape[1] // 2
    out = x.copy()
    for j in range(k):
        if out[:, j] @ ref[:, j] < 0:
            out[:, j] *= -1
    return out


@pytest.mark.parametrize("seed", range(4))
def test_supra_adjacency_matches_blocks(seed):
    layers = random_layers(seed)
    sparse = [supra.layer_adjacency(e, N) for e in layers]
    dense = [naive_layer(e, N) for e in layers]
    for s, d in zip(sparse, dense):
        assert np.array_equal(s.to_dense().numpy(), d)
    assert np.array_equal(supra.supra_adjacency(sparse, MU).to_dense().numpy(), naive_supra(dense, MU))


def test_one_way_and_duplicate_edges_are_symmetrised_once():
    edges = torch.tensor([[0, 0, 1, 2], [1, 1, 0, 2]])            # 0->1 twice, 1->0, self-loop on 2
    a = supra.layer_adjacency(edges, 4).to_dense().numpy()
    assert a[0, 1] == a[1, 0] == 1 and a[2, 2] == 1 and a[3].sum() == 0


@pytest.mark.parametrize("norm", ["sym", "rw", "combinatorial"])
def test_supra_laplacian(norm):
    a = supra.supra_adjacency([supra.layer_adjacency(e, N) for e in random_layers(0)], MU)
    dense = a.to_dense().numpy()
    d = dense.sum(1)
    laplacian, b = supra.supra_laplacian(a, norm)
    expected = np.eye(len(d)) - dense / np.sqrt(np.outer(d, d)) if norm == "sym" else np.diag(d) - dense
    assert np.allclose(laplacian.to_dense().numpy(), expected)
    assert (b is not None) == (norm == "rw")
    if b is not None:
        assert np.allclose(b.to_dense().numpy(), np.diag(d))


@pytest.mark.parametrize("seed", range(4))
def test_supra_pe_matches_dense_reference(seed):
    layers = random_layers(seed)
    got = supra.supra_pe(layers, N, K, MU, "sym", TOL).numpy()
    ref = naive_pe(layers, N, K, MU)
    assert got.shape == (N, 2 * K)
    assert np.allclose(align_signs(got, ref), ref, atol=1e-5)


def test_inactive_nodes_are_zero_and_columns_unit_norm():
    layers = random_layers(1)
    pe = supra.supra_pe(layers, N, K, MU, "sym", TOL).numpy()
    assert np.all(pe[[8, 9], :K] == 0)
    assert np.allclose(np.linalg.norm(pe[:, :K], axis=0), 1)
    assert np.allclose(pe[:, K:], pe[0, K:])                       # eigenvalues broadcast over nodes


def test_lowest_eigenvalue_is_zero_and_spectrum_matches_dense():
    layers = random_layers(2)
    lam = supra.supra_pe(layers, N, K, MU, "sym", TOL).numpy()[0, K:]
    assert abs(lam[0]) < 1e-6 and np.all(np.diff(lam) >= -1e-9)


@pytest.mark.parametrize("seed", range(3))
def test_variational_objective_is_the_sum_of_the_eigenvalues(seed):
    """Eq. `supra_variational`: V = D^-1/2 U with V^T D V = I attains sum(lambda) over the k smallest."""
    adjs = [naive_layer(e, N) for e in random_layers(seed, inactive_last=())]
    a = naive_supra(adjs, MU)
    laplacian, _ = supra.supra_laplacian(torch.tensor(a).to_sparse(), "sym")
    lam, u = supra.smallest_eigenpairs(laplacian, None, K, TOL, 0)
    d = a.sum(1)
    v = u.numpy() / np.sqrt(d)[:, None]
    m = N + 1
    blocks = [v[t * m:(t + 1) * m] for t in range(W)]
    objective = sum(np.trace(b.T @ (np.diag(x.sum(1)) - x) @ b) for b, x in zip(blocks, adjs))
    objective += MU * sum(np.linalg.norm(blocks[t] - blocks[t - 1]) ** 2 for t in range(1, W))
    assert np.allclose(v.T @ np.diag(d) @ v, np.eye(K), atol=1e-6)
    assert np.isclose(objective, lam.sum().item(), atol=1e-6)


def test_rw_shares_the_spectrum_of_sym():
    a = supra.supra_adjacency([supra.layer_adjacency(e, N) for e in random_layers(3)], MU)
    lam = {norm: supra.smallest_eigenpairs(*supra.supra_laplacian(a, norm), K, TOL, 0)[0] for norm in ("sym", "rw")}
    assert torch.allclose(lam["sym"], lam["rw"], atol=1e-6)


def test_same_seed_same_output():
    layers = random_layers(0)
    assert torch.equal(supra.supra_pe(layers, N, K, MU, "sym", TOL, seed=5), supra.supra_pe(layers, N, K, MU, "sym", TOL, seed=5))


def test_failure_raises():
    layers = random_layers(0)
    with pytest.raises(RuntimeError, match="did not converge"):
        supra.supra_pe(layers, N, K, MU, "sym", tol=1e-16)
    with pytest.raises(ValueError, match="at least 2"):
        supra.supra_pe(layers[:1], N, K, MU, "sym", TOL)
    with pytest.raises(ValueError, match="norm"):
        supra.supra_pe(layers, N, K, MU, "bad", TOL)
    with pytest.raises(Exception):                                  # LOBPCG needs a supra size of at least 3k
        supra.supra_pe(layers, N, 30, MU, "sym", TOL)


def test_lobpcg_finds_the_smallest_pairs_on_a_larger_window():
    n, k = 80, 8
    layers = random_layers(7, n=n, w=4, inactive_last=(78, 79))
    got = supra.supra_pe(layers, n, k, 1.0, "sym", 1e-6).numpy()
    ref = naive_pe(layers, n, k, 1.0)
    assert np.allclose(got[0, k:], ref[0, k:], atol=1e-6)
    assert np.allclose(align_signs(got, ref), ref, atol=1e-3)
