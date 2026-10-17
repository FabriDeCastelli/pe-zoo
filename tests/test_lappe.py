"""The Laplacian positional encoding of Dwivedi & Bresson, as copied from graphdeeplearning/graphtransformer, and its cache."""
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_encodings import K, N, T, W, toy_graph

from pe_zoo import encode_snapshots, lappe

UPSTREAM = Path(__file__).with_name("upstream_laplacian_positional_encoding.txt")


def ring_with_chords(n=14, seed=0):
    rng = np.random.default_rng(seed)
    pairs = [(i, (i + 1) % n) for i in range(n)] + [tuple(rng.choice(n, 2, replace=False)) for _ in range(8)]
    return torch.tensor(pairs, dtype=torch.long).T


def test_the_function_is_the_upstream_text():
    """Lines 168-184 of data/molecules.py at graphtransformer@654068c, character for character."""
    assert inspect.getsource(lappe.laplacian_positional_encoding).rstrip("\n") == UPSTREAM.read_text()


def test_matches_a_dense_reference():
    n, k = 14, 3
    edges = ring_with_chords(n)
    a = np.zeros((n, n))
    a[edges[0], edges[1]] = a[edges[1], edges[0]] = 1.0
    scale = a.sum(1).clip(1) ** -0.5
    values, vectors = np.linalg.eigh(np.eye(n) - scale[:, None] * a * scale[None, :])
    pe = lappe.lappe(edges, n, k).numpy()
    assert pe.shape == (n, k) and pe.dtype == np.float32
    for j in range(k):                                              # eigenvector 1+j, up to sign
        ref = vectors[:, 1 + j]
        assert np.allclose(pe[:, j] * np.sign(pe[:, j] @ ref), ref, atol=1e-5)


def test_one_way_edges_are_symmetrised_before_the_function_sees_them():
    edges = ring_with_chords()
    both = torch.cat([edges, edges.flip(0)], dim=1)
    assert torch.equal(lappe.lappe(edges, 14, 3), lappe.lappe(both, 14, 3))


def test_the_dgl_stand_in():
    graph = lappe.Graph(torch.tensor([[0, 0, 2], [1, 2, 1]]), 4)
    assert graph.number_of_nodes() == 4 and graph.in_degrees().tolist() == [0, 2, 1, 0]
    assert graph.adjacency_matrix_scipy(return_edge_ids=False).toarray().tolist() == [[0, 1, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]


def test_too_few_nodes_raises():
    with pytest.raises(ValueError, match="fewer than k"):
        lappe.lappe(ring_with_chords(5), 5, 5)


def test_snapshot_cache_rows_and_lookup(tmp_path):
    g = toy_graph()
    enc = encode_snapshots(g, tmp_path, k=K)
    assert enc.array.shape == (T, N, K) and enc.is_static is False
    for step in (0, 7, T - 1):
        assert np.allclose(enc.array[step].astype(np.float32), lappe.lappe(torch.from_numpy(g.edges_at(step)[0]), N, K).numpy(), atol=1e-3)
        assert np.array_equal(enc.at(step), enc.array[step])
    with pytest.raises(IndexError):
        enc.at(T)
    for window in (W, 2):                                           # the encoding of the last snapshot of each window
        task = g.task("node_forecasting", window=window, horizon=2, split="test")
        assert np.array_equal(enc.rows(task), enc.array[np.asarray(task.window_starts) + window - 1])


def test_snapshot_cache_fixed_topology_reuse_and_key(tmp_path):
    g = toy_graph(static=True)
    enc = encode_snapshots(g, tmp_path, k=K)
    assert enc.array.shape == (N, K) and enc.is_static
    task = g.task("node_forecasting", window=W, horizon=2, split="test")
    assert enc.rows(task).shape == (len(task), N, K)
    stamp = (enc.path / "config.json").stat().st_mtime_ns
    assert encode_snapshots(g, tmp_path, k=K).path == enc.path and (enc.path / "config.json").stat().st_mtime_ns == stamp
    assert len({enc.path, encode_snapshots(g, tmp_path, k=K + 1).path, encode_snapshots(g, tmp_path, k=K, dtype="float32").path,
                encode_snapshots(toy_graph(static=True, name="other"), tmp_path, k=K).path}) == 4
    config = json.loads((enc.path / "config.json").read_text())["config"]
    assert config["method"]["source"].endswith("data/molecules.py") and "window" not in config
    with pytest.raises(ValueError, match="not other"):
        enc.rows(toy_graph(static=True, name="other").task("node_forecasting", window=W, horizon=2))
