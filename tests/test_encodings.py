import json
from importlib.metadata import version

import numpy as np
import pytest
import torch
from tgdata.schema import Split, TemporalGraph

from pe_zoo import encode_windows, supra
from pe_zoo.encodings import config_hash

N, T, W, K = 12, 16, 4, 3


def toy_graph(static=False, name="toy", shift=0):
    """N nodes in a ring plus, at step t, chords that move with t (unless static)."""
    edges, ptr = [], [0]
    for t in range(1, T + 1) if not static else [1]:
        step = np.array([[i, (i + 1) % N] for i in range(N)] + [[i, (i + 2 + (0 if static else (t + shift) % 3)) % N] for i in range(0, N, 2)]).T
        edges.append(step)
        ptr.append(ptr[-1] + step.shape[1])
    edge_index = np.concatenate(edges, axis=1)
    return TemporalGraph(name=name, domain="toy", tasks=["node_forecasting"], time_mode="discrete",
                         x=np.random.default_rng(0).normal(size=(T, N, 1)).astype("float32"),
                         edge_index=edge_index, edge_weight=np.ones(edge_index.shape[1], dtype="float32"),
                         edge_ptr=None if static else np.array(ptr), meta={"num_nodes": N, "freq": "1h"},
                         splits={"default": Split(fractions={"train": 0.5, "val": 0.25, "test": 0.25})})


def encode(g, tmp_path, **kwargs):
    return encode_windows(g, tmp_path, window=W, k=K, dtype="float32", **kwargs)


def test_rows_are_supra_pe_of_each_window(tmp_path):
    g = toy_graph()
    enc = encode(g, tmp_path)
    assert enc.array.shape == (T - W + 1, N, 2 * K)
    for start in (0, 5, T - W):
        layers = [torch.from_numpy(g.edges_at(s)[0]) for s in range(start, start + W)]
        expected = supra.supra_pe(layers, N, K, 1.0, "sym", 1e-6, seed=supra.SEED + start).float().numpy()
        assert np.allclose(enc.array[start], expected, atol=1e-6)


def test_at_is_the_window_ending_at_t(tmp_path):
    enc = encode(toy_graph(), tmp_path)
    for t in (W - 1, 7, T - 1):
        assert np.array_equal(enc.at(t), enc.array[t - W + 1])
    for bad in (W - 2, T, -1):
        with pytest.raises(IndexError):
            enc.at(bad)


def test_rows_follow_window_starts_for_any_split_and_task(tmp_path):
    g = toy_graph()
    enc = encode(g, tmp_path)
    for split in ("train", "val", "test"):
        task = g.task("node_forecasting", window=W, horizon=2, split=split)
        assert np.array_equal(enc.rows(task), enc.array[task.window_starts])
    with pytest.raises(ValueError, match="not 3"):
        enc.rows(g.task("node_forecasting", window=3, horizon=2))


def test_every_window_row_k_starts_at_step_k(tmp_path):
    g = toy_graph()
    task = g.task("node_forecasting", window=W, horizon=0, split=None)
    assert np.array_equal(task.window_starts, np.arange(T - W + 1))


def test_fixed_topology_is_one_array_broadcast_to_every_window(tmp_path):
    g = toy_graph(static=True)
    enc = encode(g, tmp_path)
    assert enc.array.shape == (N, 2 * K) and enc.is_static
    task = g.task("node_forecasting", window=W, horizon=2, split="test")
    rows = enc.rows(task)
    assert rows.shape == (len(task), N, 2 * K) and np.array_equal(rows[0], enc.at(W - 1))
    layers = [torch.from_numpy(g.edge_index)] * W
    assert np.allclose(enc.array, supra.supra_pe(layers, N, K, 1.0, "sym", 1e-6).float().numpy(), atol=1e-6)


def test_cache_is_reused_and_written_atomically(tmp_path):
    g = toy_graph()
    first = encode(g, tmp_path)
    stamp = (first.path / "config.json").stat().st_mtime_ns
    second = encode(g, tmp_path)
    assert second.path == first.path and (second.path / "config.json").stat().st_mtime_ns == stamp
    assert [p.name for p in tmp_path.iterdir()] == [first.path.name]              # no .partial left
    finished = json.loads((first.path / "config.json").read_text())
    assert finished["hash"] == first.path.name == config_hash(finished["config"])
    assert finished["pe_zoo"] == version("pe-zoo") and finished["shape"] == [T - W + 1, N, 2 * K]


def test_every_field_changes_the_directory_but_device_does_not(tmp_path):
    g = toy_graph()
    base = encode(g, tmp_path).path
    assert encode(g, tmp_path, device=torch.device("cpu")).path == base
    others = [encode(g, tmp_path, mu=0.5), encode(g, tmp_path, norm="rw"), encode(g, tmp_path, tol=1e-7),
              encode_windows(g, tmp_path, window=W + 1, k=K, dtype="float32"),
              encode_windows(g, tmp_path, window=W, k=K + 1, dtype="float32"),
              encode_windows(g, tmp_path, window=W, k=K, dtype="float16"),
              encode(toy_graph(name="other"), tmp_path), encode(toy_graph(shift=1), tmp_path)]
    assert len({base, *(e.path for e in others)}) == 1 + len(others)


def test_float16_storage_error_is_small(tmp_path):
    g = toy_graph()
    full, half = encode(g, tmp_path), encode_windows(g, tmp_path, window=W, k=K, dtype="float16")
    assert half.array.dtype == np.float16
    assert np.abs(full.array - half.array.astype(np.float32)).max() < 1e-3


def test_a_cache_not_built_from_every_window_is_refused(tmp_path):
    enc = encode(toy_graph(), tmp_path)
    path = enc.path / "config.json"
    saved = json.loads(path.read_text())
    saved["config"]["windows"]["stride"] = 2
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="every window"):
        type(enc)(enc.path)


def test_bad_inputs(tmp_path):
    with pytest.raises(ValueError, match="dtype"):
        encode_windows(toy_graph(), tmp_path, window=W, dtype="bfloat16")
    with pytest.raises(ValueError, match="fewer than"):
        encode_windows(toy_graph(), tmp_path, window=T + 1)


def test_real_dataset_pems08(tmp_path):
    tgdata = pytest.importorskip("tgdata")
    try:
        g = tgdata.load("pems08")
    except Exception as error:                                                    # not in TGDATA_ROOT
        pytest.skip(str(error))
    enc = encode_windows(g, tmp_path, window=12)
    task = g.task(split="test")
    rows = enc.rows(task)
    assert enc.array.shape == (g.num_nodes, 16) and rows.shape == (len(task), g.num_nodes, 16)
    assert np.isfinite(enc.array.astype(np.float32)).all() and np.allclose(np.linalg.norm(enc.array[:, :8].astype(np.float32), axis=0), 1, atol=1e-2)
