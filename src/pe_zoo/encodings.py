"""Supra-Laplacian positional encodings of every window of a temporal-graph dataset, cached on disk under a hash of their configuration.

Row k of the saved array is the encoding of the window of steps k, ..., k + window - 1 (tgdata's "every window of a
series, in order" recipe), so any split of the dataset maps onto it through `task.window_starts`. A dataset whose
topology is fixed has the same encoding in every window: one `[nodes, 2k]` array is saved instead.
"""
import hashlib
import json
import logging
import os
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from . import supra

logger = logging.getLogger("pe_zoo")

ARRAY_FILE, CONFIG_FILE = "enc.npy", "config.json"
EVERY_WINDOW = {"split": None, "stride": 1, "horizon": 0}
DTYPES = ("float32", "float16")
# What the encoding is, beyond the parameters: constants of the implementation, hashed with them.
METHOD = {"solver": "lobpcg", "maxiter": supra.MAXITER, "seed": supra.SEED, "sign": "solver", "scale": "l2",
          "weights": "binary", "global_node": "per layer, coupled", "remove_inactive": True,
          "inactive_rows": "latest active row of the window, else zero"}


def config_hash(config: dict) -> str:
    """16 hex characters of the SHA-256 of the canonical JSON of `config`."""
    text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def topology_fingerprint(g) -> str:
    """SHA-256 of the edges (weights are ignored by the encoding), so a rebuilt dataset of the same name gets its own cache."""
    digest = hashlib.sha256()
    for array in (g.edge_index.T, g.edge_ptr):
        if array is not None:
            for start in range(0, len(array), 1 << 20):
                digest.update(np.ascontiguousarray(array[start:start + (1 << 20)], dtype=np.int64).tobytes())
    return digest.hexdigest()[:16]


class PositionalEncodings:
    """A finished cache directory: `array` is [windows, nodes, 2k] (or [nodes, 2k] for a fixed topology)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.config = json.loads((self.path / CONFIG_FILE).read_text())["config"]
        if self.config["windows"] != EVERY_WINDOW:
            raise ValueError(f"{self.path.name} was not built from every window (split=None, stride=1, horizon=0)")

    @property
    def array(self) -> np.ndarray:
        return np.load(self.path / ARRAY_FILE, mmap_mode="r")

    @property
    def is_static(self) -> bool:
        return self.array.ndim == 2

    def at(self, t: int) -> np.ndarray:
        """[nodes, 2k]: the encoding of the window of steps t - window + 1, ..., t (the encoding is that of step t)."""
        window, steps = self.config["window"], self.config["dataset"]["num_steps"]
        if not window - 1 <= t < steps:
            raise IndexError(f"t={t} has no window of {window} steps ending at it in a series of {steps} steps")
        return self.array if self.is_static else self.array[t - window + 1]

    def rows(self, task) -> np.ndarray:
        """[samples, nodes, 2k]: one encoding per sample of `task` (any tgdata task of any split), by its `window_starts`."""
        if task.g.name != self.config["dataset"]["name"] or task.window != self.config["window"]:
            raise ValueError(f"{self.path.name} holds windows of {self.config['window']} steps of "
                             f"{self.config['dataset']['name']}, not {task.window} of {task.g.name}")
        starts = np.asarray(task.window_starts)
        if self.is_static:
            return np.broadcast_to(self.array, (len(starts), *self.array.shape))
        return self.array[starts]


def encode_windows(g, cache_dir, *, window: int, k: int = 8, mu: float = 1.0, norm: str = "sym", tol: float = 1e-6,
                   dtype: str = "float16", device="cpu") -> PositionalEncodings:
    """Encode every window of `g` with `supra.supra_pe`, or return the cache of an identical earlier run.

    `g` is a tgdata dataset. Everything that determines the result, and nothing else (not the device), is
    hashed into the directory name; `config.json` holds it in full.
    """
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
    if g.edge_index is None:
        raise ValueError(f"{g.name} has no topology")
    if g.num_steps < window:
        raise ValueError(f"{g.name} has {g.num_steps} steps, fewer than window={window}")
    config = {
        "dataset": {"name": g.name, "tgdata": _tgdata_version(), "num_nodes": g.num_nodes, "num_steps": g.num_steps,
                    "topology_sha256": topology_fingerprint(g)},
        "window": window, "k": k, "mu": mu, "norm": norm, "tol": tol, "dtype": dtype, "windows": EVERY_WINDOW,
        "method": METHOD,
    }
    path = Path(cache_dir) / config_hash(config)
    if (path / CONFIG_FILE).exists():
        logger.info("encodings cached at %s", path)
        return PositionalEncodings(path)

    windows = 1 if g.is_static else g.num_steps - window + 1
    shape = (g.num_nodes, 2 * k) if g.is_static else (windows, g.num_nodes, 2 * k)
    logger.info("encoding %d window(s) of %s: %s", windows, g.name, json.dumps(config))
    partial = path.with_name(f"{path.name}.partial{os.getpid()}")
    partial.mkdir(parents=True)
    out = np.lib.format.open_memmap(partial / ARRAY_FILE, mode="w+", dtype=dtype, shape=shape)
    start = time.time()
    for row in range(windows):
        edges = [g.edge_index] * window if g.is_static else [g.edges_at(step)[0] for step in range(row, row + window)]
        edges = [torch.tensor(e, dtype=torch.long, device=device) for e in edges]
        pe = supra.supra_pe(edges, g.num_nodes, k, mu, norm, tol, seed=supra.SEED + row)
        value = pe.cpu().numpy().astype(dtype)
        if g.is_static:
            out[:] = value
        else:
            out[row] = value
        if row % 500 == 0:
            logger.info("window %d/%d, %.0fs", row, windows, time.time() - start)
    out.flush()
    del out
    finished = {"config": config, "hash": path.name, "shape": list(shape), "seconds": round(time.time() - start, 1),
                "pe_zoo": version("pe-zoo")}
    (partial / CONFIG_FILE).write_text(json.dumps(finished, indent=1))
    try:
        partial.rename(path)                                                    # a cache is complete or absent
    except OSError:
        shutil.rmtree(partial)                                                  # another run finished first
    return PositionalEncodings(path)


def _tgdata_version() -> str:
    import tgdata
    return tgdata.__version__
