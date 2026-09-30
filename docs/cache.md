# Cache and lookup

## The `t` convention

`t` is a step of the series and the **last** step of the window: `enc.at(t)` is the encoding of steps
`t − w + 1, ..., t`. It raises `IndexError` unless `w − 1 ≤ t < T`. Nothing is padded.

Row `k` of the saved array is the window that *starts* at step `k`, so `at(t)` reads row `t − w + 1`. Batches map to rows
only through `task.window_starts` (the first input step of each sample, in every tgdata task), which is what
`enc.rows(task)` uses. It raises if the task's `window` or dataset differs from the cache.

The cache always holds every window (`split=None, stride=1, horizon=0`). This is written to `config.json`; opening a
cache that says otherwise raises.

## Layout

```
<cache_dir>/<hash>/enc.npy       [T − w + 1, n, 2k]    (or [n, 2k] for a fixed topology)
<cache_dir>/<hash>/config.json   {config, hash, shape, seconds, pe_zoo}
```

`<hash>` is the first 16 hex characters of the SHA-256 of the canonical JSON of `config`: the dataset (name, tgdata
version, number of nodes and steps, a fingerprint of `edge_index` and `edge_ptr`), `window`, `k`, `mu`, `norm`, `tol`,
`dtype`, the every-window flags, and the constants of the method (solver, iteration limit, seed, sign, scale, weights,
global node, node removal). The device is not part of it.

A run writes `<hash>.partial<pid>/` and renames it when done, so a cache is complete or absent. A finished cache is
reused.

## Fixed and time-varying topology

With `edge_ptr`, each window has its own snapshots and every window is solved. Without it, the topology is cloned `w`
times, one window is solved, and `enc.array` is `[n, 2k]`; `at` returns it and `rows(task)` returns a read-only
broadcast view `[samples, n, 2k]`.

## Storage type

`dtype="float16"` (default) or `"float32"`. The solve is always float64; the array is cast when written. After L2
normalisation the entries are of order `1/√n`, well inside float16's range; the tests bound the rounding error.
