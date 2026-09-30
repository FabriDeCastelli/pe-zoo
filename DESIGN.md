# pe-zoo — design

Cached supra-Laplacian positional encodings (Φ_p) of a sliding window of `w` snapshots, computed from the topology only.
Follows tsfm-zoo (src layout, uv, `.envrc`, tests, hashed cache). Single release: **v0.1.0**.

## Conventions

- `t` is a step of the series and **the last step of the window**: the window is `t-w+1..t`, its cache row is `t-w+1`.
  There is no `at` argument. Out-of-range `t` (`t < w-1` or `t >= T`) raises.
- Row `k` of the cache is the window starting at step `k`. Batches map to rows **only** through `task.window_starts`
  (first input step, in every task type). No per-task `t` conversion is documented or used.
- Φ_p of a window is the block of the **last step** of its eigenvectors, exactly `U_t` of eq. `supra_pe_def`.
- The cache must hold every window: built with `split=None, stride=1, horizon=0`, i.e. `window_starts == arange(T-w+1)`.
  This is recorded in `config.json` and checked on load; anything else raises.
- Symmetrisation and binarisation: `A ← 1[A + Aᵀ > 0]` (union of both directions, weight 1). Duplicates collapse, self-loops are kept.
  `edge_weight` is ignored (binary, as in Dwivedi et al. and GraphGPS).

## Equation → function (`src/pe_zoo/supra.py`)

| LaTeX | function | note |
|---|---|---|
| `supra_laplacian_def`, A_sup | `supra_adjacency(layers, mu)` | block tridiagonal, diagonal blocks A_τ, off-diagonals μ·I |
| `supra_laplacian_def`, L_sup | `supra_laplacian(A_sup, norm)` | `sym`: I − D^-½ A D^-½ (default); `rw`: (D−A)v = λDv, LOBPCG with B = D; `combinatorial`: D − A |
| `supra_variational` | (tests only) | the eigenvectors attain the objective's minimum, Σ λ; `rw` is its generalised problem |
| `supra_pe_def` | `supra_pe(layers, mu, k, ...)` | `[U_t ‖ 1 λᵀ]`, `[n, 2k]` |

Each docstring cites its equation. Tests compare every function to a dense, naive transcription of the LaTeX on small graphs.
If code and LaTeX disagree, stop and ask.

## Deliberate deviations from the LaTeX (stated in the docstrings)

1. **Global node** (paper arXiv 2506.01596v1): one per layer, joined with weight 1 to the *active* nodes of that layer, part of the vertex set
   (in D, and coupled across layers with μ, since the paper's B = I is over all of V). Supra size `(n_active + 1)·w`. Global rows are dropped
   from the output. TIDES leaves the global nodes uncoupled; that is an audit item.
2. **Inactive (node, layer) copies are removed** (degree 0 in the symmetrised A_τ; a self-loop counts). They get no global edge. Their output row is
   zeros in the eigenvector half; the eigenvalue half is still λ.
3. **`scale = l2`**: each eigenvector column is L2-normalised over the returned slice (active nodes of the last layer, no global node).
   The LaTeX unit norm is over the whole supra vector; the literature (GraphGPS default, Dwivedi) uses L2 per eigenvector, no √n rescaling.
   Fixed, not an option; recorded in the key as a constant.
4. **Binary weights** (see Conventions).

Not fixed on purpose: eigenvector sign and the basis inside a repeated eigenvalue are the solver's output (no sign convention),
deterministic for a given `seed`. The trivial eigenpair (λ≈0) is kept, as j = 1..k in the LaTeX.

## Solver

- `torch.lobpcg` only (`largest=False`), float64, sparse. Runs on a GPU chosen with `~/.claude/free-gpu` (never hardcoded; off-limits if another user's
  process is on it), or on CPU. Device and batch size are not in the key. Whether windows are batched or solved one at a time is decided by a benchmark.
- Converged output: after each solve the residual is recomputed independently; above the bound raises (no silent retry).
- Public parameters, and only these: `window` (≥ 2), `k = 8`, `mu = 1.0`, `norm = "sym"`, `tol = 1e-6`, `dtype = "float16"`, `device = "cpu"`.
- `MAXITER = 1000`, `SEED = 0` are module constants. Everything, `tol` included, is written into the config and the hash.
- Convergence: the residual `‖Lu − λBu‖/‖Bu‖` of every pair must be ≤ `tol` × the largest absolute row sum of L (the scale LOBPCG stops at); otherwise `RuntimeError`.
  The initial block of window `i` is drawn with seed `SEED + i`.
- Simplicity rule: no options beyond the above, no fallbacks, no retries, no state between windows; each function is a few lines and does one thing.
- `torch.lobpcg` needs a supra size well above `3k`; smaller problems raise a clear error (no dense fallback).

## Cache and API (`src/pe_zoo/encodings.py`, same pattern as `tsfm_zoo/embeddings.py`)

- Directory = first 16 hex chars of SHA-256 of the canonical JSON (`sort_keys`, compact separators) of:
  `dataset {name, tgdata version, num_nodes, num_steps, topology_sha256}`, `window`, `mu`, `norm`, `k`, `solver {name: lobpcg, tol, maxiter}`, `seed`,
  `sign: "solver"`, `scale: "l2"`, `weights: "binary"`, `global_node: "per-layer, coupled"`, `remove_inactive: true`, `windows {split: null, stride: 1, horizon: 0}`, `dtype`.
  Excluded: device, batch size.
- `topology_sha256` hashes `edge_index` and `edge_ptr` (edge_weight is ignored, so it is not hashed).
- `config.json` (`{config, hash, shape, seconds, pe_zoo}`) sits beside `enc.npy`. A run writes `<hash>.partial<pid>/` and renames it; a finished cache is reused.
- Storage `dtype`: `float16` default, or `float32` (numpy has no bfloat16). Test bounds the rounding error against float32. Solve is always float64.
- **Time-varying topology** (`edge_ptr` present): `enc.npy` is `[T-w+1, n, 2k]`.
- **Fixed topology**: the topology is cloned `w` times, one window is solved, `enc.npy` is `[n, 2k]`, `config.kind = "static"`.

## API

```python
enc = pe_zoo.encode_windows(g, cache_dir, window=12)   # PositionalEncodings (finished cache); k, mu, norm, tol, dtype, device optional
enc.at(t)          # [n, 2k], window t-w+1..t
enc.rows(task)     # [len(task), n, 2k] via task.window_starts (a broadcast view for a fixed topology)
enc.array, enc.config
```

`rows(task)` raises if `task.window != w` or the dataset name differs. `encode_windows` reads tgdata with `TGDATA_ROOT` and never downloads.

## Project

- `~/pe-zoo`: `src/pe_zoo/{supra.py,encodings.py,__init__.py}`, `tests/`, `docs/`, `.envrc`, `pyproject.toml` (hatchling), `uv.lock`, `DESIGN.md`, `AUDIT.md`.
- `.envrc`: venv `/raid/f.decastelli/venvs/pe-zoo`, `UV_CACHE_DIR`, `PE_ZOO_CACHE=/raid/f.decastelli/pe-zoo-cache`, `TGDATA_ROOT=/raid/f.decastelli/tgdata_build/_hubcache2`.
- Dependencies: `torch`, `numpy`; optional extra `tgdata` pinned to v0.3.0 (like tsfm-zoo); dev `pytest`. No scipy.
- Tests: dense naive LaTeX reference vs each function (small graphs, tiny hand-made ones and a real dynamic-edge dataset from `TGDATA_ROOT`); the variational
  objective; residual check; sign-invariant subspace comparison for repeated eigenvalues; cache hit/miss, key sensitivity (each field changes the hash, device/batch size do not),
  atomic rename, `at`/`rows` consistency, `window_starts == arange` check, the float16 error bound. Run after each change.
- Docs in tgdata's README style: centred title, tagline, Install, "What you can do" (bold verb-led headings, one tiny code block each), "Learn more" → `docs/`.
- Git: init `main`, remote `https://github.com/FabriDeCastelli/pe-zoo.git`. One commit per feature, messages describe the change only (no attribution lines), one annotated tag `v0.1.0`
  at the end. Push once the features are done and the tests pass.

## AUDIT.md (next, after this design is confirmed)

Every logic bug or divergence found in `TIDES/training/src/engine/{eigensolver.py,positional_encodings.py}` against the LaTeX and the paper, with line references.
Nothing in TIDES is changed or copied without asking.
