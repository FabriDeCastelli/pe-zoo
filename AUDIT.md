# Audit of the TIDES positional-encoding code

Files: `TIDES/training/src/engine/eigensolver.py` (E) and `positional_encodings.py` (P), read completely against the LaTeX and the paper (arXiv 2506.01596v1).
Read, not executed: each item is a reading of the code. Nothing was changed. "pe-zoo" says what the new package does instead.

## A. Divergences from the LaTeX / paper

| # | where | finding | pe-zoo |
|---|---|---|---|
| A1 | P:13-19, 531-550 | The inter-layer coupling `μ·I` covers nodes `0..N-2` only, so the global node (index N-1) is **not** coupled across layers. The paper's B = I is over all of V, global nodes included. | coupled |
| A2 | P:574-591 | Symmetrising with `cat([e, e.flip])` then `coalesce()` **sums duplicates**: an edge already present in both directions gets weight 2, a one-way edge weight 1 (`make_unique=False` by default, P:131). `edge_weight` is ignored (all ones). | union, weight 1 |
| A3 | P:552-559 | `circular=True` adds wrap-around coupling between the first and last layers. Not in the LaTeX (tridiagonal). Default False. | removed |
| A4 | P:353-358 | `operator_name='adj'` takes the *smallest* eigenpairs of the adjacency (the most negative), not L_sup. Not in the LaTeX. | removed |
| A5 | P:464-485, 133, 365-370 | `normalize_laplacian=False` (D−A) and the `normalize='pes'` row-normalisation are options outside the LaTeX; `normalize='ev'` (default) re-normalises columns that LOBPCG already returns orthonormal. | `norm` ∈ {sym, rw, combinatorial}; fixed `l2` |
| A6 | P:389-402, 428-458, 488-514 | Inactive rows are removed (as in the paper) but then **filled** with the previous window's PEs (`use_old_pes_for_non_active_nodes=True`, P:136, 443) or random unit vectors (P:438). After a slide by one step, supra row `τ` of this window is layer `τ+1` of the previous one, so the fill is misaligned in time. The fill also depends on call history. | zeros |
| A7 | E:104-114, 151-154 | `use_traj` returns the LOBPCG iteration history instead of the eigenpairs. Outside the LaTeX. | removed |

## B. Logic bugs

| # | where | finding |
|---|---|---|
| B1 | E:159-182, 137-140 | The `simple` solver is power iteration `V ← L V` with QR/L2 normalisation. It converges to the **largest** eigenpairs of L, not the smallest. Its Rayleigh-quotient eigenvalues are of the wrong end of the spectrum. |
| B2 | E:54-56 | The `exact` solver calls `torch.lobpcg`, not a dense solve (the dense one is `extreamly_exact`, E:58). The small-matrix fallback `L.shape[0] <= 5*m` (E:39-42) sends small problems to it, so small inputs go through LOBPCG with `niter` unset. Whether `torch.lobpcg` accepts `n < 3k` is not verified. |
| B3 | E:117-124, P:122-123 | Defaults `maxiter=10, tol=0`: LOBPCG runs at most 10 iterations and never stops early, so the result is an unconverged approximation of the k smallest. No residual is checked (the only check, E:147-149, is dead code: the local `verbose=False` shadows `self.verbose`). |
| B4 | E:125-135 | On any exception LOBPCG is silently retried with a new random start, printing a message. |
| B5 | E:88-94 | The random start uses the global torch RNG, unseeded. Results are not reproducible run to run. |
| B6 | P:46-60, 166, 222 | `LimitedList` pads the first call with `w-1` **copies of the first snapshot**, so the first `w-1` windows are not real windows. It is stateful, reset only when the caller passes `current_t == 0`. Skipped, repeated or out-of-order `t` yields wrong windows. |
| B7 | P:212-220 | If `edge_index` equals the previous call's, the cached PE is returned **without pushing the snapshot** into the window (and without the `current_t == 0` reset). A repeated snapshot leaves the window unchanged, though the earlier layers it contains have shifted. |
| B8 | P:325-331 | `no_information_leakage=True` returns the *previous* call's PE; on the first call `last_pes` is `None`. |
| B9 | P:235-256 | If the `try` around a precalculated load fails, `pes` is never assigned, yet the message says "resolving to calculation". The next use (P:315) raises `NameError`/`UnboundLocalError`. |
| B10 | P:240-249 | Sign flipping is applied only when loading from the on-disk cache, not when computing; it multiplies the **cached tensor in place** (`self.ev_dict`), so flips accumulate across calls. It also assumes the second half is the eigenvalues. |
| B11 | P:262-265, 318-320 | `pdb.set_trace()` on NaN. Hangs any non-interactive job. |
| B12 | P:127, 385, 571 | `num_nodes=None` by default but `num_nodes + 1` is used: a `TypeError` if it is not passed. |

## C. Cache-key problems

| # | where | finding |
|---|---|---|
| C1 | P:93, 602-609 | The key excludes `only_last_layer_pes` and `random_multiply_by_sign`. Two runs that differ in them share one directory, although the saved tensors have **different shapes** (`only_last_layer_pes`). |
| C2 | P:225-226 | Files are `pes_{prefix}_{t}.pt`. The key has no dataset name, no topology fingerprint and no version. Two datasets with equal `num_nodes` and parameters collide, and a rebuilt dataset silently reuses old files. |
| C3 | P:310-313 | Files are written straight to their final path (no temp + rename), so a crash leaves a truncated `.pt` that loads as valid or errors. |

## D. Performance, not correctness

- E:84: LOBPCG is always moved to CPU. This is the documented bottleneck.
- P:429-431, 509, 495-507: Python `set`s and per-layer Python loops over `n·w` nodes.

## Not bugs, checked

- Ordering (`LimitedList` oldest first, block offsets `t·N`) and the reshape to `(w, n, k)` then `transpose(0, 1)` (P:420-424) are consistent.
- The last-layer slice `supra_pe_mat[-N:-1]` (P:409) is the last layer without the global node.
- The trivial eigenpair is kept (matches j = 1..k).
- The normalised Laplacian (P:468-479) is `I − D^-½ A D^-½` with `0` for zero degree.

## Decision needed

Nothing in TIDES has been changed. For each group, should I (a) leave TIDES untouched (pe-zoo already avoids all of them), or (b) fix it there? My recommendation is (a) for everything: the fixes matter only if TIDES numbers are reused, and A1, A2, A6, B1, B3, B6 and C2 change results, so any TIDES run that used `TemporalPE` should be treated as using a different encoding from the LaTeX.
