# The encoding

For a window of `w` snapshots ending at step `t`, pe-zoo builds the supra-adjacency `A_sup` (block tridiagonal:
`A_τ` on the diagonal, `μ·I` between adjacent snapshots), the supra-Laplacian
`L_sup = I − D_sup^{-1/2} A_sup D_sup^{-1/2}`, its `k` smallest eigenpairs by LOBPCG, and returns
`Φ_p = [U_t ‖ 1_n λᵀ]` of shape `[n, 2k]`: the block of the last snapshot of the eigenvectors, next to the eigenvalues.
Only the topology is used.

| equation | function (`pe_zoo.supra`) |
|---|---|
| `A_τ` | `layer_adjacency(edge_index, n)` |
| `A_sup` | `supra_adjacency(layers, mu)` |
| `L_sup` | `supra_laplacian(adjacency, norm)` |
| eigenpairs | `smallest_eigenpairs(laplacian, b, k, tol, seed)` |
| `Φ_p` | `supra_pe(edge_layers, n, k, mu, norm, tol)` |

Every function is tested against a dense, naive transcription of the LaTeX, and the variational objective is checked
to equal the sum of the `k` eigenvalues.

## Operators

`norm="sym"` (default) is `I − D^{-1/2} A D^{-1/2}`. `"rw"` solves `(D − A)v = λDv`, the generalised problem of the
variational objective (`VᵀDV = I`). `"combinatorial"` is `D − A`.

## Deviations from the equations

1. **A global node per snapshot**, joined with weight 1 to the nodes that have an edge in that snapshot and coupled to
   its copies in adjacent snapshots by `μ` (as in [arXiv 2506.01596](https://arxiv.org/abs/2506.01596)). The global rows
   are not returned.
2. **Inactive (node, snapshot) copies are removed** from the supra graph. A node with no edge in the last snapshot gets
   zeros in the eigenvector half; its eigenvalue half is still `λ`.
3. **Each eigenvector column is L2-normalised** over the returned block (the active nodes of the last snapshot). The
   solver's unit norm is over the whole supra vector.
4. **The adjacency is binary and symmetric**: `A ← 1[A + Aᵀ > 0]`. `edge_weight` is ignored; self-loops are kept.

## Not fixed on purpose

The sign of an eigenvector, and the basis inside a repeated eigenvalue, are whatever the solver returns (deterministic
for the seed, `SEED + window index`). The trivial eigenpair (`λ ≈ 0`) is kept, as `j = 1..k` in the equation.

## Convergence

The solver is `torch.lobpcg` in float64. After each solve the residual `‖Lu − λBu‖/‖Bu‖` of every pair is recomputed;
if any is above `tol` times the largest absolute row sum of `L`, a `RuntimeError` is raised. There is no retry and no
fallback. LOBPCG needs a supra size of at least `3k`, and `window ≥ 2`.
