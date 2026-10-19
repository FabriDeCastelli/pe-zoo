# Laplacian positional encodings (the baseline)

The plain Laplacian positional encoding of Dwivedi & Bresson, computed on each snapshot independently, to compare with the
[supra-Laplacian encoding](encoding.md). `lappe.laplacian_positional_encoding` is copied unchanged from
[graphdeeplearning/graphtransformer](https://github.com/graphdeeplearning/graphtransformer/blob/654068c3e0924dea7041b9c3d84df52bba72ed4d/data/molecules.py)
(commit `654068c`, MIT licence), the code of "A Generalization of Transformer Networks to Graphs". A test compares its text
with the upstream lines, character for character. The same computation is in
[benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns), which also zero-pads the output when the graph
has no more than `k` nodes; pe-zoo follows graphtransformer, and raises instead.

```python
enc = pe_zoo.encode_snapshots(g, "cache/", k=8)   # [steps, nodes, 8], or [nodes, 8] for a fixed topology
enc.at(t)                                         # the snapshot at step t
enc.rows(task)                                    # for each sample, the snapshot at the last step of its window
```

## What the function does

`L = I − N A N` with `N = diag(max(in-degree, 1))^{-1/2}`, then `np.linalg.eig` on the dense `L`, the eigenvectors sorted by
eigenvalue, and columns `1..k` kept (the first is skipped). The output has no eigenvalues, no normalisation and no sign
convention; upstream flips the sign of each eigenvector at random per batch *in its training loop*, which is not part of
the function and is not applied here.

## What pe-zoo adds around it

- A stand-in for the DGL graph the function takes (`lappe.Graph`: `number_of_nodes`, `in_degrees`, `adjacency_matrix_scipy`,
  `ndata`) and for `dgl.backend.asnumpy`, so DGL is not needed.
- The graph is the symmetric binary adjacency of the snapshot (`A ← 1[A + Aᵀ > 0]`), the same as the supra-Laplacian
  encoding uses without its global node. The upstream datasets are bidirected, so the function sees the kind of graph it was
  written for.
- `lappe.lappe(edge_index, n, k)` raises if `n ≤ k`.
- `lappe.lappe_active(edge_index, n, k)` is the same encoding at the active nodes (those with an edge), solved on them plus
  `min(k, n − m)` isolated nodes (`m` active), which is what the whole-graph run reduces to: an isolated node is a block `[1]`
  of `L`, and only `k` of them can reach the kept columns. `L` is the upstream one; being symmetric it goes to
  `scipy.linalg.eigh` for its lowest `k + 1` eigenpairs instead of the general `np.linalg.eig`. The active rows equal
  `lappe`'s up to the eigenvectors' sign and basis; the inactive rows are zero, where the whole-graph run puts unit vectors
  on `k` of them. `encode_snapshots` uses it.
- The cache of [cache.md](cache.md): the directory is the hash of the dataset, `k`, `dtype` and the method (function and
  upstream commit). `at(t)` is snapshot `t`; `rows(task)` is, for each sample, the snapshot at `window_starts + window − 1`, the
  step at which the supra-Laplacian encoding of the same window is read. Any window length works, since a snapshot does not
  depend on it.

## Things to know

- **Many components.** On a sparse snapshot the graph has many connected components, each with its own eigenvalue 0, and
  upstream skips only the first eigenvector. Isolated nodes have eigenvalue 1, so the lowest eigenvectors live on the active
  nodes and the other rows are zero. On `twittertennis-rg17` about 81% of the node rows of a snapshot are all zero.
- **Cost.** A dense eigendecomposition per snapshot, cubic in the nodes it sees. `lappe` (`np.linalg.eig`) sees all `n`;
  `lappe_active` (and so `encode_snapshots`) sees the active ones plus at most `k`, hundreds in a weekly snapshot of a
  transfer network that has over 100,000 nodes, with `eigh` for the lowest `k + 1` pairs: on 6 threads 0.04 s at 1000
  nodes, 0.4 s at 3000, 5 s at 6000, 58 s at 12000, against 0.6 s, 12 s and more for `np.linalg.eig` at 1000 and 3000.
  A fixed topology is encoded once.
- **Failure of LAPACK's subset drivers.** `lappe_active` asks `scipy.linalg.eigh` for the lowest `k + 1` pairs only. On some
  snapshots with many repeated eigenvalues (a `mint-drgn` snapshot with 20 components) those drivers raise `LinAlgError`;
  it then computes the full decomposition (`driver="evd"`) and keeps the same columns. A regression test holds that snapshot.
- **Repeated eigenvalues** have an arbitrary basis, as for any eigendecomposition. This is the common case on a sparse
  snapshot: every connected component adds an eigenvalue 0, so the kept columns are a choice inside that eigenspace.
- **`lappe_active` against `lappe`.** The Laplacian is block diagonal over the active nodes and the isolated ones, so both
  have the same lowest eigenvalues and eigenspaces; the columns are the same up to sign and, in a repeated eigenvalue, basis.
  Measured on `twittertennis-rg17` (snapshots 0, 30, 60, 90, 119, `k = 8`, 4 to 14 components): the lowest `k + 1`
  eigenvalues agree, and the mean cosine of the principal angles between the two sets of columns at the active nodes is
  0.64 to 0.96, the same as between `lappe` and `lappe` run on the same graph with relabelled nodes (0.74 to 0.96). So their
  difference is within the arbitrariness of upstream itself. The inactive rows differ in the other case: when the active
  graph has `k` or fewer eigenvalues below 1, the kept columns include eigenvalue 1, and `lappe` puts unit vectors on isolated
  nodes where `lappe_active` has zeros.
