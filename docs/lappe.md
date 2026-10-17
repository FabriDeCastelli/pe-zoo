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
- The cache of [cache.md](cache.md): the directory is the hash of the dataset, `k`, `dtype` and the method (function and
  upstream commit). `at(t)` is snapshot `t`; `rows(task)` is, for each sample, the snapshot at `window_starts + window − 1`, the
  step at which the supra-Laplacian encoding of the same window is read. Any window length works, since a snapshot does not
  depend on it.

## Things to know

- **Many components.** On a sparse snapshot the graph has many connected components, each with its own eigenvalue 0, and
  upstream skips only the first eigenvector. Isolated nodes have eigenvalue 1, so the lowest eigenvectors live on the active
  nodes and the other rows are zero. On `twittertennis-rg17` about 81% of the node rows of a snapshot are all zero.
- **Cost.** `np.linalg.eig` of a dense `n × n` matrix, per snapshot: about 0.08 s at `n = 1000`, cubic after that. A fixed
  topology is encoded once.
- **Repeated eigenvalues** have an arbitrary basis, as for any eigendecomposition.
