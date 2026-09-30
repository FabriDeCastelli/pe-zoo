# Encoding diagnostics

Plots that show how the supra-Laplacian encodings of a time-varying [tgdata](https://github.com/FabriDeCastelli/tgdata)
dataset move with its topology. The encodings come from the [window cache](cache.md), so a rerun is instant. Needs the
`plot` extra (`uv sync --extra plot`).

```bash
python examples/plot_encodings.py --dataset twittertennis-rg17 --window 4
python examples/plot_encodings.py --path /path/to/a/local/tgdata/build --window 4 --nodes 3 17
```

Without `--nodes` it takes the three nodes with the most edges. Figures are saved under `outputs/plots/<dataset>/`.
A fixed-topology dataset has one encoding for every window, so it is not plotted over time (`plot_window` still works).

## Figures

For every window length (`w<window>_*.png`):

| file | shows | read it as |
|---|---|---|
| `eigenvalues` | the k smallest eigenvalues of each window over time | drift or a rhythm in the spectrum follows the topology; λ₁ ≈ 0 is the trivial pair |
| `components` | \|entry\| of the first three eigenvectors at chosen nodes over time | a node whose magnitude tracks its activity is being placed by its edges; flat lines mean the encoding is not moving |
| `window` | one window as nodes by eigenvector, with its eigenvalues | large isolated entries are eigenvectors localized on a few nodes; the label gives the share of all-zero rows |
| `matrix` | subspace similarity between every pair of windows | blocks are stretches of stable topology; a repeating pattern off the diagonal is a recurring topology |
| `lags` | mean subspace similarity of windows `lag` steps apart, next to the edge-set Jaccard similarity of steps `lag` apart (dashed) | peaks at the same lags in both mean the encoding follows a recurrence of the topology |

## In code

```python
from pe_zoo import plots

pe, times = enc.array[:], np.arange(window - 1, g.num_steps)      # times: the last step of each window
fig = plots.plot_eigenvalues(times, pe)
fig = plots.plot_components(times, pe, nodes=[0, 5], components=3)
fig = plots.plot_window(pe[100])                                   # one window: [nodes, 2k]
fig = plots.plot_similarity_matrix(times, pe, start=10, end=60)
lags = np.arange(1, 31)
fig = plots.plot_lag_curves(lags, {"encoding": plots.lag_similarity(pe, lags), "topology": plots.edge_jaccard(g, lags)})
```

Every function returns the matplotlib figure and neither shows nor saves it. `window_similarity(pe)` is the matrix behind
`matrix`.

## Things to know

- **Signs are arbitrary.** Each window is solved on its own, so an eigenvector's sign changes between windows. The time
  plots show absolute values, and the similarities compare the subspaces spanned by the eigenvectors: mean cosine of the
  principal angles, `1` for the same subspace whatever the signs and the basis, `0` for orthogonal ones.
- **The two curves in `lags` are on different scales.** Jaccard similarity of sparse edge sets is low by nature; compare
  where the curves peak, not their heights.
- **The similarity matrix is quadratic in the number of windows.** Show a range (`start`, `end`) for a long series.
