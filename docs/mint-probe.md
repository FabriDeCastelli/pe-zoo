# Does Φ_p add to the edge count on MiNT? A probe

MiNT's Edge Growth/Shrink task (`graph_classification`, `target="edge_gs"`, `window=1`) is won by a rule with no
learning: score = −(edges in the snapshot), AUC 0.789 on the 20 test networks. This probe asks whether the
supra-Laplacian encoding adds anything to that count, and whether a different `w`, `k` or `μ` changes the answer.
The definition of Φ_p is untouched: only its parameters vary.

## Protocol (fixed before any test network was touched)

- **Networks:** 10 training networks with at least 280 snapshots, at evenly spaced quantiles of their median edges per
  snapshot (`examples/probe_mint_networks.json`). The 20 test networks are not used.
- **Probe:** a pooled logistic regression, fit on the train splits and scored by AUC on the val splits, per network then
  averaged. Features are standardised per network with its own train snapshots. Eigenvector signs and bases are
  arbitrary, so only sign- and basis-free summaries of Φ_p are used: eigenvalues 2..k, the log of the number of active
  nodes, and the log participation ratio of each eigenvector.
- **Feature sets:** the edge-count rule (no fit); topology (log edges, log active nodes); topology + Φ_p; Φ_p alone; topology +
  Φ_p summaries shuffled over time within each network (a control for capacity).
- **Arms:** the default (`w=8, k=8, μ=1`) and one factor at a time: `w ∈ {2, 4, 16}`, `k ∈ {4, 16}`, `μ ∈ {0.1, 10}`; `tol=1e-4`,
  binary symmetric graph as MiNT's own (see below). Windows at every 4th snapshot (consecutive MiNT snapshots overlap by 6/7),
  the same anchors for every arm (`probe_mint_features.py`); 1,635 train and 355 val samples. All windows converged.

The graph is already MiNT's: its code builds each snapshot as a plain `nx.Graph` (undirected, duplicates collapsed),
removes the diagonal, and passes no edge weights (`prepare()` returns `weights = None`); amounts only enter four pooled
degree features. On `mint-adx` and `mint-bag` 57% and 60% of the edge rows of a snapshot are duplicates, and self-loops are
a few dozen unique pairs, so pe-zoo's binary union is the same graph and no weighted variant is tried.

## Result (mean val AUC over the 10 networks)

| arm | edge count | topology | + Φ_p | Φ_p alone | + shuffled Φ_p | Φ_p gain | networks up |
|---|---|---|---|---|---|---|---|
| default | 0.772 | 0.772 | 0.665 | 0.616 | 0.732 | −0.107 | 3/10 |
| w=2 | 0.772 | 0.772 | 0.671 | 0.615 | 0.732 | −0.102 | 2/10 |
| w=4 | 0.772 | 0.772 | 0.689 | 0.615 | 0.718 | −0.084 | 2/10 |
| w=16 | 0.772 | 0.772 | 0.670 | 0.626 | 0.745 | −0.102 | 0/10 |
| k=4 | 0.772 | 0.772 | 0.739 | 0.687 | 0.742 | −0.033 | 4/10 |
| k=16 | 0.772 | 0.772 | 0.669 | 0.611 | 0.715 | −0.103 | 2/10 |
| μ=0.1 | 0.772 | 0.772 | 0.731 | 0.711 | 0.761 | −0.042 | 2/10 |
| μ=10 | 0.772 | 0.772 | 0.726 | 0.667 | 0.724 | −0.046 | 1/10 |

No arm lets Φ_p add to the topology features: the gain is negative everywhere, and run-to-run noise is about 0.015. Φ_p alone is
above 0.5 (0.61 to 0.71), so the summaries carry something, but nothing beyond what the edge count already gives. The loss
shrinks with fewer features (`k=4`) and where the real summaries look like the shuffled control. On the full 10,000-sample
default run (every window, `lag 0`) the picture is the same (0.689 for topology + Φ_p against 0.774). Adding the change
against the previous 14 snapshots made the baseline worse and Φ_p still hurt.

By the pre-registered rule (a gain above the noise on the validation splits), no configuration goes on to the test networks.

## What this does not show

- A pooled linear probe on sign-free summaries is weaker than a node-level model (MOSAIC's transformer with trajectory
  pooling). A null here does not clear a configuration of being useful there.
- About 35 validation samples per network, and labels that are strongly autocorrelated: per-network AUCs are noisy. The
  mean over 10 networks and the paired comparison against the same networks' topology baseline are what to read.
- One factor at a time, no interactions.

## Reproducing

```bash
OMP_NUM_THREADS=2 python examples/probe_mint_features.py --network mint-aioz --window 8 --k 8 --mu 1.0 --shard 0/8 --out OUT   # per network, arm and shard
python examples/probe_mint.py --features OUT --window 8 --k 8 --mu 1.0                                                          # the probe of one arm
```

The 640 jobs of the screen took about 8 minutes at 32 processes of 2 threads. Needs tgdata ≥ 0.4 (the MiNT networks).
