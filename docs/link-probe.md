# Does Φ_p help predict the next snapshot's edges? A probe on AS-733 and Bitcoin-Alpha

Same style as the [MiNT probe](mint-probe.md). Data: the topology of `tgdata.converters.temporal_link` builds of the LAMP release
(FabrizioDeCastelli/temporal-link-level-prediction); candidate pairs and 1:1 negatives are read from the release's files. From the window of
w snapshots ending at t, score the pairs for the edges of snapshot t + 1. Features: persistence, share of the window holding the pair, log
degree sum, log common neighbours (baseline), and the product of the two nodes' eigenvector rows of Φ_p (sign-free). Logistic regression fit on
train steps, AUC on val steps; the test split is untouched. `examples/probe_link.py`.

## AS-733 (val AUC on pairs that are not edges of snapshot t, i.e. new links; 48.7k pairs)

Overall AUC is 0.995 for persistence alone (98.7% of next edges are already present) and 0.999+ with the baseline, so it cannot discriminate.

| arm | baseline | + Φ_p | change | + shuffled Φ_p | Φ_p alone |
|---|---|---|---|---|---|
| w=8, k=8, μ=1 | 0.9618 | 0.9616 | −0.0001 | 0.9616 | 0.868 |
| w=2 | 0.9238 | 0.9276 | **+0.0038** | 0.9245 | 0.838 |
| w=4 | 0.9353 | 0.9363 | +0.0011 | 0.9349 | 0.837 |
| w=16 | 0.9663 | 0.9634 | −0.0029 | 0.9661 | 0.832 |
| k=4 | 0.9618 | 0.9595 | −0.0022 | 0.9617 | 0.830 |
| k=16 | 0.9618 | 0.9635 | +0.0017 | 0.9620 | 0.885 |
| μ=0.1 | 0.9618 | 0.9586 | −0.0031 | 0.9619 | 0.797 |
| μ=10 | 0.9618 | 0.9615 | −0.0002 | 0.9614 | 0.871 |

## Bitcoin-Alpha (val AUC, 3,448 pairs, every snapshot an anchor)

Windows hold about 11 edges per snapshot and 3 to 4% of nodes are active. w=2, w=4 and k=16 cannot run: the supra graph has fewer than 3k rows and
LOBPCG raises.

| arm | baseline | + Φ_p | change | + shuffled Φ_p | Φ_p alone |
|---|---|---|---|---|---|
| w=8, k=8, μ=1 | 0.8615 | 0.8617 | +0.0002 | 0.8643 | 0.608 |
| w=16 | 0.9167 | 0.9173 | +0.0006 | 0.9169 | 0.678 |
| k=4 | 0.8615 | 0.8617 | +0.0002 | 0.8645 | 0.608 |
| μ=0.1 | 0.8615 | 0.8617 | +0.0002 | 0.8650 | 0.608 |
| μ=10 | 0.8615 | 0.8616 | +0.0002 | 0.8601 | 0.608 |

## Reading

- Φ_p alone is far above chance on new links (0.80 to 0.89 on AS-733), but it adds at most +0.004 to the window baseline and is negative in 4 of 8 arms. The
  standard error of an AUC on 48.7k pairs is about 0.002, from one seed, so only `w=2` (+0.0038) is borderline.
- It helps only where the baseline has little history (`w=2`), and the baseline with a longer window (0.9663 at `w=16`) beats every Φ_p arm.
- On Bitcoin-Alpha the gain is under 0.001 and the shuffled control scores as high; history length, not spectral structure, drives the baseline (0.862 to 0.917).
- The identical Φ_p-alone AUC across arms on Bitcoin-Alpha (0.608) points to one degree-like column.
- Not done: the test split, several seeds, and a node-level model on top of Φ_p.
