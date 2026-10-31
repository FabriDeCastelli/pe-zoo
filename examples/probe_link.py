"""Does Φ_p help predict the next snapshot's edges? A linear probe on pair features, against window-topology baselines.

    python examples/probe_link.py --dataset bitcoin-alpha --window 8 --k 8 --mu 1.0
    python examples/probe_link.py --dataset as-733 --window 8 --k 8 --mu 1.0 --split test      # only for the chosen configuration

Task, as in the LAMP release (FabrizioDeCastelli/temporal-link-level-prediction): from snapshot t, score its candidate
pairs for the edges of snapshot t + 1 (the positives) against as many sampled negatives, read from the release's files.
The topology comes from the tgdata build of `tgdata.converters.temporal_link`. Everything a feature may see is the window
of the w snapshots ending at t.

Pair features: the baselines (the pair is an edge of snapshot t, the share of the window's snapshots holding it, log
degree sum, log common neighbours, all on the window) and the product of the two nodes' eigenvector rows of Φ_p, which
does not change when an eigenvector's sign flips. Screening protocol, fixed before the test split is touched: the probe is
fit on the train steps and scored by AUC on the val steps; `--split test` is for the one chosen configuration. The AUC is
reported over all candidate pairs and over the pairs that are not edges of snapshot t, which are the new links.
"""
import argparse
import sys
import types
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from tgdata.io import load_dir

import pe_zoo
from probe_mint import auc, fit_logistic, standardise

PARTS = ("train", "val", "test")
FEATURES = ("persistence", "baseline", "baseline+phi_p", "phi_p", "baseline+shuffled phi_p")


def read_candidates(raw):
    """Per part, the candidate pairs and labels of each snapshot, from the release's files."""
    class DynamicData:                       # stand-in for LAMP's src.datasets.dynamic_data.DynamicData: only its state is read
        pass

    for module in ("src", "src.datasets", "src.datasets.dynamic_data"):
        sys.modules.setdefault(module, types.ModuleType(module))
    sys.modules["src.datasets.dynamic_data"].DynamicData = DynamicData
    out = {}
    for part in PARTS:
        storage = torch.load(Path(raw) / f"{part}.pt", weights_only=False).__dict__["_storage"]
        out[part] = [(pairs.numpy(), label.numpy().ravel()) for pairs, label in zip(storage["link_pred_ids"], storage["y"])]
    return out


def symmetric(edges, n):
    a = sp.coo_matrix((np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])), shape=(n, n)).tocsr()
    return ((a + a.T) > 0).astype(np.float32).tocsr()


def anchor_features(g, enc, t, window, k, pairs, label, rng, cap):
    """Rows of pair features at anchor t: baseline [P, 4] and the Φ_p Hadamard product [P, k], with the labels."""
    n, ptr, edges = g.num_nodes, np.asarray(g.edge_ptr), np.asarray(g.edge_index)
    if len(label) > cap:
        keep = rng.choice(len(label), cap, replace=False)
        pairs, label = pairs[:, keep], label[keep]
    u, v = pairs
    current = symmetric(edges[:, ptr[t]:ptr[t + 1]], n)
    steps = [symmetric(edges[:, ptr[s]:ptr[s + 1]], n) for s in range(t - window + 1, t + 1)]
    count, union = sum(steps), (sum(steps) > 0).astype(np.float32).tocsr()
    degree = np.asarray(union.sum(1)).ravel()
    pair = lambda m: np.asarray(m[u, v]).ravel()
    persistence = pair(current)
    baseline = np.stack([persistence, pair(count) / window, np.log1p(degree[u]) + np.log1p(degree[v]),
                         np.log1p(np.asarray(union[u].multiply(union[v]).sum(1)).ravel())], 1)
    eigenvectors = np.asarray(enc.array[t - window + 1][:, :k], dtype=np.float32)
    return baseline, eigenvectors[u] * eigenvectors[v], label


def samples(g, enc, candidates, part, offset, window, k, stride, cap, rng):
    base, had, y = [], [], []
    for i in range(0, len(candidates[part]), stride):
        t = offset + i
        if t < window - 1:
            continue
        b, h, label = anchor_features(g, enc, t, window, k, *candidates[part][i], rng, cap)
        base.append(b), had.append(h), y.append(label)
    return np.concatenate(base), np.concatenate(had), np.concatenate(y).astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["as-733", "bitcoin-alpha"])
    parser.add_argument("--build", default="/raid/f.decastelli/tgdata_build", help="directory of the tgdata builds")
    parser.add_argument("--raw", default="/raid/f.decastelli/LAMP/temporal-link-level-prediction", help="the release's folders")
    parser.add_argument("--cache-dir", default=None, help="the pe-zoo cache (default: $PE_ZOO_CACHE)")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--anchor-stride", type=int, default=4, help="every n-th snapshot of a split is an anchor")
    parser.add_argument("--pairs", type=int, default=4000, help="candidate pairs kept per anchor")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="the split scored; the probe is always fit on train")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    import os
    cache = args.cache_dir or os.environ["PE_ZOO_CACHE"]
    g = load_dir(Path(args.build) / args.dataset)
    enc = pe_zoo.encode_windows(g, cache, window=args.window, k=args.k, mu=args.mu, tol=args.tol)
    candidates = read_candidates(Path(args.raw) / args.dataset)
    sizes = {part: len(candidates[part]) for part in PARTS}
    offsets = {"train": 0, "val": sizes["train"], "test": sizes["train"] + sizes["val"]}
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    data = {part: samples(g, enc, candidates, part, offsets[part], args.window, args.k, args.anchor_stride, args.pairs, rng)
            for part in ("train", args.split)}
    print(f"{args.dataset} w={args.window} k={args.k} mu={args.mu}: pairs train {len(data['train'][2])}, {args.split} {len(data[args.split][2])}, "
          f"positive rate {data[args.split][2].mean():.2f}, {args.split} pairs not in the current snapshot "
          f"{int((data[args.split][0][:, 0] == 0).sum())}")
    train, other = data["train"], data[args.split]
    new = other[0][:, 0] == 0
    blocks = {"persistence": (slice(0, 1), None), "baseline": (slice(0, 4), None)}
    results = {}
    for kind in FEATURES:
        parts = {"persistence": ["persistence"], "baseline": ["baseline"], "baseline+phi_p": ["baseline", "phi_p"],
                 "phi_p": ["phi_p"], "baseline+shuffled phi_p": ["baseline", "shuffled"]}[kind]
        x_train, x_other = [], []
        for part in parts:
            if part in ("persistence", "baseline"):
                a, b = train[0][:, blocks[part][0]], other[0][:, blocks[part][0]]
            else:
                a, b = train[1], other[1]
                if part == "shuffled":
                    a, b = a[rng.permutation(len(a))], b[rng.permutation(len(b))]
            a, b = standardise(a, b)
            x_train.append(a), x_other.append(b)
        x_train, x_other = np.concatenate(x_train, 1), np.concatenate(x_other, 1)
        w, b = fit_logistic(x_train, train[2])
        score = x_other @ w + b
        results[kind] = (auc(score, other[2]), auc(score[new], other[2][new]))
    print(f"{'features':26s} {args.split} AUC, all pairs | on pairs not in the current snapshot")
    for kind in FEATURES:
        print(f"{kind:26s} {results[kind][0]:.4f}            | {results[kind][1]:.4f}")
    print(f"baseline+phi_p minus baseline: all {results['baseline+phi_p'][0] - results['baseline'][0]:+.4f}, new links {results['baseline+phi_p'][1] - results['baseline'][1]:+.4f}")


if __name__ == "__main__":
    main()
