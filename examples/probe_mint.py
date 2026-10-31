"""Does Φ_p add to the edge count on MiNT's Edge Growth/Shrink task? A linear probe on sign-free summaries of the cached encodings.

    python examples/probe_mint.py --arm w8-k8-mu1.0-tol0.0001                    # the default configuration, cached by MOSAIC

Screening protocol, fixed before any test network is touched: the probe is fit on the train splits of the chosen
training networks (features standardised per network with its own train snapshots) and scored by AUC on their val
splits. With --lag L every feature block is extended by its change against the mean of the L previous snapshots (a
causal detrending: the label is a growth trend, and the summaries drift over a network's history); all feature sets
use the same samples, those with a full lag. Feature sets: the edge-count rule (-edges, no fit), the topology features alone (log edges, log active nodes),
those plus the summaries of Φ_p (eigenvalues, active count, participation ratio of each eigenvector), Φ_p alone, and a
control that shuffles the Φ_p summaries over time within each network. Eigenvector signs and bases are arbitrary, so
only invariant summaries are used. Needs tgdata >= 0.4 (the MiNT networks) and the `plot`-free core: torch, numpy, scipy.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import tgdata
from scipy.stats import rankdata

from pe_zoo.encodings import PositionalEncodings

FEATURES = ("count", "topology", "topology+phi_p", "phi_p", "topology+shuffled phi_p")


def auc(score, label):
    positive = label == 1
    if positive.all() or not positive.any():
        return np.nan
    rank = rankdata(score)
    return (rank[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * (~positive).sum())


def phi_p_summaries(array, rows, k):
    """[len(rows), 2k]: eigenvalues 2..k, log active count, log participation ratio of each eigenvector (all sign- and basis-free)."""
    out = []
    for row in rows:
        x = np.asarray(array[row], dtype=np.float32)
        u, lam = x[:, :k], x[0, k:]
        active = (np.abs(u).sum(1) > 0).sum()
        participation = 1.0 / np.maximum((u ** 4).sum(0), 1e-12)
        out.append(np.concatenate([lam[1:], [np.log(max(active, 1))], np.log(participation)]))
    return np.array(out)


def topology_features(g, anchors):
    ptr, edges = np.asarray(g.edge_ptr), np.asarray(g.edge_index)
    return np.array([[np.log(ptr[t + 1] - ptr[t]), np.log(len(np.unique(edges[:, ptr[t]:ptr[t + 1]])))] for t in anchors])


def network_data(name, root, arm, window, k, lag):
    """Per snapshot t: topology features, Φ_p summaries (t >= window - 1), and the train/val anchors with a full lag."""
    g = tgdata.load(name)
    pe = PositionalEncodings(os.path.realpath(Path(root) / f"{name}__pe__{arm}"))
    steps = np.arange(g.num_steps)
    first = window - 1 + lag                                         # the first anchor whose lag window is complete
    blocks = {"topology": topology_features(g, steps),
              "phi_p": np.concatenate([np.full((window - 1, 2 * k), np.nan), phi_p_summaries(pe.array, np.arange(len(pe.array)), k)])}
    out = {"blocks": blocks}
    for split in ("train", "val"):
        anchors = np.asarray(g.task("graph_classification", target="edge_gs", window=1, split=split).anchors)
        anchors = anchors[anchors >= first]
        out[split] = {"anchors": anchors, "y": np.asarray(g.y["edge_gs"].values)[anchors].astype(int),
                      "minus_edges": -np.exp(blocks["topology"][anchors, 0])}
    return out


def block(data, name, split, lag):
    """Level of a feature block at the split's anchors, and with `lag`, its change against the mean of the previous `lag` snapshots."""
    x, anchors = data["blocks"][name], data[split]["anchors"]
    level = x[anchors]
    if not lag:
        return level
    previous = np.stack([x[anchors - j] for j in range(1, lag + 1)]).mean(0)
    return np.concatenate([level, level - previous], 1)


def standardise(train, other):
    mean, std = train.mean(0), train.std(0)
    std = np.where(std > 0, std, 1.0)
    return (train - mean) / std, (other - mean) / std


def fit_logistic(x, y, l2=1e-2):
    x, y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    w, b = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True), torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimiser.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, y) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    optimiser.step(closure)
    return w.detach().numpy(), b.detach().numpy()


def design(data, kind, rng, lag):
    """Feature matrices of the train and val splits of one network (standardised with its train split)."""
    parts = {"topology": ("topology",), "topology+phi_p": ("topology", "phi_p"), "phi_p": ("phi_p",), "topology+shuffled phi_p": ("topology", "phi_p")}[kind]
    train, val = [], []
    for part in parts:
        a, b = block(data, part, "train", lag), block(data, part, "val", lag)
        if kind.endswith("shuffled phi_p") and part == "phi_p":
            a, b = a[rng.permutation(len(a))], b[rng.permutation(len(b))]
        a, b = standardise(a, b)
        train.append(a), val.append(b)
    return np.concatenate(train, 1), np.concatenate(val, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--networks", default=str(Path(__file__).with_name("probe_mint_networks.json")),
                        help="JSON list of mint-* names: 10 train networks with at least 280 snapshots, picked at evenly spaced quantiles of their median edges per snapshot")
    parser.add_argument("--root", default="/raid/f.decastelli/mosaic-cache/by-name")
    parser.add_argument("--arm", default="w8-k8-mu1.0-tol0.0001", help="the Φ_p cache suffix: w<w>-k<k>-mu<mu>-tol<tol>")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--lag", type=int, default=0, help="0: levels only; L: also the change against the previous L snapshots")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="JSON file for the per-network AUCs")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    names = json.load(open(args.networks))
    data = {n: network_data(n, args.root, args.arm, args.window, args.k, args.lag) for n in names}
    print(f"arm {args.arm}, lag {args.lag}: {len(names)} networks, train/val samples {sum(len(d['train']['y']) for d in data.values())}/{sum(len(d['val']['y']) for d in data.values())}")
    scores = {kind: {} for kind in FEATURES}
    for n, d in data.items():
        scores["count"][n] = auc(d["val"]["minus_edges"], d["val"]["y"])
    for kind in FEATURES[1:]:
        designs = {n: design(d, kind, rng, args.lag) for n, d in data.items()}
        w, b = fit_logistic(np.concatenate([designs[n][0] for n in names]), np.concatenate([data[n]["train"]["y"] for n in names]))
        for n in names:
            scores[kind][n] = auc(designs[n][1] @ w + b, data[n]["val"]["y"])
    print(f"{'features':28s} mean val AUC over networks | per network: " + " ".join(f"{n[5:9]:>6s}" for n in names))
    for kind in FEATURES:
        v = np.array([scores[kind][n] for n in names])
        print(f"{kind:28s} {np.nanmean(v):.4f}                    | " + " ".join(f"{x:6.3f}" for x in v))
    base, full = np.array([scores["topology"][n] for n in names]), np.array([scores["topology+phi_p"][n] for n in names])
    diff = full - base
    print(f"topology+phi_p minus topology: mean {np.nanmean(diff):+.4f}, networks higher {int((diff > 0).sum())}/{len(names)}, "
          f"paired t {np.nanmean(diff) / (np.nanstd(diff, ddof=1) / np.sqrt(len(names)) + 1e-12):+.2f}")
    if args.out:
        json.dump(scores, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
