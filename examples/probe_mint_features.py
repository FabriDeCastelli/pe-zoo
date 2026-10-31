"""The sign-free Φ_p summaries of the windows ending at a fixed set of anchors of one MiNT network, for one configuration.

    python examples/probe_mint_features.py --network mint-aioz --window 8 --k 8 --mu 1.0 --out OUT [--shard 3/8]

The anchors are the snapshots t with t % STRIDE == STRIDE - 1 and t >= FIRST, the same for every configuration whatever
its window (consecutive MiNT snapshots overlap by 6/7, so windows a few days apart are near-duplicates). Writes
OUT/w<w>-k<k>-mu<mu>/<network>.shard<i>of<n>.npz with `anchors` and `x` [anchors, 2k] float32: the summaries of
`probe_mint.phi_p_summaries` of the window of steps t - w + 1 .. t, from `supra.supra_pe` directly (float64, no float16
cache), seeded per window. A window whose LOBPCG does not converge at `--tol` gets NaN and is counted; the probe drops
those samples for every feature set. Run on the CPU with OMP_NUM_THREADS capped. Needs tgdata >= 0.4.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import tgdata
import torch

from pe_zoo import supra
from probe_mint import phi_p_summaries

STRIDE, FIRST = 4, 15                                               # FIRST: the widest window of the screen (16) has its first anchor at 15


def anchored_steps(num_steps):
    """The snapshots t of the screen: t % STRIDE == STRIDE - 1 and t >= FIRST, the same for every window length."""
    return np.arange(FIRST, num_steps, STRIDE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--shard", default="0/1", help="i/n: every n-th anchor from the i-th")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))
    index, count = map(int, args.shard.split("/"))
    path = Path(args.out) / f"w{args.window}-k{args.k}-mu{args.mu}" / f"{args.network}.shard{index}of{count}.npz"
    if path.exists():
        print(f"{path} exists")
        return
    g = tgdata.load(args.network)
    anchors = anchored_steps(g.num_steps)[index::count]
    out = np.full((len(anchors), 2 * args.k), np.nan, dtype=np.float32)
    failed, start = 0, time.time()
    for i, t in enumerate(anchors):
        edges = [torch.tensor(g.edges_at(step)[0], dtype=torch.long) for step in range(t - args.window + 1, t + 1)]
        try:
            pe = supra.supra_pe(edges, g.num_nodes, args.k, args.mu, "sym", args.tol, seed=supra.SEED + int(t)).numpy()
        except RuntimeError:                                        # LOBPCG did not converge
            failed += 1
            continue
        out[i] = phi_p_summaries(pe[None], [0], args.k)[0]
    print(f"{args.network} w{args.window} k{args.k} mu{args.mu} shard {args.shard}: {len(anchors)} windows in {time.time() - start:.0f}s, "
          f"{failed} not converged at tol={args.tol}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_name(path.name + ".tmp.npz"), anchors=anchors, x=out, failed=failed)
    path.with_name(path.name + ".tmp.npz").rename(path)                # a result is complete or absent


if __name__ == "__main__":
    main()
