"""How do the supra-Laplacian encodings of a time-varying dataset move? Figures saved under outputs/plots/<dataset>/.

    python examples/plot_encodings.py --dataset twittertennis-rg17 --window 4
    python examples/plot_encodings.py --path /path/to/a/local/tgdata/build --window 4 --nodes 3 17

Encodes every window (through the cache, so a rerun is instant) and saves the eigenvalues over time, the first
eigenvectors of a few nodes over time, one window as a heatmap, the similarity of the windows, and the similarity by lag
next to the edge-set Jaccard similarity of the topology. Without --nodes it takes the most active nodes. Needs the
`tgdata` and `plot` extras. A fixed-topology dataset has one encoding for all windows and is not plotted over time.
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import tgdata
from tgdata.io import load_dir

import pe_zoo
from pe_zoo import plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="twittertennis-rg17")
    parser.add_argument("--path", help="a local tgdata build directory, instead of loading --dataset by name")
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--nodes", type=int, nargs="+", help="node ids (default: the three most active)")
    parser.add_argument("--max-lag", type=int, default=30)
    parser.add_argument("--cache-dir", default=os.environ.get("PE_ZOO_CACHE"))
    parser.add_argument("--out-dir", default="outputs/plots")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not args.cache_dir:
        raise SystemExit("set --cache-dir or PE_ZOO_CACHE")

    g = load_dir(args.path) if args.path else tgdata.load(args.dataset)
    enc = pe_zoo.encode_windows(g, args.cache_dir, window=args.window, k=args.k, device=args.device)
    if enc.is_static:
        raise SystemExit(f"{g.name} has a fixed topology: one encoding for every window, nothing to plot over time")
    pe, times = enc.array[:].astype(np.float32), np.arange(args.window - 1, g.num_steps)
    nodes = args.nodes or np.argsort(-np.bincount(g.edge_index.ravel(), minlength=g.num_nodes))[:3].tolist()
    lags = np.arange(1, min(args.max_lag, len(pe) - 1) + 1)
    out = Path(args.out_dir) / g.name
    out.mkdir(parents=True, exist_ok=True)
    tag = f"w{args.window}"
    figures = {
        "eigenvalues": plots.plot_eigenvalues(times, pe, title=f"{g.name}, window {args.window}"),
        "components": plots.plot_components(times, pe, nodes, title=f"{g.name}, window {args.window}"),
        "window": plots.plot_window(pe[len(pe) // 2], title=f"{g.name}: the window ending at step {times[len(pe) // 2]}"),
        "matrix": plots.plot_similarity_matrix(times, pe, start=times[0], end=times[-1] + 1, title=f"{g.name}: window similarity"),
        "lags": plots.plot_lag_curves(lags, {"encoding": plots.lag_similarity(pe, lags), "topology": plots.edge_jaccard(g, lags)},
                                      title=f"{g.name}: similarity against lag"),
    }
    for name, fig in figures.items():
        fig.savefig(out / f"{tag}_{name}.png", dpi=130)
    print(f"saved {len(figures)} figures to {out}")


if __name__ == "__main__":
    main()
