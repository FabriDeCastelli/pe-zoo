# Development

```bash
source .envrc            # venv in /raid/f.decastelli/venvs/pe-zoo, PE_ZOO_CACHE, TGDATA_ROOT
uv sync --extra tgdata --extra plot
pytest
```

`TGDATA_ROOT` points at the datasets already on this machine; pe-zoo never downloads.

Tests compare each function with a dense numpy transcription of the LaTeX on small graphs, check LOBPCG against `eigh`
on a larger window, and exercise the cache (reuse, atomic write, key sensitivity, `at` / `rows`).

## GPUs

`encode_windows(..., device=device)` runs the sparse LOBPCG on that device. The GPU is shared: pick a free one with
`~/.claude/free-gpu` (it prints free indices, `-1` if none) and never use one that another user's process occupies.
The default is the CPU: about 0.3 s per window at PeMS08 scale (170 nodes, 12 snapshots).
