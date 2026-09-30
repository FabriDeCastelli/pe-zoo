<h1 align="center">pe-zoo</h1>

<p align="center">Cached supra-Laplacian positional encodings of a sliding window of snapshots, from the topology alone.</p>

## Install

In a uv project (nothing is cloned into it; `uv.lock` pins the version):

```bash
uv add "pe-zoo[tgdata,plot] @ git+ssh://git@github.com/FabriDeCastelli/pe-zoo.git" --tag v0.2.0
```

With pip: `pip install "pe-zoo[tgdata,plot] @ git+ssh://git@github.com/FabriDeCastelli/pe-zoo.git@v0.2.0"`.
The `tgdata` extra installs [tgdata](https://github.com/FabriDeCastelli/tgdata) v0.3.0, which supplies the datasets.

## What you can do

**Encode every window of a dataset** (computed once, then read from disk)

```python
import tgdata, pe_zoo

g = tgdata.load("pems08")
enc = pe_zoo.encode_windows(g, "cache/", window=12)   # k=8 eigenpairs of the supra-Laplacian of 12 snapshots
enc.array.shape                                       # (170, 16): a fixed topology has one encoding for all windows
```

**Look up a window by its last step**

```python
enc.at(t)                    # [nodes, 2k]: the window of steps t-11, ..., t, read at step t
```

**Get the encodings of a split** (rows follow the task's samples, whatever the task)

```python
task = g.task(split="test")
enc.rows(task).shape         # (samples, 170, 16)
```

**Encode one window of edge lists** (no dataset, no cache)

```python
from pe_zoo.supra import supra_pe

phi = supra_pe([edges_t0, edges_t1, edges_t2], n=100, k=8, mu=1.0, norm="sym", tol=1e-6)   # float64 [100, 16]
```

**Choose the operator and the storage** (each choice gets its own cache directory)

```python
pe_zoo.encode_windows(g, "cache/", window=12, k=16, mu=0.5, norm="rw", tol=1e-8, dtype="float32", device=device)
```

**Plot them** (eigenvalues, eigenvectors and window similarity over time; needs the `plot` extra)

```python
from pe_zoo import plots

plots.plot_eigenvalues(times, enc.array[:])          # times: the last step of each window
```

## Learn more

- [The encoding](docs/encoding.md): the equations, where each lives in the code, and the deliberate deviations.
- [Cache and lookup](docs/cache.md): what determines a cache directory, the `t` convention, fixed and time-varying topologies.
- [Diagnostics](docs/diagnostics.md): the plots, and how to read them.
- [Development](docs/development.md): setup, tests, GPUs.
