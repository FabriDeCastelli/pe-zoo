"""Supra-Laplacian positional encoding of one window of snapshots (one function per equation).

Deviations from the LaTeX, all deliberate (DESIGN.md): a global node per layer joined to the active
nodes of that layer (arXiv 2506.01596), inactive (node, layer) copies removed (a node inactive in the last
layer takes its latest active row of the window), binary symmetric adjacency, and each eigenvector column
L2-normalised over the returned rows.
"""
import torch

MAXITER = 1000
SEED = 0
NORMS = ("sym", "rw", "combinatorial")


def layer_adjacency(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """A_tau: binary symmetric adjacency of one snapshot plus a global node with index n, [n+1, n+1] sparse.

    The global node is joined to every node that has an edge in the snapshot (its active nodes).
    """
    edges = torch.unique(torch.cat([edge_index, edge_index.flip(0)], dim=1), dim=1)
    active = torch.unique(edges)
    glob = torch.full_like(active, n)
    edges = torch.cat([edges, torch.stack([active, glob]), torch.stack([glob, active])], dim=1)
    values = torch.ones(edges.shape[1], dtype=torch.float64, device=edges.device)
    return torch.sparse_coo_tensor(edges, values, (n + 1, n + 1)).coalesce()


def active_nodes(layer: torch.Tensor) -> torch.Tensor:
    """[n+1] bool: the nodes joined to the global node of `layer`, and the global node itself."""
    rows, cols = layer.indices()
    keep = torch.zeros(layer.shape[0], dtype=torch.bool, device=rows.device)
    keep[rows[cols == layer.shape[0] - 1]] = True
    keep[-1] = True
    return keep


def supra_adjacency(layers: list[torch.Tensor], mu: float) -> torch.Tensor:
    """A_sup, eq. `supra_laplacian_def`: block tridiagonal, A_tau on the diagonal, mu*I between adjacent layers."""
    w, m = len(layers), layers[0].shape[0]
    indices = [layer.indices() + i * m for i, layer in enumerate(layers)]
    values = [layer.values() for layer in layers]
    below = torch.arange((w - 1) * m, device=indices[0].device)
    coupling = torch.stack([below, below + m])
    indices += [coupling, coupling.flip(0)]
    values += [torch.full((coupling.shape[1],), mu, dtype=torch.float64, device=below.device)] * 2
    return torch.sparse_coo_tensor(torch.cat(indices, dim=1), torch.cat(values), (w * m, w * m)).coalesce()


def restrict(adjacency: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """The rows and columns of `adjacency` where `keep` is True."""
    new_index = torch.cumsum(keep, 0) - 1
    rows, cols = adjacency.indices()
    inside = keep[rows] & keep[cols]
    size = int(keep.sum())
    return torch.sparse_coo_tensor(torch.stack([new_index[rows[inside]], new_index[cols[inside]]]),
                                   adjacency.values()[inside], (size, size)).coalesce()


def supra_laplacian(adjacency: torch.Tensor, norm: str = "sym") -> tuple[torch.Tensor, torch.Tensor | None]:
    """(L, B) such that the wanted eigenpairs solve L u = lambda B u; B is None for a standard problem.

    `sym` is eq. `supra_laplacian_def`, L = I - D^-1/2 A D^-1/2. `rw` is the generalised problem
    (D - A) v = lambda D v of eq. `supra_variational` (V^T D V = I), so B = D. `combinatorial` is D - A.
    """
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
    rows, cols = adjacency.indices()
    values = adjacency.values()
    degree = torch.sparse.sum(adjacency, dim=1).to_dense()
    diagonal = torch.arange(len(degree), device=rows.device).repeat(2, 1)
    if norm == "sym":
        scale = degree.rsqrt()
        laplacian = torch.sparse_coo_tensor(torch.cat([adjacency.indices(), diagonal], dim=1),
                                            torch.cat([-values * scale[rows] * scale[cols], torch.ones_like(degree)]),
                                            adjacency.shape)
        return laplacian.coalesce(), None
    laplacian = torch.sparse_coo_tensor(torch.cat([adjacency.indices(), diagonal], dim=1),
                                        torch.cat([-values, degree]), adjacency.shape).coalesce()
    return laplacian, (torch.sparse_coo_tensor(diagonal, degree, adjacency.shape) if norm == "rw" else None)


def smallest_eigenpairs(laplacian: torch.Tensor, b: torch.Tensor | None, k: int, tol: float, seed: int):
    """The k smallest eigenpairs (lambda [k], U [size, k]) of L u = lambda B u by LOBPCG.

    Raises if LOBPCG fails or, for any pair, the residual ||L u - lambda B u|| / ||B u|| is above `tol`
    times a bound on ||L|| (its largest absolute row sum), which is the scale LOBPCG itself stops at.
    """
    generator = torch.Generator(device=laplacian.device).manual_seed(seed)
    start = torch.randn(laplacian.shape[0], k, generator=generator, dtype=torch.float64, device=laplacian.device)
    values, vectors = torch.lobpcg(laplacian, k=k, B=b, X=start, largest=False, tol=tol, niter=MAXITER)
    lu = laplacian @ vectors
    bu = vectors if b is None else b @ vectors
    residual = (lu - values * bu).norm(dim=0) / bu.norm(dim=0)
    rows, _ = laplacian.indices()
    bound = torch.zeros(laplacian.shape[0], dtype=torch.float64, device=rows.device).index_add_(0, rows, laplacian.values().abs()).max()
    if not (residual <= tol * bound).all():
        raise RuntimeError(f"LOBPCG did not converge: worst residual {residual.max():.2e} > {tol * bound:.2e} (tol={tol:.0e})")
    return values, vectors


def supra_pe(edge_layers: list[torch.Tensor], n: int, k: int, mu: float, norm: str, tol: float,
             seed: int = SEED) -> torch.Tensor:
    """Phi_p of a window, eq. `supra_pe_def`: [U_t || 1 lambda^T], float64 [n, 2k].

    `edge_layers` are the [2, E] edge indices of the w snapshots, oldest first; U_t is the block of the
    last snapshot, without the global node. A node inactive in the last snapshot takes its row from the
    latest snapshot of the window where it is active, and zeros if there is none.
    """
    if len(edge_layers) < 2:
        raise ValueError("the window needs at least 2 snapshots")
    layers = [layer_adjacency(edges, n) for edges in edge_layers]
    keep = torch.stack([active_nodes(layer) for layer in layers])                       # [w, n+1]
    laplacian, b = supra_laplacian(restrict(supra_adjacency(layers, mu), keep.flatten()), norm)
    values, vectors = smallest_eigenpairs(laplacian, b, k, tol, seed)
    u = torch.zeros(n, k, dtype=torch.float64, device=vectors.device)
    for active, block in zip(keep[:, :-1], vectors.split(keep.sum(1).tolist())):
        u[active] = block[:-1]                                                          # the global node is the last row
    u = u / u.norm(dim=0)
    if not torch.isfinite(u).all():
        raise RuntimeError("an eigenvector has no mass on the active nodes of the window, so it cannot be normalised")
    return torch.cat([u, values.expand(n, k)], dim=1)
