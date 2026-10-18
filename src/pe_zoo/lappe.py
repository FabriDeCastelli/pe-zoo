"""Laplacian positional encodings (Dwivedi & Bresson, "A Generalization of Transformer Networks to Graphs", 2020).

`laplacian_positional_encoding` is copied unchanged from
https://github.com/graphdeeplearning/graphtransformer/blob/654068c3e0924dea7041b9c3d84df52bba72ed4d/data/molecules.py
(MIT licence, Copyright (c) 2020 Vijay Prakash Dwivedi, Xavier Bresson). Around it there is only what it needs to run
here: a stand-in for the DGL graph it takes, and `lappe`, which hands it the symmetric binary adjacency of a snapshot.
The upstream training loop flips the sign of each eigenvector at random per batch; that is not part of the function and
is not applied here.
"""
from types import SimpleNamespace

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import torch

from .supra import symmetric_edges

# The one DGL function the upstream code calls: a tensor of the graph's backend to numpy.
dgl = SimpleNamespace(backend=SimpleNamespace(asnumpy=lambda x: x.detach().cpu().numpy()))


class Graph:
    """The four members of a DGL graph that `laplacian_positional_encoding` uses, over an edge list [2, E]."""

    def __init__(self, edge_index: torch.Tensor, num_nodes: int):
        self.edge_index, self.num_nodes, self.ndata = edge_index, num_nodes, {}

    def number_of_nodes(self) -> int:
        return self.num_nodes

    def in_degrees(self) -> torch.Tensor:
        return torch.bincount(self.edge_index[1], minlength=self.num_nodes)

    def adjacency_matrix_scipy(self, return_edge_ids=False) -> sp.coo_matrix:
        """A[src, dst] = 1 per edge (the encoding is only used on symmetric graphs, where the orientation does not matter)."""
        src, dst = self.edge_index.cpu().numpy()
        return sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(self.num_nodes, self.num_nodes))


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    
    return g


def lappe(edge_index: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """[n, k] float32: eigenvectors 1..k (the trivial one skipped) of the normalised Laplacian of one snapshot.

    The graph is the symmetric binary adjacency of `edge_index` [2, E], as in supra.layer_adjacency without the global
    node. Raises if the graph has no more than k nodes (upstream would return fewer than k columns).
    """
    if n <= k:
        raise ValueError(f"a graph of {n} nodes has {n - 1} non-trivial eigenvectors, fewer than k={k}")
    graph = laplacian_positional_encoding(Graph(symmetric_edges(edge_index), n), k)
    return graph.ndata["lap_pos_enc"]


def lappe_active(edge_index: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """[n, k] float32: `lappe` at the active nodes (those with an edge), from the lowest k + 1 eigenpairs of L on the active nodes only.

    An isolated node is a block `[1]` of L, so the whole-graph run is the active nodes plus `n - m` such blocks. Only
    `min(k, n - m)` of them can reach the k kept columns, so that is all that is added: the active rows are those of
    `lappe`, up to the eigenvectors' arbitrary sign and basis. L is the one of `laplacian_positional_encoding`; being
    symmetric, it goes to `scipy.linalg.eigh` for its lowest k + 1 eigenpairs, 15 to 30 times faster than the general
    `np.linalg.eig` on a few thousand nodes, at a cost cubic in `m + k` instead of `n`. The inactive rows are zero here; the
    whole-graph run puts unit vectors on k of them, which nothing reads.
    """
    if n <= k:
        raise ValueError(f"a graph of {n} nodes has {n - 1} non-trivial eigenvectors, fewer than k={k}")
    edges = symmetric_edges(edge_index)
    active = torch.unique(edges)
    pe = torch.zeros(n, k)
    if len(active):
        size = len(active) + min(k, n - len(active))
        graph = Graph(torch.searchsorted(active, edges.contiguous()), size)
        scale = sp.diags(graph.in_degrees().numpy().clip(1) ** -0.5)
        laplacian = (sp.eye(size) - scale @ graph.adjacency_matrix_scipy().astype(float) @ scale).toarray()
        vectors = scipy.linalg.eigh(laplacian, subset_by_index=[0, k])[1][:, 1:]  # eigenvectors 1..k, by increasing eigenvalue
        pe[active] = torch.from_numpy(vectors[: len(active)]).float()
    return pe
