import typing
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import coalesce


import torch
from torch import Tensor


def cumsum(x: Tensor, dim: int = 0) -> Tensor:
    r"""Returns the cumulative sum of elements of :obj:`x`.
    In contrast to :meth:`torch.cumsum`, prepends the output with zero.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int, optional): The dimension to do the operation over.
            (default: :obj:`0`)

    Example:
        >>> x = torch.tensor([2, 4, 1])
        >>> cumsum(x)
        tensor([0, 2, 6, 7])

    """
    size = x.size()[:dim] + (x.size(dim) + 1, ) + x.size()[dim + 1:]
    out = x.new_empty(size)

    out.narrow(dim, 0, 1).zero_()
    torch.cumsum(x, dim=dim, out=out.narrow(dim, 1, x.size(dim)))

    return out

def dense_to_sparse(
    adj: Tensor,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        adj (torch.Tensor): The dense adjacency matrix of shape
            :obj:`[num_nodes, num_nodes]` or
            :obj:`[batch_size, num_nodes, num_nodes]`.
        mask (torch.Tensor, optional): A boolean tensor of shape
            :obj:`[batch_size, num_nodes]` holding information about which
            nodes are in each example are valid. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:
        >>> # For a single adjacency matrix:
        >>> adj = torch.tensor([[3, 1],
        ...                     [2, 0]])
        >>> dense_to_sparse(adj)
        (tensor([[0, 0, 1],
                [0, 1, 0]]),
        tensor([3, 1, 2]))

        >>> # For two adjacency matrixes:
        >>> adj = torch.tensor([[[3, 1],
        ...                      [2, 0]],
        ...                     [[0, 1],
        ...                      [0, 2]]])
        >>> dense_to_sparse(adj)
        (tensor([[0, 0, 1, 2, 3],
                [0, 1, 0, 3, 3]]),
        tensor([3, 1, 2, 1, 2]))

        >>> # First graph with two nodes, second with three:
        >>> adj = torch.tensor([[
        ...         [3, 1, 0],
        ...         [2, 0, 0],
        ...         [0, 0, 0]
        ...     ], [
        ...         [0, 1, 0],
        ...         [0, 2, 3],
        ...         [0, 5, 0]
        ...     ]])
        >>> mask = torch.tensor([
        ...         [True, True, False],
        ...         [True, True, True]
        ...     ])
        >>> dense_to_sparse(adj, mask)
        (tensor([[0, 0, 1, 2, 3, 3, 4],
                [0, 1, 0, 3, 3, 4, 3]]),
        tensor([3, 1, 2, 1, 2, 3, 5]))
    """
    if adj.dim() < 2 or adj.dim() > 3:
        raise ValueError(f"Dense adjacency matrix 'adj' must be two- or "
                         f"three-dimensional (got {adj.dim()} dimensions)")

    if mask is not None and adj.dim() == 2:
        warnings.warn("Mask should not be provided in case the dense "
                      "adjacency matrix is two-dimensional")
        mask = None

    if mask is not None and mask.dim() != 2:
        raise ValueError(f"Mask must be two-dimensional "
                         f"(got {mask.dim()} dimensions)")

    if mask is not None and adj.size(-2) != adj.size(-1):
        raise ValueError(f"Mask is only supported on quadratic adjacency "
                         f"matrices (got [*, {adj.size(-2)}, {adj.size(-1)}])")

    if adj.dim() == 2:
        edge_index = adj.nonzero().t()
        edge_attr = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_attr
    else:
        flatten_adj = adj.view(-1, adj.size(-1))
        if mask is not None:
            flatten_adj = flatten_adj[mask.view(-1)]
        edge_index = flatten_adj.nonzero().t()
        edge_attr = flatten_adj[edge_index[0], edge_index[1]]

        if mask is None:
            offset = torch.arange(
                start=0,
                end=adj.size(0) * adj.size(2),
                step=adj.size(2),
                device=adj.device,
            )
            offset = offset.repeat_interleave(adj.size(1))
        else:
            count = mask.sum(dim=-1)
            offset = cumsum(count)[:-1]
            offset = offset.repeat_interleave(count)

        edge_index[1] += offset[edge_index[0]]

        return edge_index, edge_attr
