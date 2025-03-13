from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import spmm

class GINE_LSPE_Conv(MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper.

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn_h: torch.nn.Module, nn_p: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn_h = nn_h
        self.nn_p = nn_p

        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        if edge_dim is not None:
            self.lin_e_h = torch.nn.Sequential(Linear(edge_dim + (edge_dim + 20) * 2, edge_dim), torch.nn.ReLU(), Linear(edge_dim, edge_dim + 20))
            self.lin_e_p = torch.nn.Sequential(Linear(edge_dim + 40, edge_dim), torch.nn.ReLU(), Linear(edge_dim, 20))

        else:
            self.lin = None
        self.reset_parameters()
        self.e_for = {0 : 'h', 1: 'p'}
        self.flag = 0

        self.graph_norm = GraphNorm(edge_dim)
        self.batch_norm = torch.nn.BatchNorm1d(edge_dim)

    def reset_parameters(self):
        reset(self.nn_h)
        reset(self.nn_p)
        self.eps.data.fill_(self.initial_eps)
        reset(self.lin_e_h)
        reset(self.lin_e_p)

    def forward(
        self,
        h: Union[Tensor, OptPairTensor],
        p: Union[Tensor, OptPairTensor], 
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None, 
        batch_index = None
    ) -> Tensor:


        # edge_attr_h = torch.cat([h[edge_index[0]], h[edge_index[1]], edge_attr], dim = -1)
        # edge_attr_h = self.lin_e_h(edge_attr_h)

        # edge_attr_p = torch.cat([p[edge_index[0]], p[edge_index[1]], edge_attr], dim = -1)
        # edge_attr_p = self.lin_e_p(edge_attr_p)

        h = torch.cat([h, p], dim = -1)

        if isinstance(h, Tensor):
            h = (h, h)
        if isinstance(p, Tensor):
            p = (p, p)


        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        self.e_for = 'h'
        h_out = self.propagate(edge_index, x=h, edge_attr=edge_attr, size=size)

        self.e_for = 'p'
        p_out = self.propagate(edge_index, x=p, edge_attr=edge_attr, size=size)

        h_r = h[1]
        if h_r is not None:
            h_out = h_out + (1 + self.eps) * h_r
        
        p_r = p[1]
        if p_r is not None:
            p_out = p_out + (1 + self.eps) * p_r


        
        h_out = self.nn_h(h_out)
        p_out = self.nn_p(p_out)

        #h_out = self.graph_norm(h_out, batch_index)

        return h_out, p_out
        # return self.nn_h(h_out), self.nn_p(p_out), edge_attr

    def message(self, x_i, x_j: Tensor, edge_attr: Tensor) -> Tensor:            
        edge_attr = torch.cat([x_i, x_j, edge_attr], dim = -1)
        if self.e_for == 'h':
            edge_attr = self.lin_e_h(edge_attr)
        elif self.e_for == "p":
            edge_attr = self.lin_e_p(edge_attr) 
        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'