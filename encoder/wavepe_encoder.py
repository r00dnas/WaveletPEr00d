import torch
import torch.nn as nn
import torch.nn.functional as F

from compat import torch_sparse


def full_edge_index(edge_index, batch=None):
    """
    Retunr the Full batched sparse adjacency matrices given by edge indices.
    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.
    Implementation inspired by `torch_geometric.utils.to_dense_adj`
    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.
    Returns:
        Complementary edge index.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch,
                        dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short,
                         device=edge_index.device)

        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        # _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_full = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_full

class WavePE_Encoder(nn.Module):
    def __init__(self, config, fill_value=0.0):
        super().__init__()
        self.config = config
        self.encoder = load_positional_encoder(config.ckpt_pos_encoder_path)
    
        if config.learnable:
            #self.mlp = nn.Sequential(nn.Linear(20, 128), nn.ReLU(),nn.Linear(128, config.atom_dim))
            if config.concat:
                self.pe_embedding = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 20))
                self.mlp = nn.Sequential(nn.Linear(config.atom_dim + 20, config.atom_dim), nn.ReLU(), nn.Linear(config.atom_dim, config.atom_dim), nn.Dropout(config.dropout))
            else:
                self.mlp = nn.Linear(20, config.atom_dim, bias=True)
        if config.freeze:
            self.freeze()

    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, data):
        if self.config.freeze:
            with torch.no_grad():
                node_pe = self.encoder.encode(data)
        else:
            node_pe = self.encoder.encode(data)
        if self.config.concat:
            node_pe = self.pe_embedding(node_pe)
            data.x = torch.cat([data.x, node_pe], dim = -1) 
            data.x = self.mlp(data.x)
        else:
            node_pe = self.mlp(node_pe)
            data.x = data.x + node_pe
        pair_pe = data.edge_attr
        
        if self.config.use_full_graph:
            out_idx = data.edge_index
            out_val = pair_pe
            edge_index_full = full_edge_index(out_idx, batch=data.batch)
            edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
            # zero padding to fully-connected graphs
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = torch_sparse.coalesce(
                out_idx, out_val, data.num_nodes, data.num_nodes,
                op="add"
            )

            data.edge_index = out_idx  
            data.edge_attr = out_val
        else:
            data.edge_attr = data.edge_attr
        data.pos = node_pe
        return data