import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool, GINEConv, global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class DeepSet(nn.Module):
    def __init__(self, config, dim = None):
        super().__init__()
        if hasattr(config, "atom_dim"):
            self.node_emb = AtomEncoder(config.atom_dim)
            self.edge_emb = BondEncoder(config.bond_dim)
            self.config = config
        else:
            self.node_emb = nn.Embedding(21, dim)
            self.edge_emb = nn.Embedding(4, dim)