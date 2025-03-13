import torch 
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym import GNNGraphHead, global_mean_pool, global_add_pool
from net.layer import ResGatedGraphConv
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from utils.commons import load_positional_encoder
from encoder.wavepe_encoder import WavePE_Encoder
from layer.gps_layer import GPSLayer

pyg_nn.ResGatedGraphConv = ResGatedGraphConv

class GraphTransformer(nn.Module):
    def __init__(self, config, out_dim):
        super().__init__()

        self.config = config
        self.num_layer = config.num_layer
         
        self.atom_encoder = AtomEncoder(config.atom_dim)
        self.bond_encoder = BondEncoder(config.bond_dim)

        #### positional encoder ####
        self.pos_encoder = WavePE_Encoder(config)

        dim = self.config.atom_dim 
        self.layers = nn.ModuleList()
        
        for i in range(self.num_layer):
            if config.local_gnn_type == "gat":
                conv = pyg_nn.GATv2Conv(dim, dim, heads = 4, concat = False, edge_dim =dim) 
            elif config.local_gnn_type == "gine": 
                mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
                conv = pyg_nn.GINEConv(mlp, edge_dim = dim) 
            elif config.local_gnn_type == "transformer_conv":
                conv = pyg_nn.TransformerConv(dim, dim, heads = 4, concat = False, edge_dim = dim)
            
            self.layers.append(pyg_nn.GPSConv(dim, conv, dropout = config.dropout, attn_dropout=config.attn_dropout))
                
             
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
        )
    
    def forward(self, batch):
        batch.x = self.atom_encoder(batch.x)
        batch.edge_attr = self.bond_encoder(batch.edge_attr) 
        batch = self.pos_encoder(batch) 
        for layer in range(self.num_layer): 
            batch.x = self.layers[layer](batch.x, batch.edge_index, batch =batch.batch, edge_attr = batch.edge_attr)
        return self.prediction_head(batch)[0]

    @property 
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)