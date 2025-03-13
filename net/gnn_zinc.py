import torch 
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from layer.gated_layer import GatedGCNLayer
from torch_geometric.graphgym import GNNGraphHead, global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from encoder.wavepe_encoder import WavePE_Encoder
import torch.nn.functional as F

class GNN_VirtualNode(nn.Module):
    def __init__(self, config, out_dim = 11):
        super().__init__()

        self.config = config
        self.num_layer = config.num_layer
        
        
        if config.dataset == "zinc":
            self.node_encoder = nn.Embedding(28, config.atom_dim)
            self.edge_encoder = nn.Embedding(4, config.bond_dim)
        elif config.dataset in ["MNIST", "CIFAR10"]:
            input_dim = 1 if config.dataset == "MNIST" else 3
            self.node_encoder = nn.Linear(input_dim, config.atom_dim)
            self.edge_encoder = nn.Linear(1, config.bond_dim) 
    

        #### positional encoder ####
        self.pos_encoder = WavePE_Encoder(config)

        
        dim = self.config.atom_dim
      
        self.vn_embedding = nn.Embedding(1, dim)
        torch.nn.init.constant_(self.vn_embedding.weight.data, 0)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.mlp_vn = nn.ModuleList()

        for i in range(self.num_layer):
            if config.local_gnn_type == "gated_gcn":
                self.layers.append(GatedGCNLayer(
                dim, dim, dropout = self.config.dropout, 
                residual = True, equivstable_pe=True 
                ))
            elif config.local_gnn_type == "gat":
                self.layers.append(pyg_nn.GATv2Conv(dim, dim, heads = 4, concat = False, edge_dim = dim))
                self.norms.append(nn.BatchNorm1d(dim))
            elif config.local_gnn_type == "gine":
                #self.layers.append(GINEConvLayer(dim, dim, dropout = config.dropout, residual=True))
                mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
                self.layers.append(pyg_nn.GINEConv(mlp, edge_dim = dim))
                self.norms.append(nn.BatchNorm1d(dim))
            elif config.local_gnn_type == "transformer_conv":
                self.layers.append(pyg_nn.TransformerConv(dim, dim, heads = 4, concat = False, edge_dim = dim))
                self.norms.append(nn.BatchNorm1d(dim))

        for i in range(self.num_layer-1):
            mlp = nn.Sequential(
                nn.Linear(dim, dim), 
                nn.BatchNorm1d(dim), 
                nn.ReLU(),
                nn.Linear(dim, dim), 
                nn.BatchNorm1d(dim), 
                nn.ReLU()
            )
            self.mlp_vn.append(mlp)
         
             
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, out_dim),
        )


    def forward(self, batch):
        if self.config.dataset == "zinc":
            batch.x = self.node_encoder(batch.x.squeeze())
            batch.edge_attr = self.edge_encoder(batch.edge_attr) 
        else:
            batch.x = self.node_encoder(batch.x)
            batch.edge_attr = self.edge_encoder(batch.edge_attr.unsqueeze(-1))
        vn_embedding = self.vn_embedding(torch.zeros(batch.batch[-1].item() + 1).to(batch.edge_index.dtype).to(batch.x.device))  
        
        batch = self.pos_encoder(batch) 
        #batch.x = self.pre_mp(batch.x)
        for layer in range(self.num_layer):
            batch.x = batch.x + vn_embedding[batch.batch]
            if self.config.local_gnn_type == "gated_gcn":
                batch = self.layers[layer](batch)
            else:
                batch.x = self.layers[layer](batch.x, batch.edge_index, batch.edge_attr)
                batch.x = F.leaky_relu(self.norms[layer](batch.x), negative_slope=0.1)
                batch.x = F.dropout(batch.x, p = self.config.dropout, training = self.training)
            if layer < self.num_layer - 1:
                vn_embedding_temp = global_add_pool(batch.x, batch.batch) + vn_embedding
                vn_embedding = F.dropout(self.mlp_vn[layer](vn_embedding_temp), p = self.config.dropout, training = self.training)
        x = global_mean_pool(batch.x, batch.batch)
        return self.mlp(x)
        return self.prediction_head(batch)[0]

    @property 
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class GNN(nn.Module):
    def __init__(self, config, out_dim):
        super().__init__()

        self.config = config
        self.num_layer = config.num_layer
         
        self.atom_encoder = nn.Embedding(21, config.atom_dim)
        self.bond_encoder = nn.Embedding(4, config.bond_dim)

        #### positional encoder ####
        self.pos_encoder = WavePE_Encoder(config)

        dim = self.config.atom_dim 
        
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        #self.mlp_vn = nn.ModuleList()

        for i in range(self.num_layer):
            if config.local_gnn_type == "gated_gcn":
                self.layers.append(GatedGCNLayer(
                dim, dim, dropout = self.config.dropout, 
                residual = True, equivstable_pe=True 
                ))
            elif config.local_gnn_type == "gat":
                self.layers.append(pyg_nn.GATv2Conv(dim, dim, heads = 4, concat = False, edge_dim = dim))
                self.norms.append(nn.BatchNorm1d(dim))
            elif config.local_gnn_type == "gine":
                #self.layers.append(GINEConvLayer(dim, dim, dropout = config.dropout, residual=True))
                mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
                self.layers.append(pyg_nn.GINEConv(mlp, edge_dim = dim))
                self.norms.append(nn.BatchNorm1d(dim))
            elif config.local_gnn_type == "transformer_conv":
                self.layers.append(pyg_nn.TransformerConv(dim, dim, heads = 4, concat = False, edge_dim = dim))
                self.norms.append(nn.BatchNorm1d(dim))
             
        self.prediction_head = GNNGraphHead(dim, out_dim)
        self.prediction_head.pooling_fun = global_mean_pool
        #self.graph_pred_linear = torch.nn.Linear(dim, out_dim)


    def forward(self, batch):
        batch.x = self.atom_encoder(batch.x.squeeze())
        batch.edge_attr = self.bond_encoder(batch.edge_attr) 
        batch = self.pos_encoder(batch) 
        #batch.x = self.pre_mp(batch.x)
        for layer in range(self.num_layer): 
            if self.config.local_gnn_type == "gated_gcn":
                batch = self.layers[layer](batch) 
            else:
                batch.x = self.layers[layer](batch.x, batch.edge_index, batch.edge_attr)
            batch.x = self.norms[layer](batch.x)
            if layer == self.num_layer - 1:
                h = F.dropout(batch.x, self.config.dropout, training=self.training)
            else:
                h = F.dropout(F.leaky_relu(batch.x, negative_slope=0.1), self.config.dropout, training=self.training)
            if self.config.residual:
                batch.x = batch.x + h
            else:
                batch.x = h
        return self.prediction_head(batch)[0]

    @property 
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class GraphTransformer(nn.Module):
    def __init__(self, config, out_dim):
        super().__init__()

        self.config = config
        self.num_layer = config.num_layer
         
        if config.dataset == "zinc":
            self.node_encoder = nn.Embedding(28, config.atom_dim)
            self.edge_encoder = nn.Embedding(4, config.bond_dim)
        elif config.dataset in ["MNIST", "CIFAR10"]:
            input_dim = 3 if config.dataset == "MNIST" else 5
            self.node_encoder = nn.Linear(input_dim, config.atom_dim)
            self.edge_encoder = nn.Linear(1, config.bond_dim) 

        #### positional encoder ####
        self.pos_encoder = WavePE_Encoder(config)

        dim = self.config.atom_dim 
        self.layers = nn.ModuleList()
        
        for i in range(self.num_layer):
            if config.local_gnn_type == "gat":
                conv = pyg_nn.GATv2Conv(dim, dim, heads = 4, concat = False, edge_dim =dim) 
            elif config.local_gnn_type == "sage":
                conv =pyg_nn.SAGEConv(dim, dim)
            elif config.local_gnn_type == "gated_gcn":
                conv = pyg_nn.ResGatedGraphConv(dim, dim, edge_dim = dim)
            elif config.local_gnn_type == "gine": 
                mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
                conv = pyg_nn.GINEConv(mlp, edge_dim = dim) 
            elif config.local_gnn_type == "transformer_conv":
                conv = pyg_nn.TransformerConv(dim, dim, heads = 4, concat = False, edge_dim = dim)
            elif config.local_gnn_type == "gin":
                mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
                conv = pyg_nn.GINConv(mlp)
            
            attn_kwargs = {'dropout': 0.5}
            self.layers.append(pyg_nn.GPSConv(dim, conv, dropout = config.dropout, attn_kwargs=attn_kwargs))
                
             
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, out_dim),
        )
    
    def forward(self, batch):
        if self.config.dataset == "zinc":
            batch.x = self.node_encoder(batch.x.squeeze())
            batch.edge_attr = self.edge_encoder(batch.edge_attr) 
        else:
            batch.x = torch.cat([batch.x, batch.pos] ,dim = -1)
            batch.x = self.node_encoder(batch.x)
            if self.config.local_gnn_type != "sage":
                batch.edge_attr = self.edge_encoder(batch.edge_attr.unsqueeze(-1))
        batch = self.pos_encoder(batch) 
        for layer in range(self.num_layer): 
            if self.config.local_gnn_type == "gin":
                batch.x = self.layers[layer](batch.x, batch.edge_index, batch =batch.batch)
            elif self.config.local_gnn_type == "sage":
                batch.x = self.layers[layer](batch.x, batch.edge_index, batch =batch.batch)
            else:
                batch.x = self.layers[layer](batch.x, batch.edge_index, batch =batch.batch, edge_attr = batch.edge_attr)
        x = global_mean_pool(batch.x, batch.batch)
        return self.mlp(x)

    @property 
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention

class GPS(torch.nn.Module):
    def __init__(self, config, channels: int, pe_dim: int, num_layers: int,
                 attn_type: str, attn_kwargs):
        super().__init__()

        self.node_emb = Embedding(28, channels)
        self.edge_emb = Embedding(4, channels)
        
        self.pos_encoder = WavePE_Encoder(config)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(nn), heads=4,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, batch):
        batch.x = self.node_emb(batch.x.squeeze(-1))
        batch.edge_attr = self.edge_emb(batch.edge_attr)
        batch = self.pos_encoder(batch) 
        for conv in self.convs:
            batch.x = conv(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
        x = global_add_pool(batch.x, batch.batch)
        return self.mlp(x)
    @property 
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1
