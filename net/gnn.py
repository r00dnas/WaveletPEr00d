import torch 
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from layer.gated_layer import GatedGCNLayer
from layer.gine_layer import GINEConvLayer, GIN_WithEdgeUpdate_Conv, GINEConv_v2
from torch_geometric.graphgym import GNNGraphHead, global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from encoder.wavepe_encoder import WavePE_Encoder
import torch.nn.functional as F

class GNN_VirtualNode(nn.Module):
    def __init__(self, config, out_dim = 11):
        super().__init__()

        self.config = config
        self.num_layer = config.num_layer
        
        self.atom_encoder = AtomEncoder(config.atom_dim)
        self.bond_encoder = BondEncoder(config.bond_dim)

        #### positional encoder ####
        self.pos_encoder = WavePE_Encoder(config)

        #dim = self.config.atom_dim + self.config.pos_dim 
        dim = self.config.atom_dim
        edge_dim = self.config.bond_dim + 2 * self.config.pos_dim
         
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
        
        self.prediction_head = GNNGraphHead(dim, out_dim)
        self.prediction_head.pooling_fun = global_mean_pool


    def forward(self, batch):
        vn_embedding = self.vn_embedding(torch.zeros(batch.batch[-1].item() + 1).to(batch.edge_index.dtype).to(batch.x.device))  
        batch.x = self.atom_encoder(batch.x)
        batch.edge_attr = self.bond_encoder(batch.edge_attr) 
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
        return self.prediction_head(batch)[0]

    @property 
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class GNN(nn.Module):
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
        batch.x = self.atom_encoder(batch.x)
        batch.edge_attr = self.bond_encoder(batch.edge_attr) 
        batch = self.pos_encoder(batch) 
        #batch.x = self.pre_mp(batch.x)
        for layer in range(self.num_layer): 
            skip_x = batch.x
            if self.config.local_gnn_type == "gated_gcn":
                batch = self.layers[layer](batch) 
            else:
                batch.x = self.layers[layer](batch.x, batch.edge_index, batch.edge_attr)
            batch.x = self.norms[layer](batch.x)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                batch.x = F.dropout(batch.x, self.config.dropout, training = self.training)
            else:
                batch.x = F.dropout(F.relu(batch.x), self.config.dropout, training = self.training)
        return self.prediction_head(batch)[0]

    @property 
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class GINETransfer(nn.Module):
    def __init__(self, config, out_dim):
        super().__init__()

        self.config = config
        self.num_layer = config.num_layer
        self.dropout = config.dropout

        if config.dataset_name =='zinc':
            self.atom_encoder = nn.Embedding(21, config.atom_dim)
        else:
            self.atom_encoder = AtomEncoder(config.atom_dim)
        #self.bond_encoder = BondEncoder(config.bond_dim)

        #### positional encoder ####
        self.pos_encoder = WavePE_Encoder(config)

        #dim = self.config.atom_dim + self.config.pos_dim 
        dim = self.config.atom_dim
        edge_dim = self.config.bond_dim

        if config.scale: 
            self.mlp = nn.Sequential(nn.Linear(dim, dim*2), nn.ReLU(), nn.Linear(dim*2,dim), nn.Dropout(self.config.dropout))
        
        self.layers = nn.ModuleList()
         
        for i in range(self.num_layer):
            self.layers.append(GINEConv_v2(dim, self.config.dropout, self.config.residual, edge_dim, 
                                           self.config.use_norm, self.config.affine)) 
        
        # self.prediction_head = GNNGraphHead(dim + 20, out_dim)
        # self.prediction_head.pooling_fun = global_mean_pool
        #self.mlp_pe = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 20))
        dim = dim + 20
        self.prediction_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1))
        self.residual = config.residual

    def forward(self, batch):
        if self.config.dataset_name == "zinc":
            batch.x = batch.x.squeeze()
        batch.x = self.atom_encoder(batch.x)
        x = batch.x
        batch = self.pos_encoder(batch)
        #batch.x = x
        # if self.config.scale:
        #     batch.x = self.mlp(batch.x)
        for layer in range(self.num_layer): 
            batch = self.layers[layer](batch) 

        pe = batch.pos
        h =  batch.x
        h = torch.cat([h, pe], dim = -1)
        h = global_add_pool(h, batch.batch)
        return self.prediction_head(h)

        return self.prediction_head(batch)[0]

    @property 
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class GATTransfer(nn.Module):
    def __init__(self, config, out_dim):
        super().__init__()

        self.config = config
        self.num_layer = config.num_layer
        self.dropout = config.dropout

        if config.dataset_name =='zinc':
            self.atom_encoder = nn.Embedding(21, config.atom_dim)
        else:
            self.atom_encoder = AtomEncoder(config.atom_dim)
            self.bond_encoder = BondEncoder(config.bond_dim)
        #self.bond_encoder = BondEncoder(config.bond_dim)

        #### positional encoder ####
        self.pos_encoder = WavePE_Encoder(config)

        #dim = self.config.atom_dim + self.config.pos_dim 
        dim = self.config.atom_dim
 
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
         
        for i in range(self.num_layer):
            self.layers.append(pyg_nn.GATv2Conv(dim, dim, heads = 4, edge_dim = dim, concat = False))
            self.norms.append(nn.LayerNorm(dim))
        
        self.prediction_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, out_dim))
        self.residual = config.residual

    def forward(self, batch):
        if self.config.dataset_name == "zinc":
            batch.x = batch.x.squeeze()
        batch.x = self.atom_encoder(batch.x)
        batch.edge_attr = self.bond_encoder(batch.edge_attr)
        batch = self.pos_encoder(batch)
        for layer in range(self.num_layer): 
            x_prev = batch.x
            batch.x = self.layers[layer](batch.x, batch.edge_index, batch.edge_attr) 
            batch.x = F.relu(self.norms[layer](batch.x))
            batch.x = batch.x + x_prev
        h = global_mean_pool(batch.x, batch.batch)
        return self.prediction_head(h)

    @property 
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)