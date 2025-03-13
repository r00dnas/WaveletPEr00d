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
     

        #### positional encoder ####
        self.pos_encoder = WavePE_Encoder(config)
        self.node_encoder = nn.Linear(20, config.atom_dim)

        
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
                self.layers.append(pyg_nn.TransformerConv(dim, dim, heads = 4, concat = False, edge_dim = None))
                self.norms.append(nn.BatchNorm1d(dim))
            elif config.local_gnn_type == "gin":
                mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
                conv = pyg_nn.GINConv(mlp)
                self.layers.append(conv)
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
        batch.x = self.pos_encoder.encoder.encode(batch)
        batch.x = self.node_encoder(batch.x) 
        vn_embedding = self.vn_embedding(torch.zeros(batch.batch[-1].item() + 1).to(batch.edge_index.dtype).to(batch.x.device))  
     
        #batch.x = self.pre_mp(batch.x)
        for layer in range(self.num_layer):
            batch.x = batch.x + vn_embedding[batch.batch]
            if self.config.local_gnn_type == "gated_gcn":
                batch = self.layers[layer](batch)
            else:
                if self.config.local_gnn_type in ["gin", "transformer_conv"]:
                    batch.x = self.layers[layer](batch.x, batch.edge_index)
                else:
                    batch.x = self.layers[layer](batch.x, batch.edge_index, batch.edge_attr)
                batch.x = F.leaky_relu(self.norms[layer](batch.x), negative_slope=0.1)
                batch.x = F.dropout(batch.x, p = self.config.dropout, training = self.training)
            if layer < self.num_layer - 1:
                vn_embedding_temp = global_add_pool(batch.x, batch.batch) + vn_embedding
                vn_embedding = F.dropout(self.mlp_vn[layer](vn_embedding_temp), p = self.config.dropout, training = self.training)
        x = global_add_pool(batch.x, batch.batch)
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
         
        
        if self.config.dataset == "NCI1":
            #self.node_encoder = nn.Linear(37, config.atom_dim)
            self.node_encoder = pyg_nn.MLP([37, config.atom_dim, config.atom_dim], dropout=0.5)
            input_dim = 52
        elif self.config.dataset == "NCI109":
            #self.node_encoder = nn.Linear(37, config.atom_dim)
            self.node_encoder = pyg_nn.MLP([38, config.atom_dim, config.atom_dim], dropout = 0.5)
            input_dim = 52
        elif self.config.dataset == "MUTAG":
            #self.node_encoder = nn.Linear(37, config.atom_dim)
            self.node_encoder = pyg_nn.MLP([7, config.atom_dim, config.atom_dim], dropout=0.5)
            input_dim = 52
        elif self.config.dataset == "PROTEINS":
            #self.node_encoder = nn.Linear(37, config.atom_dim)
            self.node_encoder = pyg_nn.MLP([3, config.atom_dim, config.atom_dim], dropout=0.5)
            input_dim = 52
        else:
            input_dim = 20

        #### positional encoder ####
        self.pos_encoder = WavePE_Encoder(config)

        dim = config.atom_dim
        
    
        self.layers = nn.ModuleList()
        for _ in range(self.num_layer):
            mlp = pyg_nn.MLP([input_dim, dim, dim])
            self.layers.append(pyg_nn.GINConv(nn = mlp, train_eps = False))
            input_dim = dim
        
    
        self.mlp = pyg_nn.MLP([dim, dim, out_dim], norm = None, dropout=0.5)


    def forward(self, batch):
        if self.config.dataset in ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI']:
            batch.x = self.pos_encoder.encoder.encode(batch)
            #batch.x = self.node_encoder(batch.x)
            x = batch.x
        else: 
            x = self.node_encoder(batch.x) 
            node_pe = self.pos_encoder.encoder.encode(batch) 
            x = torch.cat([x, node_pe], dim = -1)
        
        for layer in self.layers:
            x = layer(x, batch.edge_index).relu()
        
        h = global_add_pool(x, batch.batch)

        return self.mlp(h)

    @property 
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class GraphTransformer(nn.Module):
    def __init__(self, config, out_dim):
        super().__init__()

        self.config = config
        self.num_layer = config.num_layer
         
        if config.dataset in ['ENZYMES', "PROTEINS"]:
            self.node_encoder = nn.Linear(3, config.atom_dim) 
        elif config.dataset in ['COLLAB', 'IMDB-BINARY']:
            self.node_encoder = None
        elif config.dataset == 'NCI1':
            self.node_encoder = nn.Linear(10, config.atom_dim)
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
            nn.Dropout(0.5), 
            nn.ReLU(),
            nn.Linear(dim // 4, out_dim),
        )
    
    def forward(self, batch):
        if self.config.dataset in ['COLLAB', 'IMDB-BINARY']:
            batch.x = self.pos_encoder.encoder.encode(batch)
            batch.x = self.node_encoder(batch.x)
        else:
            batch.x = self.node_encoder(batch.x)
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

class DeepWaveletNetwork(nn.Module):
    def __init__(self, config, out_dim):
        super().__init__()

        self.pos_encoder = WavePE_Encoder(config)
        self.node_encoder = nn.Linear(10, config.atom_dim)
        dim = 20 + config.atom_dim
        self.mlp_node = nn.Sequential(
            nn.Linear(dim, dim), 
            nn.ReLU(), 
            nn.Linear(dim, dim)
        )
        self.mlp_graph = nn.Sequential(
            nn.Linear(dim, dim // 2), 
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(), 
            nn.Linear(dim // 4, out_dim)
        )
    def forward(self, batch):
        batch.pe = self.pos_encoder.encoder.encode(batch)
        batch.x = self.node_encoder(batch.x)
        batch.x = torch.cat([batch.x, batch.pe], dim = -1)
        h = self.mlp_node(batch.x)
        h = global_add_pool(h, batch.batch)
        return self.mlp_graph(h)
        