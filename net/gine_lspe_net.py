import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool, GINEConv, global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from layer.gine_lspe import GINE_LSPE_Conv
from layer.sw_pooling import SWE_Pooling


class GINE_LSPE(nn.Module):
    def __init__(self, dim, n_layer, pooling, config = None):
        super().__init__()
        self.config = None
        if hasattr(config, "atom_dim"):
            self.node_emb = AtomEncoder(config.atom_dim)
            self.edge_emb = BondEncoder(config.bond_dim)
            self.config = config
        else:
            self.node_emb = nn.Embedding(21, dim)
            self.edge_emb = nn.Embedding(4, dim)

        self.pooling = config.graph_pooling
        if self.pooling == "OT":
            self.h_pooler = SWE_Pooling(dim, 20, 64)
            self.p_pooler = SWE_Pooling(20, 10, 8)

        self.n_layer = n_layer
        self.convs = nn.ModuleList()
        for _ in range(n_layer):
            nn_h = nn.Sequential(nn.Linear(dim + 20, dim + 20), nn.BatchNorm1d(dim + 20), nn.ReLU(), nn.Linear(dim + 20, dim))
            nn_p = nn.Sequential(nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, 20)) 
            conv = GINE_LSPE_Conv(nn_h, nn_p, edge_dim = dim)
            self.convs.append(conv)
            
        
            
        if self.pooling == "OT":
            dim = 72
            self.mlp = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 4), nn.ReLU(), nn.Linear(dim // 4, 1))
        else:
            #self.Whp = nn.Linear(dim + 20, dim) 
            dim = dim + 20
            self.mlp = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 4), nn.ReLU(), nn.Linear(dim // 4, 1))

    
        from net.lightning_model import Lightning_WavePE_AutoEncoder
        ckpt_path =  "/cm/shared/khangnn4/WavePE/ckpts/PCBA_debug_1/PCBA_epoch=99_train_loss=0.010_val_loss=0.011_val_best_loss=0.011.ckpt"
        autoencoder = Lightning_WavePE_AutoEncoder.load_from_checkpoint(ckpt_path, map_location = "cpu")
        self.pos_encoder = autoencoder.net.encoder
         

    def forward(self, data, x = None, edge_index = None, edge_attr = None, batch_index = None):
        if x is None:
            x, edge_index, edge_attr, batch_index = data.x, data.edge_index, data.edge_attr, data.batch
        if self.config is not None:
            h = self.node_emb(x)
        else:
            h = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr) 
        
        p = self.pos_encoder.encode(data)
        #print(p.sum())
        for i in range(self.n_layer):
            h_skip = h
            p_skip = p
            h, p= self.convs[i](h, p, edge_index, edge_attr, size = None, batch_index=batch_index)
            #h = F.dropout(h, p = self.config.dropout, training = self.training)
            h = h + h_skip
            p = p + p_skip

        if self.pooling == "OT":  
            h, mask = to_dense_batch(h, batch_index)
            h = self.h_pooler(h, mask)
            p, mask = to_dense_batch(p, batch_index)
            p = self.p_pooler(p, mask)
            #print(h.shape)
            h = torch.cat([h, p], dim = -1)
            return self.mlp(h)

        h = torch.cat([h, p], dim = -1)
        #h = self.Whp(h)
        #h = self.aggregator(h, batch_index)
        if self.pooling == 'mean':
            h = global_mean_pool(h, batch_index)
        else:
            h = global_add_pool(h, batch_index)
        return self.mlp(h)