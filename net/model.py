import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
import net.equi_layer as equi_layer


class Equivariant_SecondOrder_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.config.device = "cpu"
        self.build_model()
        
    def build_model(self):
        self.layers = nn.ModuleList([])
        self.layers.append(equi_layer.layer_2_to_2(self.config.num_scale, self.config.hidden_dims[0], device = self.config.device))
        for i in range(1, self.config.num_layer):
            self.layers.append(equi_layer.layer_2_to_2(self.config.hidden_dims[i-1], self.config.hidden_dims[i], device = self.config.device))
        self.layers.append(equi_layer.layer_2_to_1(self.config.hidden_dims[-1], self.config.hidden_dims[-1], device = self.config.device))
        self.mlp = nn.Sequential(
            nn.Linear(self.config.hidden_dims[-1], self.config.hidden_dims[-1] * 2), 
            nn.ReLU(),
            nn.Linear(self.config.hidden_dims[-1] * 2, self.config.latent_dim)
        )
        self.num_layer = len(self.layers)
    
    def forward(self, data):
        dense_x, mask = pyg_utils.to_dense_batch(data.x, data.batch)
        W = pyg_utils.to_dense_adj(data.edge_index_wavepe, data.batch, data.edge_attr_wavepe)
        W = W.permute(0, 3, 1, 2).float() #### b x k x n x n
        A = pyg_utils.to_dense_adj(data.edge_index, data.batch)
        mask = mask.to(W.dtype)
        out = W
        for i in range(self.num_layer):
            out = self.layers[i](out)
            out = F.relu(out)  #### b x k x n
        out = out.permute(0, 2, 1)
        out = self.mlp(out).permute(0, 2, 1)
        return out, A, mask 

    def encode(self, data):
        dense_x, mask = pyg_utils.to_dense_batch(data.x, data.batch)
        W = pyg_utils.to_dense_adj(data.edge_index_wavepe, data.batch, data.edge_attr_wavepe)
        W = W.permute(0, 3, 1, 2).float() #### b x k x n x n
        out = W
        for i in range(self.num_layer):
            out = self.layers[i](out)
            out = F.relu(out)  #### b x k x n
        out = out.permute(0, 2, 1)
        out = self.mlp(out) ### b x n x k
        out = out * mask.float().unsqueeze(-1).to(out.device)
        out = out.transpose(1, 2) 
        # edge_out = torch.einsum("...i,...j -> ...ij", out, out) 
        out = out.transpose(1, 2)
        return out[mask, :] #edge_out.permute(0, 3, 1, 2) #### convert to contiguous batch of tensor as data.x in torch geometric ####

class Equivariant_SecondOrder_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        self.reversed_hidden_dims = self.config.hidden_dims[::-1]
        self.layers.append(equi_layer.layer_1_to_2(self.config.latent_dim, self.reversed_hidden_dims[0], device = self.config.device))
        for i in range(1, self.config.num_layer):
            self.layers.append(equi_layer.layer_2_to_2(self.reversed_hidden_dims[i-1], self.reversed_hidden_dims[i], device = self.config.device))
        self.layers.append(equi_layer.layer_2_to_2(self.reversed_hidden_dims[-1], self.reversed_hidden_dims[-1], device = self.config.device))
        self.mlp = nn.Sequential(
            nn.Linear(self.reversed_hidden_dims[-1], self.reversed_hidden_dims[-1] * 2), 
            nn.ReLU(),
            nn.Linear(self.reversed_hidden_dims[-1] * 2, self.config.num_scale)
        )
        self.num_layer = len(self.layers)

    def forward(self, P, mask):
        out = P
        mask = mask.unsqueeze(1)
        for i in range(self.num_layer):
            out = self.layers[i](out)
            out = F.relu(out)  #### b x k x n x n
        out = out.permute(0, 2, 3, 1) ### b x n x n x k
        out = self.mlp(out).permute(0, 3, 1, 2)
        #out = mask.unsqueeze(-1) * out * mask.unsqueeze(-2)
        return out

 
class WavePE_AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.build_model()
    
    def build_model(self):
        self.encoder = Equivariant_SecondOrder_Encoder(self.config)
        self.decoder = Equivariant_SecondOrder_Decoder(self.config)
        
    def num_parameters(self):
        return sum(p.numel() for p in list(self.encoder.parameters()) + list(self.decoder.parameters()))

    def fit(self, data):
        P, W, mask = self.encoder(data)
        W_hat = self.decoder(P, mask)
        W_hat = (W_hat + W_hat.permute(0,1,3,2)) / 2
        W_hat = W_hat.sigmoid()
        return W_hat, W, mask
    
    def forward(self, data):
        P, W, mask = self.encoder(data)
        W_hat = self.decoder(P, mask)
        return W_hat, W

    def get_pe(self, data):
        return self.encoder.encode(data)
