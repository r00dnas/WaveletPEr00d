import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader

class GraphDataset(Dataset):
    def __init__(self, n=32, nodes=5, feat=8):
        self.n, self.nodes, self.feat = n, nodes, feat
    def __len__(self): return self.n
    def __getitem__(self, idx):
        x = torch.randn(self.nodes, self.feat)
        # simple ring graph
        edges = [(i, (i+1)%self.nodes) for i in range(self.nodes)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
        # fields expected by your model
        edge_index_wavepe = edge_index.clone()
        edge_attr_wavepe  = torch.ones(edge_index_wavepe.size(1), 1, dtype=torch.float)  # [E]
        # optional generic attr
        edge_attr = edge_attr_wavepe.clone()
        return Data(x=x,
                    edge_index=edge_index,
                    edge_index_wavepe=edge_index_wavepe,
                    edge_attr_wavepe=edge_attr_wavepe,
                    edge_attr=edge_attr)

# minimal Lightning DataModule that returns PyG batches
try:
    import lightning as L
except Exception:
    L = None

class Lightning_Dataset(L.LightningDataModule if L else object):
    def __init__(self, *args, **kwargs):
        if L: super().__init__()
    def train_dataloader(self):
        return GeoLoader(GraphDataset(), batch_size=4, shuffle=True)
    def val_dataloader(self):
        return GeoLoader(GraphDataset(), batch_size=4)
    def test_dataloader(self):
        return GeoLoader(GraphDataset(), batch_size=4)
