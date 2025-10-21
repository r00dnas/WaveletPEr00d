import torch
import torch.nn as nn
import torch.nn.functional as F  # if you use F.relu, etc.

# run_wavepe.py
from torch_geometric.loader import DataLoader
from pt_dataset import PTDataset
# from models.wavepe import WavePEModel
# model = WavePEModel(in_dim=41, ...)

ds = PTDataset(r"C:\Users\r00dn\Wavelet\WaveletPEr00d\Data")
for batch in DataLoader(ds, batch_size=1):
    print(batch.x.shape, batch.edge_index.shape)
    # out = model(batch)   # <- your code here
