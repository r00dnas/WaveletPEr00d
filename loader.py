# loader.py (or wherever you build datasets)
from torch_geometric.loader import DataLoader
from pt_dataset import PTDataset

ds = PTDataset(r"C:\Users\r00dn\Wavelet\WaveletPEr00d\Data")  # uses Data\processed\data.pt
loader = DataLoader(ds, batch_size=1, shuffle=False)
