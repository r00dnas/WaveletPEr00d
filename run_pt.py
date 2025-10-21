# run_pt.py
from torch_geometric.loader import DataLoader
from pt_dataset import PTDataset

root = r"C:\Users\r00dn\Wavelet\WaveletPEr00d\Data"
ds = PTDataset(root)                 # reads processed\data.pt
loader = DataLoader(ds, batch_size=1)

# plug loader into your existing training/eval entrypoint
# example:
# from train_eval_tu import train_epoch
# for batch in loader: train_epoch(batch)

# quick check
data = ds[0]
print(data)
