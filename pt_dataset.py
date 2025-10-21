import os, torch
from torch_geometric.data import Dataset

class PTDataset(Dataset):
    def __init__(self, root, pt_name="data.pt", **kw):
        self.pt_name = pt_name
        super().__init__(root, **kw)

    @property
    def raw_file_names(self): return []

    @property
    def processed_file_names(self): return [self.pt_name]

    def process(self): pass

    def len(self): return 1

    def get(self, idx):
        path = os.path.join(self.processed_dir, self.pt_name)
        return torch.load(path, weights_only=False)
