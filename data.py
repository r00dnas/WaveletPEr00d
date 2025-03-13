import torch
from torch.utils.data import Dataset, DataLoader
from tdc.generation import MolGen
from tdc.chem_utils import MolConvert
import pygsp
from pygsp import graphs, filters, plotting
import torch_geometric.utils as pyg_utils
from torch_geometric.loader import DataLoader as pyg_Dataloader
from torch_geometric.data import Batch
from tqdm import tqdm

import numpy as np
import torch_sparse
from torch_sparse import SparseTensor
from scipy import sparse

from lightning import LightningDataModule
from torch_geometric.datasets import MoleculeNet
from transform import WaveletTransform


mol_to_graph = MolConvert(src = "SMILES", dst = "PyG")

def add_node_attr(data, value,
                  attr_name):
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data        

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform = None):
        super().__init__()
        self.df = df.dropna()
        self.transform=transform
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        smiles = self.df.iloc[idx]['smiles']
        pyg_data = mol_to_graph(smiles)
        if self.transform is not None:
            pyg_data = self.transform(pyg_data)
        return pyg_data

def collate_pyg_graph(data_list):
    return Batch.from_data_list(data_list)

class Lightning_Dataset(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(logger = False)

        self.config = config

        transform = WaveletTransform(config.scales, approximation_order=config.approximation_order, tolerance=config.tolerance)

        if self.config.data_name == "ZINC":
            data = MolGen(name = self.config.data_name)
            split = data.get_split()
            ### use 1000 samples for debugging ###
            self.train_dataset = GraphDataset(split['train'][:1000], transform)
            self.val_dataset = GraphDataset(split['valid'], transform)
        elif self.config.data_name == "PCBA":
            dataset = MoleculeNet("./data", "PCBA", transform=transform)
            train_num = int(len(dataset) * 0.9)
            self.train_dataset = dataset[:train_num]
            self.val_dataset = dataset[train_num:]

    def train_dataloader(self):
        if self.config.data_name == "ZINC":
            return DataLoader(self.train_dataset, batch_size = self.config.batch_size, 
                            shuffle = True, collate_fn = collate_pyg_graph, num_workers=6)
        else:
            return pyg_Dataloader(self.train_dataset, batch_size = self.config.batch_size, shuffle = True, num_workers= 6)
        
    def val_dataloader(self):
        if self.config.data_name == "ZINC":
            return DataLoader(self.val_dataset, batch_size = self.config.batch_size, 
                            shuffle = False, collate_fn = collate_pyg_graph, num_workers=6)
        else:
            return pyg_Dataloader(self.val_dataset, batch_size = self.config.batch_size, shuffle = False, num_workers= 6)