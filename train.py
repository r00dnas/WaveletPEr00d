
from torch_geometric.datasets import LRGBDataset, MoleculeNet
from data import WaveletTransform


transform = WaveletTransform([2,3,4], 3, 1e-5)
dataset = LRGBDataset(root = "/cm/shared/khangnn4/WavePE/data", name = "PCQM-Contact")
