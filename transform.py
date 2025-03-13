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

class FilterNoneEdge():
    def __call__(self, data):
        if data.edge_index.size[1] == 0:
            return False
        return True

class WaveletTransform:
    def __init__(self, scales, approximation_order, tolerance):
        self.scales = scales 
        self.approximation_order = approximation_order
        self.tolerance = tolerance

    def __call__(self, data):
        A = pyg_utils.to_dense_adj(data.edge_index)[0].numpy()
        G = graphs.Graph(A) 
        try:
            G.estimate_lmax()
        except:
            G._lmax = 2.0 
        n_node = A.shape[0]
        wavelets = self.calculate_all_wavelets(G, n_node)
        rel_pe = SparseTensor.from_dense(wavelets, has_value=True)
        rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
        rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)
        data = add_node_attr(data, rel_pe_idx, "edge_index_wavepe")
        data = add_node_attr(data, rel_pe_val, "edge_attr_wavepe")
        return data
        
    def calculate_wavelet(self, graph, n_node, chebyshev):
        impulse = np.eye(n_node, dtype = int)
        wavelet_coefficients = pygsp.filters.approximations.cheby_op(
            graph, chebyshev, impulse 
        )
        wavelet_coefficients[wavelet_coefficients < self.tolerance] = 0
        ind_1, ind_2 = wavelet_coefficients.nonzero()
        n_count = n_node
        return torch.from_numpy(wavelet_coefficients)

    def calculate_all_wavelets(self, graph, n_node):
        wavelet_tensors = []
        for i, scale in enumerate(self.scales):
            heat_filter = pygsp.filters.Heat(graph, tau = [scale])
            chebyshev = pygsp.filters.approximations.compute_cheby_coeff(heat_filter, m = self.approximation_order)
            wavelets = self.calculate_wavelet(graph, n_node, chebyshev)
            wavelet_tensors.append(wavelets)
        return torch.stack(wavelet_tensors, dim = -1)
        