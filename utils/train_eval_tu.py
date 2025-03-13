import sys
import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader


from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def k_fold(dataset, folds, epoch_select, n_splits):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]

    skf_semi = StratifiedKFold(n_splits, shuffle=True, random_state=12345)
    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        idx_train = train_mask.nonzero(as_tuple=False).view(-1)

        for _, idx in skf_semi.split(torch.zeros(idx_train.size()[0]), dataset.data.y[idx_train]):
            idx_train = idx_train[idx]
            break

        train_indices.append(idx_train)

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)