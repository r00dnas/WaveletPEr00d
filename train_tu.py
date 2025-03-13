import argparse
import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader

from torch_geometric.utils import degree
import transformers

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from tqdm import tqdm

import time
from transform import WaveletTransform
from utils.commons import load_config
from utils.metrics import MAE
from net.gnn_tu import GNN, GNN_VirtualNode, DeepWaveletNetwork
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, default = "mnist")
parser.add_argument("--model_type", type = str, default = "gnn_vn")
parser.add_argument("--num_layer", type = int, default = 5)
parser.add_argument("--local_gnn_type", type = str, default = "transformer_conv")
parser.add_argument("--atom_dim", type = int, default = 512)
parser.add_argument("--bond_dim", type = int, default = 512)
parser.add_argument("--scheduler", type = str, default = "cosine_with_warmup")
parser.add_argument("--warmup_steps", type = int, default = 10)
parser.add_argument("--num_epoch", type = int, default = 1000)
parser.add_argument("--no_concat", action="store_false", dest = "concat")
parser.add_argument("--learnable", action = "store_true", dest = "learnable")
parser.add_argument("--no_freeze", action = "store_false", dest = 'freeze')
parser.add_argument("--no_residual", action = "store_false", dest = "residual")
parser.add_argument("--lr", type =float, default = 1e-3)
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--ckpt_pos_encoder_path", type = str)
parser.add_argument("--dropout", type = float, default = 0.1)
parser.add_argument("--attn_dropout", type = float, default = 0.5)
parser.add_argument('--not_use_full_graph', action="store_false", dest = "use_full_graph")

args = parser.parse_args()
print(args.concat)
pretrained_config = load_config(f"config/pretrain.yml")
pre_transform = WaveletTransform(pretrained_config.scales, approximation_order=pretrained_config.approximation_order, tolerance=pretrained_config.tolerance) 


path = "/cm/shared/khangnn4/WavePE/data/"

class RandomFeature:
    def __call__(self, data):
        data.x = torch.randn(data.num_nodes, 10)
        return data


if args.dataset in ["IMDB-BINARY", "COLLAB", "IMDB-MULTI"]:
    transform = RandomFeature() 
else:
    transform = None
    

dataset = TUDataset(
    path, 
    name = args.dataset, 
    transform=transform,
    pre_transform=pre_transform
)

#out_dim = 3 if args.dataset in ['COLLAB', 'PROTEINS'] else 2
out_dim = dataset.num_classes



def train(epoch):
    model.train()
    total_loss = 0
    for idx, data in enumerate(tqdm(train_loader, total = len(train_loader))):
        data = data.to(device)
        optimizer.zero_grad()
        out = F.log_softmax(model(data), dim = 1)
        #out = model(data)
        #loss = F.cross_entropy(out, data.y)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_dataset)

@torch.no_grad()
def test(loader, test_dataset):
    model.eval()
    correct = 0
    for idx, data in enumerate(tqdm(loader, total = len(loader))):
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        # pred = model(data).argmax(dim  = -1)
        # correct += int((pred == data.y).sum())
    return correct / len(test_dataset)


if __name__=="__main__":
    n_splits = 2
    folds = 10
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=12345)
    all_acc = []
    for train_index, test_index in kf.split(torch.zeros(len(dataset)), dataset.data.y):
    
        test_dataset = dataset[test_index]
        train_dataset = dataset[train_index]
 

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.model_type == "gnn_vn":
            model = GNN_VirtualNode(args, out_dim = out_dim).to(device)
        else:
            model = GNN(args, out_dim = out_dim).to(device)
    
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

        if args.scheduler == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50,
                                        min_lr=1e-6, verbose=True)
        elif args.scheduler == "cosine_with_warmup":
            scheduler = transformers.optimization.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_epoch)
        
        best_test = 0

        for epoch in range(1, args.num_epoch):
            train_loss = train(epoch) 
            test_acc = test(test_loader, test_dataset)
            if test_acc > best_test:
                best_test = test_acc
            scheduler.step()
            print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}, Test: {test_acc:.4f}, Best Test: {best_test:.4f}\n')

        print("****** \ ******")
        print(f"Dataset: {args.dataset} | Model Type: {args.model_type} | Local GNN Type: {args.local_gnn_type} | Best Acc: {best_test:.4f}")
        all_acc.append(best_test)

    print("Average accuracy: ", np.mean(all_acc))
    print("Std: ", np.std(all_acc))