import argparse
import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader

from torch_geometric.utils import degree
import transformers

import torch_geometric.transforms as T
from torch_geometric.datasets import MNISTSuperpixels, GNNBenchmarkDataset
from tqdm import tqdm

import time
from transform import WaveletTransform
from utils.commons import load_config
from utils.metrics import MAE
from net.gnn_zinc import GNN_VirtualNode, GNN, GraphTransformer

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
transform = T.Cartesian(cat=False)

train_dataset= GNNBenchmarkDataset(root = path, name = args.dataset, split = 'train', pre_transform=pre_transform)
val_dataset= GNNBenchmarkDataset(root = path, name = args.dataset, split = 'val', pre_transform=pre_transform)
test_dataset= GNNBenchmarkDataset(root = path, name = args.dataset, split = 'test', pre_transform=pre_transform)


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size =args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model_type == "gnn":
    model = GNN(args, out_dim = 10).to(device)
elif args.model_type == "gnn_vn":
    model = GNN_VirtualNode(args, out_dim = 10).to(device)
elif args.model_type == "GT":
    model = GraphTransformer(args, out_dim = 10).to(device)
else:
    raise NotImplemented

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.scheduler == "reduce_on_plateau":
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50,
                                min_lr=1e-6, verbose=True)
elif args.scheduler == "cosine_with_warmup":
    scheduler = transformers.optimization.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_epoch)

print(f"Model: ", args.model_type)
print(f"Local GNN: ", args.local_gnn_type)
print("Number of parameters: ", model.num_trainable_parameters)

def train(epoch):
    model.train()

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 26:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
    total_loss = 0
    for idx, data in enumerate(tqdm(train_loader, total = len(train_loader))):
        data = data.to(device)
        optimizer.zero_grad()
        #out = F.log_softmax(model(data), dim = 1)
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_dataset)


def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()

    return correct / len(test_dataset)

best_test = 0
best_val = 0
for epoch in range(1, args.num_epoch):
    train_loss = train(epoch)
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    scheduler.step()
    if val_acc > best_val:
        best_val = val_acc
        best_test = test_acc
    print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}, Val: {val_acc:4f}, Test: {test_acc:.4f}, Best Test: {best_test:.4f}\n')

print("****** \ ******")
print(f"Dataset: {args.dataset} | Model Type: {args.model_type} | Local GNN Type: {args.local_gnn_type} | Best Acc: {best_test:.4f}")