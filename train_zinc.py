import argparse
import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader

from torch_geometric.utils import degree
import transformers

import time
from transform import WaveletTransform
from utils.commons import load_config
from utils.metrics import MAE
from net.gnn_zinc import GNN_VirtualNode, GNN, GraphTransformer, GPS

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, default = "zinc")
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
parser.add_argument("--val_batch_size", type = int, default = 64)

args = parser.parse_args()
print(args.concat)
pretrained_config = load_config(f"config/pretrain.yml")
transform = WaveletTransform(pretrained_config.scales, approximation_order=pretrained_config.approximation_order, tolerance=pretrained_config.tolerance) 

path = "/cm/shared/khangnn4/WavePE/data/zinc"
train_dataset = ZINC(path, subset=True, split='train', pre_transform=transform)
val_dataset = ZINC(path, subset=True, split='val', pre_transform=transform)
test_dataset = ZINC(path, subset=True, split='test', pre_transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, num_workers=0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model_type == "gnn":
    model = GNN(args, out_dim = 1).to(device)
elif args.model_type == "gnn_vn":
    model = GNN_VirtualNode(args, out_dim = 1).to(device)
elif args.model_type == "GT":
    attn_kwargs = {'dropout': 0.5}
    model = GPS(args, channels=64, pe_dim=20, num_layers=10, attn_type="multihead",
            attn_kwargs=attn_kwargs).to(device)
else:
    raise NotImplemented

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

if args.scheduler == "reduce_on_plateau":
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                min_lr=0.00001, verbose=True)
elif args.scheduler == "cosine_with_warmup":
    scheduler = transformers.optimization.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_epoch)

print(f"Model: ", args.model_type)
print(f"Local GNN: ", args.local_gnn_type)
print("Number of parameters: ", model.num_trainable_parameters)

def train(epoch):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        model.redraw_projection.redraw_projections()
        out = model(data)
        loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        # if args.model_type == "GT":
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_mae = 0
    total_loss = 0
    for i, data in enumerate(loader):
        data = data.to(device)
        out = model(data)
        loss = (out.squeeze() - data.y).abs().mean()
        total_loss += loss.item() * data.num_graphs
        total_mae += MAE(out.view(-1), data.y.view(-1))
    return total_mae / (i + 1), total_loss / len(loader.dataset)

for epoch in range(1, args.num_epoch):
    start = time.time()
    loss = train(epoch)
    val_loss, val_mae = test(val_loader)
    test_loss, test_mae = test(test_loader)
    end = time.time()
    if args.scheduler == "cosine_with_warmup":
        scheduler.step()
    elif args.scheduler == "reduce_on_plateau":
        scheduler.step(val_loss)
    else:
        pass
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_loss:.4f}, '
          f'Test: {test_loss:.4f}, Time: {end - start:.4f}')