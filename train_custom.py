# train_custom.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.transforms import RandomLinkSplit
from pt_dataset import PTDataset

def get_pos_neg(d):
    if hasattr(d, "edge_label") and hasattr(d, "edge_label_index"):
        pos = d.edge_label_index[:, d.edge_label == 1]
        neg = d.edge_label_index[:, d.edge_label == 0]
        return pos, neg
    return d.pos_edge_label_index, getattr(d, "neg_edge_label_index", None)

class EdgeAwareAE(nn.Module):
    def __init__(self, in_dim, edge_dim, hid=256, out=128):
        super().__init__()
        self.c1 = TransformerConv(in_dim, hid, heads=4, concat=True, edge_dim=edge_dim)
        self.c2 = TransformerConv(hid*4, out, heads=1, concat=False, edge_dim=edge_dim)
    def encode(self, x, ei, ea):
        h = F.relu(self.c1(x, ei, ea))
        return self.c2(h, ei, ea)

def recon_loss(z, pos_e, neg_e=None, device="cpu"):
    def score(e): return (z[e[0]] * z[e[1]]).sum(-1)
    pos = F.binary_cross_entropy_with_logits(score(pos_e), torch.ones(pos_e.size(1), device=device))
    if neg_e is None:
        return pos
    neg = F.binary_cross_entropy_with_logits(score(neg_e), torch.zeros(neg_e.size(1), device=device))
    return pos + neg

# ----- main -----
root = r"C:\Users\r00dn\Wavelet\WaveletPEr00d\Data"
data = PTDataset(root)[0]                      # expects x:[N,64], edge_attr:[E,130]
split = RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True, add_negative_train_samples=True)
tr, va, te = split(data)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x, ei, ea = data.x.to(dev), data.edge_index.to(dev), data.edge_attr.to(dev)

model = EdgeAwareAE(in_dim=x.size(1), edge_dim=ea.size(1)).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=5e-4)

for e in range(1, 401):
    model.train(); opt.zero_grad()
    z = model.encode(x, ei, ea)                # encode on full graph
    pos, neg = get_pos_neg(tr)
    loss = recon_loss(z, pos.to(dev), neg.to(dev), device=dev.type)
    loss.backward(); opt.step()
    if e % 20 == 0:
        print(f"epoch {e:03d} loss {loss.detach().item():.4f}")

with torch.no_grad():
    Z = model.encode(x, ei, ea).cpu()
torch.save(Z, r"C:\Users\r00dn\Wavelet\WaveletPEr00d\Data\processed\emb_custom.pt")
print("saved -> Data\\processed\\emb_custom.pt", tuple(Z.shape))
