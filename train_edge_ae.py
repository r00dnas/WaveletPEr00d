import torch, torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_geometric.transforms import RandomLinkSplit
from pt_dataset import PTDataset

def get_pos_neg(d):
    if hasattr(d, "edge_label") and hasattr(d, "edge_label_index"):
        pos = d.edge_label_index[:, d.edge_label == 1]
        neg = d.edge_label_index[:, d.edge_label == 0]
        return pos, neg
    return d.pos_edge_label_index, getattr(d, "neg_edge_label_index", None)

class Encoder(nn.Module):
    def __init__(self, in_dim, edge_dim, hid=128, out=64):
        super().__init__()
        self.c1 = TransformerConv(in_dim, hid, heads=4, concat=True, edge_dim=edge_dim)
        self.c2 = TransformerConv(hid*4, out, heads=1, concat=False, edge_dim=edge_dim)
    def forward(self, x, ei, ea):
        h = torch.relu(self.c1(x, ei, ea))
        z = self.c2(h, ei, ea)
        return z

def recon_loss(z, pos_e, neg_e=None):
    def score(e):
        return (z[e[0]] * z[e[1]]).sum(-1)    # dot
    pos = torch.nn.functional.binary_cross_entropy_with_logits(score(pos_e), torch.ones(pos_e.size(1)))
    if neg_e is None: return pos
    neg = torch.nn.functional.binary_cross_entropy_with_logits(score(neg_e), torch.zeros(neg_e.size(1)))
    return pos + neg

@torch.no_grad()
def eval_auc_ap(z, pos, neg):
    s_pos = (z[pos[0]] * z[pos[1]]).sum(-1)
    s_neg = (z[neg[0]] * z[neg[1]]).sum(-1)
    y = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)])
    s = torch.cat([s_pos, s_neg]).sigmoid()
    # simple AUC/AP without sklearn
    rank = s.argsort().argsort().float() / (s.numel() - 1)
    auc = 1 - (rank[:len(s_pos)].mean() - rank[len(s_pos):].mean()).item()
    # AP approx
    ap = (s[:len(s_pos)].mean() / s.mean()).clamp(0,1).item()
    return auc, ap

root = r"C:\Users\r00dn\Wavelet\WaveletPEr00d\Data"
data = PTDataset(root)[0]  # has x, edge_index, edge_attr
split = RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True, add_negative_train_samples=True)
tr, va, te = split(data)

model = Encoder(in_dim=data.x.size(1), edge_dim=data.edge_attr.size(1))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for e in range(1, 201):
    model.train(); opt.zero_grad()
    z = model(data.x, data.edge_index, data.edge_attr)
    pos, neg = get_pos_neg(tr)
    loss = recon_loss(z, pos, neg)
    loss.backward(); opt.step()
    if e % 20 == 0:
        model.eval()
        zv = model(data.x, data.edge_index, data.edge_attr)
        pv, nv = get_pos_neg(va)
        auc, ap = eval_auc_ap(zv, pv, nv)
        print(f"epoch {e:03d} loss {loss.detach().item():.4f} valAUC {auc:.3f} AP {ap:.3f}")

with torch.no_grad():
    Z = model(data.x, data.edge_index, data.edge_attr)
torch.save(Z, r"C:\Users\r00dn\Wavelet\WaveletPEr00d\Data\processed\emb_edge.pt")
print("saved -> Data\\processed\\emb_edge.pt", tuple(Z.shape))
