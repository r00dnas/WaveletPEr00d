# train_gae.py
import torch
from torch_geometric.nn import GAE, GCNConv
from torch_geometric.transforms import RandomLinkSplit
from pt_dataset import PTDataset

class Encoder(torch.nn.Module):
    def __init__(self, in_dim, hid=128, out=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, out)
    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

def get_pos_neg(d):
    if hasattr(d, "edge_label") and hasattr(d, "edge_label_index"):
        pos = d.edge_label_index[:, d.edge_label == 1]
        neg = d.edge_label_index[:, d.edge_label == 0]
        return pos, neg
    return d.pos_edge_label_index, getattr(d, "neg_edge_label_index", None)

root = r"C:\Users\r00dn\Wavelet\WaveletPEr00d\Data"
data = PTDataset(root)[0]

split = RandomLinkSplit(num_val=0.05, num_test=0.10,
                        is_undirected=True, add_negative_train_samples=True)
train_data, val_data, test_data = split(data)

enc = Encoder(data.x.size(1))
model = GAE(enc)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def train():
    model.train(); opt.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    pos, neg = get_pos_neg(train_data)
    loss = model.recon_loss(z, pos, neg)
    loss.backward(); opt.step()
    return float(loss)

@torch.no_grad()
def evaluate(d):
    model.eval()
    z = model.encode(d.x, d.edge_index)
    pos, neg = get_pos_neg(d)
    return model.test(z, pos, neg)

for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        auc, ap = evaluate(val_data)
        print(f"epoch {epoch:03d} loss {loss:.4f} valAUC {auc:.3f} AP {ap:.3f}")

with torch.no_grad():
    Z = model.encode(data.x, data.edge_index)
torch.save(Z, r"C:\Users\r00dn\Wavelet\WaveletPEr00d\Data\processed\emb.pt")
print("saved -> Data\\processed\\emb.pt", tuple(Z.shape))
