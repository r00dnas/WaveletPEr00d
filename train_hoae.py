# train_hoae.py  — minimal HOAE-style edge-reconstruction to learn node embeddings
import argparse, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

def undirected_edges_with_attr(data):
    ei = data.edge_index.t().tolist()
    EA = data.edge_attr
    if EA is None:
        raise ValueError("data.edge_attr is required")
    acc = {}
    for (u, v), attr in zip(ei, EA):
        a, b = (u, v) if u < v else (v, u)
        acc.setdefault((a, b), []).append(attr.float())
    U, V, Y = [], [], []
    for (a, b), attrs in acc.items():
        U.append(a); V.append(b)
        Y.append(torch.stack(attrs, 0).mean(0))
    return torch.tensor(U, dtype=torch.long), torch.tensor(V, dtype=torch.long), torch.stack(Y, 0)

class MLP(nn.Module):
    def __init__(self, dims, act=nn.ReLU, dropout=0.1):
        super().__init__()
        layers=[]
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1])]
            if i < len(dims)-2:
                layers += [act(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class HOAE(nn.Module):
    def __init__(self, n_nodes, emb_dim, out_dim):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, emb_dim)
        self.dec = MLP([emb_dim*2, 256, out_dim], dropout=0.2)
        nn.init.xavier_uniform_(self.emb.weight)

    def pair_decode(self, u, v):
        hu = self.emb(u); hv = self.emb(v)
        feats = torch.cat([torch.abs(hu - hv), hu * hv], dim=-1)  # symmetric + multiplicative
        return self.dec(feats)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=r".\Data\processed\data.pt")
    ap.add_argument("--out",  type=str, default=r".\Data\processed\hoae_nodes.pt")
    ap.add_argument("--dim",  type=int, default=128)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    N = int(data.num_nodes)
    u, v, Y = undirected_edges_with_attr(data)  # Y: [E, D]
    D = int(Y.size(1))
    print(f"T: torch.Size([{N}, {N}, {D}])")

    model = HOAE(N, args.dim, D)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    mse = nn.MSELoss()

    for epoch in range(1, args.epochs+1):
        model.train()
        pred = model.pair_decode(u, v)         # [E, D]
        loss = mse(pred, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:03d} mse {loss.item():.6f}")

    with torch.no_grad():
        H = model.emb.weight.detach().cpu().float()  # [N, dim]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(H, args.out)  # weights_only=False default
    print(f"saved {args.out} {tuple(H.shape)}")

if __name__ == "__main__":
    main()
