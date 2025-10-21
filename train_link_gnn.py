# train_link_gnn.py
# Minimal link predictor for HOAE node embeddings. No torch-sparse needed.

import argparse, math, os, random, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass

# ---------- utils ----------
def set_seed(s=42):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_pyg(path):
    # torch_geometric Data object saved at Data/processed/data.pt
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj

def load_hoae_nodes(path, n_expected=None):
    if Path(path).exists():
        x = torch.load(path, map_location="cpu", weights_only=False)
        if x.ndim != 2:
            raise ValueError(f"hoae_nodes.pt has shape {tuple(x.shape)}; expected [N, D]")
        if n_expected is not None and x.size(0) != n_expected:
            raise ValueError(f"N mismatch: data has {n_expected} nodes, hoae_nodes has {x.size(0)}")
        return x
    return None

def undirected_unique(edge_index, num_nodes):
    # Return edge list with u<v and without duplicates
    ei = edge_index.detach().cpu()
    E = []
    for u, v in ei.t().tolist():
        if u == v: 
            continue
        a, b = (u, v) if u < v else (v, u)
        E.append((a, b))
    E = sorted(set(E))
    u = torch.tensor([a for a, _ in E], dtype=torch.long)
    v = torch.tensor([b for _, b in E], dtype=torch.long)
    return u, v

def split_edges(u, v, train=0.7, val=0.15, seed=42):
    idx = torch.randperm(u.numel(), generator=torch.Generator().manual_seed(seed))
    n = u.numel()
    n_train = int(n * train)
    n_val = int(n * val)
    i_tr = idx[:n_train]
    i_va = idx[n_train:n_train+n_val]
    i_te = idx[n_train+n_val:]
    return (u[i_tr], v[i_tr]), (u[i_va], v[i_va]), (u[i_te], v[i_te])

def sample_neg_pairs(num_nodes, num_samples, forbid_set, seed):
    # forbid_set contains existing undirected edges (u<v)
    g = torch.Generator().manual_seed(seed)
    got = set()
    out_u, out_v = [], []
    while len(out_u) < num_samples:
        u = int(torch.randint(0, num_nodes, (1,), generator=g))
        v = int(torch.randint(0, num_nodes, (1,), generator=g))
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in forbid_set:
            continue
        if (a, b) in got:
            continue
        got.add((a, b))
        out_u.append(a); out_v.append(b)
    return torch.tensor(out_u), torch.tensor(out_v)

def compute_auc(scores, labels):
    # scores, labels: 1D tensors on CPU
    s = scores.detach().cpu().numpy()
    y = labels.detach().cpu().numpy().astype(float)
    order = s.argsort()
    s, y = s[order], y[order]
    # rank-based AUC
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    # Mann–Whitney U
    import numpy as np
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y)+1)
    sum_ranks_pos = ranks[y == 1].sum()
    U = sum_ranks_pos - pos*(pos+1)/2.0
    return float(U / (pos*neg))

# ---------- model ----------
class MLP(nn.Module):
    def __init__(self, dims, act=nn.ReLU, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1])]
            if i < len(dims)-2:
                layers += [act(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class LinkGNN(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128, use_edge_attr=False, edge_in_dim=0):
        super().__init__()
        self.use_edge_attr = use_edge_attr and edge_in_dim > 0
        self.node_enc = MLP([in_dim, hidden, out_dim])
        if self.use_edge_attr:
            self.edge_enc = MLP([edge_in_dim, hidden, out_dim])
            comb_in = out_dim*3  # [|hu-hv|, hu*hv, he]
        else:
            comb_in = out_dim*2  # [|hu-hv|, hu*hv]
        self.scorer = MLP([comb_in, hidden, 1], act=nn.ReLU, dropout=0.0)

    def encode_nodes(self, x):
        return self.node_enc(x)

    def pair_features(self, H, u, v, edge_attr=None):
        hu = H[u]; hv = H[v]
        feats = [torch.abs(hu - hv), hu * hv]
        if self.use_edge_attr:
            if edge_attr is None:
                he = torch.zeros_like(hu)
            else:
                he = self.edge_enc(edge_attr)
            feats.append(he)
        return torch.cat(feats, dim=-1)

    def forward(self, x, pos_u, pos_v, neg_u, neg_v, edge_attr_pos=None):
        H = self.encode_nodes(x)
        pos_feat = self.pair_features(H, pos_u, pos_v, edge_attr_pos)
        neg_feat = self.pair_features(H, neg_u, neg_v, None)
        pos_logit = self.scorer(pos_feat).squeeze(-1)
        neg_logit = self.scorer(neg_feat).squeeze(-1)
        return pos_logit, neg_logit, H

    @torch.no_grad()
    def score_pairs(self, x, u, v):
        H = self.encode_nodes(x)
        feats = self.pair_features(H, u, v, None)
        return self.scorer(feats).squeeze(-1)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=r".\Data\processed\data.pt")
    ap.add_argument("--hoae", type=str, default=r".\Data\processed\hoae_nodes.pt")
    ap.add_argument("--in_dim", type=int, default=128)
    ap.add_argument("--edge_in_dim", type=int, default=258)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save", type=str, default=r".\best_linkgnn.pt")
    ap.add_argument("--export_non_edges", type=str, default=r"")
    ap.add_argument("--export_topk", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_edge_attr", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    data = load_pyg(args.data)
    num_nodes = int(data.num_nodes)
    # feature matrix
    X_hoae = load_hoae_nodes(args.hoae, n_expected=num_nodes)
    if X_hoae is not None:
        x = X_hoae.float()
    else:
        x = data.x.float()
        if x is None:
            raise ValueError("No node features found and hoae_nodes.pt missing.")
    in_dim = x.size(1)

    # edges
    pos_u_all, pos_v_all = undirected_unique(data.edge_index, num_nodes)
    forbid = set(zip(pos_u_all.tolist(), pos_v_all.tolist()))

    (tr_u, tr_v), (va_u, va_v), (te_u, te_v) = split_edges(pos_u_all, pos_v_all, seed=args.seed)
    va_neg_u, va_neg_v = sample_neg_pairs(num_nodes, va_u.numel(), forbid, seed=args.seed+1)
    te_neg_u, te_neg_v = sample_neg_pairs(num_nodes, te_u.numel(), forbid, seed=args.seed+2)

    model = LinkGNN(
        in_dim=in_dim,
        hidden=256,
        out_dim=128,
        use_edge_attr=bool(args.use_edge_attr),
        edge_in_dim=args.edge_in_dim,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None

    # pre-encode edge_attr for train positives if used
    if bool(args.use_edge_attr) and getattr(data, "edge_attr", None) is not None:
        # Map (u,v) with u<v to edge_attr row by building a dict once
        ed = {}
        for (u, v), ea in zip(data.edge_index.t().tolist(), data.edge_attr):
            a, b = (u, v) if u < v else (v, u)
            ed[(a, b)] = ea
        def fetch_edge_attr(u, v):
            key = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
            return ed.get(key, torch.zeros(data.edge_attr.size(1)))
    else:
        fetch_edge_attr = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        # fresh negatives per epoch for train
        tr_neg_u, tr_neg_v = sample_neg_pairs(num_nodes, tr_u.numel(), forbid, seed=args.seed+100+epoch)

        if fetch_edge_attr is not None:
            ea_list = [fetch_edge_attr(a, b).unsqueeze(0) for a, b in zip(tr_u.tolist(), tr_v.tolist())]
            edge_attr_pos = torch.vstack(ea_list) if ea_list else None
        else:
            edge_attr_pos = None

        pos_logit, neg_logit, _ = model(x, tr_u, tr_v, tr_neg_u, tr_neg_v, edge_attr_pos=edge_attr_pos)
        y_pos = torch.ones_like(pos_logit)
        y_neg = torch.zeros_like(neg_logit)
        loss = bce(torch.cat([pos_logit, neg_logit]), torch.cat([y_pos, y_neg]))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # validation
        model.eval()
        with torch.no_grad():
            if fetch_edge_attr is not None:
                va_ea = torch.vstack([fetch_edge_attr(a, b) for a, b in zip(va_u.tolist(), va_v.tolist())])
            else:
                va_ea = None
            p_log, n_log, _ = model(x, va_u, va_v, va_neg_u, va_neg_v, edge_attr_pos=va_ea)
            scores = torch.cat([p_log, n_log]).cpu()
            labels = torch.cat([torch.ones_like(p_log), torch.zeros_like(n_log)]).cpu()
            val_auc = compute_auc(scores, labels)

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:03d} loss {loss.item():.4f} val_auc {val_auc:.3f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state,
                        "meta": {"in_dim": in_dim, "epoch": epoch, "val_auc": float(val_auc)}},
                       args.save)  # weights_only=False by default here

    print(f"saved best checkpoint → {args.save} (AUC={best_auc:.3f})")

    # optional export of all non-edges
    if args.export_non_edges:
        # enumerate all non-edges u<v
        all_u = []; all_v = []
        for u in range(num_nodes):
            for v in range(u+1, num_nodes):
                if (u, v) not in forbid:
                    all_u.append(u); all_v.append(v)
        all_u = torch.tensor(all_u, dtype=torch.long)
        all_v = torch.tensor(all_v, dtype=torch.long)

        # reload best state
        if best_state is None:
            ckpt = torch.load(args.save, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(best_state)

        model.eval()
        scores = []
        B = 4096
        with torch.no_grad():
            for i in range(0, all_u.numel(), B):
                ui = all_u[i:i+B]
                vi = all_v[i:i+B]
                s = model.score_pairs(x, ui, vi)
                scores.append(s.cpu())
        scores = torch.cat(scores).numpy()

        import pandas as pd
        df = pd.DataFrame({"u": all_u.numpy(), "v": all_v.numpy(), "score": scores})
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        out_csv = Path(args.export_non_edges)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        if args.export_topk and args.export_topk > 0:
            df.head(args.export_topk).to_csv(out_csv.with_name("predicted_top200.csv"), index=False)
        print(f"wrote {out_csv} and {out_csv.with_name('predicted_top200.csv')}")

if __name__ == "__main__":
    main()
