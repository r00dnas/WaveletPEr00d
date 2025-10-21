# pair_repr.py
import torch

def pair_features(z: torch.Tensor, pairs: torch.Tensor, kinds=("hadamard","absdiff","l1","l2","dot","cos","concat")):
    u, v = pairs[0].long(), pairs[1].long()
    zu, zv = z[u], z[v]
    feats = []
    for k in kinds:
        if k == "hadamard": feats.append(zu * zv)
        elif k == "absdiff": feats.append((zu - zv).abs())
        elif k == "l1": feats.append((zu - zv).abs().sum(-1, keepdim=True))
        elif k == "l2": feats.append(((zu - zv)**2).sum(-1, keepdim=True).sqrt())
        elif k == "dot": feats.append((zu * zv).sum(-1, keepdim=True))
        elif k == "cos":
            eps=1e-9
            nz = zu.norm(dim=-1, keepdim=True).clamp_min(eps) * zv.norm(dim=-1, keepdim=True).clamp_min(eps)
            feats.append((zu*zv).sum(-1, keepdim=True)/nz)
        elif k == "sum": feats.append(0.5*(zu+zv))
        elif k == "diff": feats.append(zu - zv)
        elif k == "concat": feats.append(torch.cat([zu, zv], dim=-1))
        else: raise ValueError(k)
    return torch.cat(feats, dim=-1)

def get_pos_neg(d):
    if hasattr(d, "edge_label") and hasattr(d, "edge_label_index"):
        pos = d.edge_label_index[:, d.edge_label == 1]
        neg = d.edge_label_index[:, d.edge_label == 0]
        return pos, neg
    return d.pos_edge_label_index, getattr(d, "neg_edge_label_index", None)

def edge_attr_from_x(edge_index, x, kinds=("hadamard","absdiff")):
    return pair_features(x, edge_index, kinds)
