# augment_graph.py
from pathlib import Path
import torch, csv

DATA = r".\Data\processed\data.pt"
TOPK = r".\Data\processed\predicted_top200.csv"
OUT  = r".\Data\processed\data_aug.pt"

data = torch.load(DATA, map_location="cpu", weights_only=False)
edge_index = data.edge_index.clone()
old_E = edge_index.size(1)  # columns before augmentation

# collect new edges (undirected → add both directions)
E = set(map(tuple, edge_index.t().tolist()))
add = []
with open(TOPK, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        u = int(row["u"]); v = int(row["v"])
        if u == v: 
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in E or (b, a) in E:
            continue
        add += [(a, b), (b, a)]
        E.add((a, b)); E.add((b, a))

if add:
    add_t = torch.tensor(add, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, add_t], dim=1)

data.edge_index = edge_index

# pad edge_attr for new edges with zeros if present
if getattr(data, "edge_attr", None) is not None:
    m, d = data.edge_attr.size()
    need = edge_index.size(1) - m
    if need > 0:
        pad = torch.zeros((need, d), dtype=data.edge_attr.dtype)
        data.edge_attr = torch.cat([data.edge_attr, pad], dim=0)

# mark which columns are augmented
aug_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
aug_mask[old_E:] = True
data.aug_mask = aug_mask

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
torch.save(data, OUT)
print(f"augmented edges (directed): +{edge_index.size(1)-old_E}")
print(f"wrote {OUT}; base_E={old_E}, aug_E={int(aug_mask.sum())}")
