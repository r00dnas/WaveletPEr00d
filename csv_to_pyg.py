import pandas as pd, torch
from torch_geometric.data import Data

df = pd.read_csv("edges.csv", header=None, names=["src","dst"])  # change if your CSV already has a header

nodes = pd.Index(pd.unique(df[["src","dst"]].values.ravel()))
nid   = pd.Series(range(len(nodes)), index=nodes)

src = df["src"].map(nid).to_numpy()
dst = df["dst"].map(nid).to_numpy()
edge_index = torch.tensor([src, dst], dtype=torch.long)

data = Data(edge_index=edge_index, num_nodes=len(nodes))
torch.save(data, "graph.pt")
print(f"nodes={data.num_nodes} edges={data.num_edges}")
