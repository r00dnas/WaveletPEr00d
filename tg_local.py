# pip install pandas torch torch_geometric
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

path = "edges.csv"                     # your CSV
df = pd.read_csv(path)                 # if no header: pd.read_csv(path, header=None, names=["src","dst"])

src_col, dst_col = df.columns[:2]      # first two columns are endpoints

# Remap arbitrary IDs → 0..N-1
nodes = pd.Index(pd.unique(df[[src_col, dst_col]].values.ravel()))
nid   = pd.Series(range(len(nodes)), index=nodes)

src = df[src_col].map(nid).to_numpy()
dst = df[dst_col].map(nid).to_numpy()
edge_index = torch.tensor([src, dst], dtype=torch.long)

# Optional: undirected graph
# edge_index = to_undirected(edge_index, num_nodes=len(nodes))

edge_attr = None
if "weight" in df.columns:
    edge_attr = torch.tensor(df["weight"].to_numpy(), dtype=torch.float).view(-1,1)

data = Data(edge_index=edge_index, num_nodes=len(nodes), edge_attr=edge_attr)
torch.save(data, "graph.pt")
