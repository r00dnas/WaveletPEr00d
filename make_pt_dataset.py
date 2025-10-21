import argparse, json, pandas as pd, numpy as np, torch
from pathlib import Path
from torch_geometric.data import Data

def load_edges(p: Path):
    df = pd.read_csv(p)
    # pick columns
    if {"src","dst"}.issubset(df.columns):
        a, b = df["src"].to_numpy(), df["dst"].to_numpy()
    else:
        a, b = df.iloc[:,0].to_numpy(), df.iloc[:,1].to_numpy()
    # drop self-loops
    m = a != b
    a, b = a[m], b[m]
    # relabel to 0..N-1
    nodes = pd.Index(np.unique(np.concatenate([a,b]))).to_list()
    remap = {v:i for i,v in enumerate(nodes)}
    a = np.vectorize(remap.get)(a)
    b = np.vectorize(remap.get)(b)
    edge_index = torch.as_tensor(np.vstack([a,b]), dtype=torch.long)  # [2,E]
    n = len(nodes)
    return edge_index, n, remap

def load_features_csv(p: Path, n: int, remap: dict):
    df = pd.read_csv(p)
    # first column is node id; remaining are feature columns
    node_col = df.columns[0]
    feats = df.drop(columns=[node_col]).to_numpy(dtype=np.float32)
    x = np.zeros((n, feats.shape[1]), dtype=np.float32)
    # align rows via remap
    for node, row in zip(df[node_col].to_numpy(), feats):
        if node in remap:
            x[remap[node]] = row
    return torch.from_numpy(x)

def build_default_features(edge_index: torch.Tensor, n: int):
    # degree and constant bias -> shape [N,2]
    deg = torch.bincount(edge_index.view(-1), minlength=n)
    x = torch.stack([deg.float(), torch.ones(n)], dim=1)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default="Data/edges.csv")
    ap.add_argument("--features_csv", default=None,
                    help="optional CSV: first col node id, rest numeric features")
    ap.add_argument("--out", default="Data/dataset.pt")
    args = ap.parse_args()

    edges_p = Path(args.edges); out_p = Path(args.out)
    edge_index, n, remap = load_edges(edges_p)

    if args.features_csv and Path(args.features_csv).exists():
        x = load_features_csv(Path(args.features_csv), n, remap)
    else:
        x = build_default_features(edge_index, n)

    # WaveletPE-required fields; placeholder attributes
    edge_index_wavepe = edge_index.clone()
    edge_attr_wavepe  = torch.ones(edge_index_wavepe.size(1), 1, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_index_wavepe=edge_index_wavepe,
        edge_attr_wavepe=edge_attr_wavepe,
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save([data], out_p)
    print(f"saved {out_p} | N={x.shape[0]} F={x.shape[1]} E={edge_index.size(1)}")

if __name__ == "__main__":
    main()
