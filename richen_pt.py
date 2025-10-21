# richen_pt.py
import argparse, torch

def build_dense_adj(edge_index, n):
    A = torch.zeros((n, n), dtype=torch.float32)
    src, dst = edge_index[0].long(), edge_index[1].long()
    A[src, dst] = 1.0
    A[dst, src] = 1.0
    A.fill_diagonal_(0.0)
    return A

def laplacian_eig(A):
    n = A.size(0)
    deg = A.sum(1)
    Dm12 = torch.pow(deg.clamp(min=1e-12), -0.5)
    L = torch.eye(n) - (Dm12[:, None] * A * Dm12[None, :])
    evals, evecs = torch.linalg.eigh(L)  # ascending
    return evals, evecs

def lap_pe(evals, evecs, k=16, eps=1e-6):
    idx = torch.arange(len(evals), device=evals.device)[evals > eps][:k]
    if idx.numel() == 0:
        return evecs[:, :min(k, evecs.size(1))]
    return evecs[:, idx]

def hks(evals, evecs, times=(0.1, 0.5, 1.0, 2.0, 5.0)):
    phi2 = evecs**2                   # N x n_eigs
    outs = []
    for t in times:
        w = torch.exp(-t * evals)     # n_eigs
        outs.append((phi2 * w[None, :]).sum(1))
    return torch.stack(outs, 1)       # N x T

def pagerank(A, alpha=0.85, iters=100, tol=1e-8):
    n = A.size(0)
    M = A + torch.eye(n)
    M = M / M.sum(1, keepdim=True).clamp(min=1e-12)
    x = torch.full((n,), 1.0/n)
    for _ in range(iters):
        x_new = alpha * (M.t() @ x) + (1 - alpha) * (1.0/n)
        if torch.norm(x_new - x, p=1) < tol:
            x = x_new; break
        x = x_new
    return x

def standardize(X):
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, unbiased=False, keepdim=True).clamp(min=1e-6)
    return (X - mu) / sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32)                 # LapPE dims
    ap.add_argument("--hks", type=str, default="0.1,0.5,1,2,5")  # comma list
    args = ap.parse_args()

    d = torch.load(args.inp, weights_only=False)
    n = int(d.num_nodes)
    A = build_dense_adj(d.edge_index, n)
    evals, evecs = laplacian_eig(A)

    pe = lap_pe(evals, evecs, k=args.k)
    hks_times = tuple(float(x) for x in args.hks.split(","))
    H = hks(evals, evecs, hks_times)
    deg = A.sum(1)
    logdeg = torch.log1p(deg)
    pr = pagerank(A)

    X = torch.cat([deg[:, None], logdeg[:, None], pr[:, None], pe, H], dim=1).float()
    X = standardize(X)
    d.x = X
    torch.save(d, args.out)
    print(f"x shape: {d.x.shape}  -> wrote {args.out}")

if __name__ == "__main__":
    main()
