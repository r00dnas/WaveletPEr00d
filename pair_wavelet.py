import torch.nn.functional as F

def pair_wavelet(x, l2=True, center=True, use_dot=True):
    # feature norm
    if center:
        x = (x - x.mean(0)) / x.std(0).clamp_min(1e-6)
    if l2:
        x = F.normalize(x, dim=-1)  # ||x_i||2 = 1

    n, d = x.shape
    xi = x[:, None, :].expand(n, n, d)
    xj = x[None, :, :].expand(n, n, d)

    had   = xi * xj                  # ∈[-1,1]
    adiff = (xi - xj).abs()          # ∈[0,2]
    cos   = (had.sum(-1, keepdim=True))  # == cosine if L2-normalized
    parts = [had, adiff, cos]
    if use_dot:
        dot = had.sum(-1, keepdim=True) / d  # scale to ~[-1,1]
        parts.append(dot)
    T = torch.cat(parts, -1)
    return 0.5 * (T + T.transpose(0, 1))
