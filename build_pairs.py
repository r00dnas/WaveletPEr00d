# build_pairs.py
import torch
from torch_geometric.transforms import RandomLinkSplit
from pt_dataset import PTDataset
from pair_repr import pair_features, get_pos_neg

root = r"C:\Users\r00dn\Wavelet\WaveletPEr00d\Data"
data = PTDataset(root)[0]                                # uses data.x (now 64-d)
split = RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True, add_negative_train_samples=True)
tr, va, te = split(data)

def pack(d):
    pos, neg = get_pos_neg(d)
    Xp = pair_features(d.x, pos, kinds=("hadamard","absdiff","dot","cos"))
    yp = torch.ones(Xp.size(0), 1)
    Xn = pair_features(d.x, neg, kinds=("hadamard","absdiff","dot","cos"))
    yn = torch.zeros(Xn.size(0), 1)
    X = torch.cat([Xp, Xn], 0)
    y = torch.cat([yp, yn], 0)
    return X, y

torch.save(pack(tr), rf"{root}\processed\pairs_train.pt")
torch.save(pack(va), rf"{root}\processed\pairs_val.pt")
torch.save(pack(te), rf"{root}\processed\pairs_test.pt")
print("wrote pairs_* to Data\\processed")
