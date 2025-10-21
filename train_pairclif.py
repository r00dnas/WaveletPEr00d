# train_pairclf.py
import torch, torch.nn as nn
root = r"C:\Users\r00dn\Wavelet\WaveletPEr00d\Data\processed"
Xtr,ytr = torch.load(rf"{root}\pairs_train.pt", weights_only=False)
Xva,yva = torch.load(rf"{root}\pairs_val.pt",   weights_only=False)

model = nn.Sequential(
    nn.Linear(Xtr.size(1), 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 1)
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

for e in range(1, 201):
    model.train(); opt.zero_grad()
    loss = loss_fn(model(Xtr).squeeze(), ytr.squeeze())
    loss.backward(); opt.step()
    if e % 20 == 0:
        with torch.no_grad():
            pr = torch.sigmoid(model(Xva).squeeze())
            pred = (pr >= 0.5).float()
            acc = (pred == yva.squeeze()).float().mean().item()
            print(f"epoch {e:03d} loss {loss.detach().item():.4f} val_acc {acc:.3f}")

torch.save(model.state_dict(), rf"{root}\pairclf.pt")
print("saved pairclf.pt")
