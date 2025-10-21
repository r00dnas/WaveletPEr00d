import torch
d = torch.load(r".\Data\processed\data.pt", weights_only=False)
d.y = torch.zeros(d.num_nodes, dtype=torch.long)  # or your real labels
torch.save(d, r".\Data\processed\data.pt")
