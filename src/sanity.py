import torch
import torch.nn.functional as F

def check_init_loss(model, loader, device):
    model.eval()
    x, y = next(iter(loader)); x, y = x.to(device), y.to(device)
    with torch.no_grad():
        loss = F.cross_entropy(model(x), y)
    print(f"init loss = {loss.item():.3f}   (expected ≈ 1.946)")
    return loss.item()

def overfit_one_batch(model, loader, device, steps=200, lr=1e-3):
    model.train()
    x, y = next(iter(loader)); x, y = x.to(device), y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for s in range(steps):
        opt.zero_grad(); loss = F.cross_entropy(model(x), y); loss.backward(); opt.step()
    acc = (model(x).argmax(1) == y).float().mean().item()
    print(f"final: loss={loss.item():.4f}  acc={acc:.3f}  (acc უნდა იყოს ≈ 1.0)")
    return acc
