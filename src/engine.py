"""სწავლების/შეფასების ლოგიკა და მეტრიკები."""
import torch
import torch.nn.functional as F
import numpy as np


def train_one_epoch(model, loader, crit, opt, device):
    model.train()
    tot_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        tot_loss += loss.item() * len(y)
        correct  += (out.argmax(1) == y).sum().item()
        total    += len(y)
    return tot_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    tot_loss = correct = total = 0
    all_p, all_y = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        tot_loss += crit(out, y).item() * len(y)
        p = out.argmax(1)
        correct += (p == y).sum().item()
        total   += len(y)
        all_p += p.cpu().tolist(); all_y += y.cpu().tolist()
    return tot_loss / total, correct / total, np.array(all_p), np.array(all_y)


def per_class_recall(preds, labels, n_classes=7):
    rec = {}
    for c in range(n_classes):
        mask = labels == c
        rec[c] = float((preds[mask] == c).mean()) if mask.sum() > 0 else 0.0
    return rec


def macro_f1(preds, labels, n_classes=7):
    f1s = []
    for c in range(n_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))
