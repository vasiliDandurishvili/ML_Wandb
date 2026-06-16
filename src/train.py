"""Wandb run ლოგიკა — თითო არქიტექტურა group, თითო კონფიგი ცალკე run.

შენიშვნა: სიჩქარისთვის რეალური ექსპერიმენტები გაშვებულია in-memory
GPU-augmentation ვერსიით (იხ. Colab notebook / configs/experiments.py).
ეს ფაილი აღწერს DataLoader-ზე დაფუძნებულ კანონიკურ train loop-ს.
"""
import torch
import wandb
from src.data import get_loaders, class_weights, EMOTIONS
from src.models import MODELS
from src.engine import train_one_epoch, evaluate, per_class_recall, macro_f1


def make_run_name(cfg):
    parts = [cfg['arch'], f"lr{cfg['lr']}", f"wd{cfg['wd']}",
             "aug" if cfg['augment'] else "noaug",
             "cw" if cfg['class_weights'] else "nocw"]
    if cfg.get('tag'):
        parts.append(cfg['tag'])
    return "_".join(parts)


def run_experiment(df, config, project="fer2013-emotion"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.init(project=project, group=config['arch'],
               name=make_run_name(config), config=config, reinit=True)
    cfg = wandb.config

    tr, va, te = get_loaders(df, batch_size=cfg.batch_size, augment=cfg.augment)
    mk = {'p': config['dropout']} if config.get('dropout') is not None else {}
    model = MODELS[cfg.arch](**mk).to(device)
    wandb.config.update({'n_params': sum(p.numel() for p in model.parameters())})

    cw = class_weights(df).to(device) if cfg.class_weights else None
    crit = torch.nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    patience = config.get('patience')

    best = 0.0; no_imp = 0
    for ep in range(cfg.epochs):
        tr_loss, tr_acc = train_one_epoch(model, tr, crit, opt, device)
        val_loss, val_acc, vp, vy = evaluate(model, va, crit, device)
        sched.step()
        log = {"epoch": ep, "train/loss": tr_loss, "train/acc": tr_acc,
               "val/loss": val_loss, "val/acc": val_acc,
               "val/f1_macro": macro_f1(vp, vy), "train_val_gap": tr_acc - val_acc,
               "lr": sched.get_last_lr()[0]}
        for c, r in per_class_recall(vp, vy).items():
            log[f"recall/{EMOTIONS[c]}"] = r
        wandb.log(log)
        if val_acc > best:
            best = val_acc; no_imp = 0
            torch.save(model.state_dict(), f"{make_run_name(config)}_best.pt")
        else:
            no_imp += 1
            if patience and no_imp >= patience:
                break

    _, test_acc, _, _ = evaluate(model, te, crit, device)
    wandb.summary["best_val/acc"] = best
    wandb.summary["test/acc"] = test_acc
    wandb.finish()
    return best, test_acc
