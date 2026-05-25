import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SkinLesionDataset
from src.transforms import train_transforms, val_transforms
from src.utils import save_checkpoint, get_logger
from model.classifier import build_model


def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)
    d = raw["data"] | raw["training"]
    d["pretrained"] = raw["model"]["pretrained"]
    return SimpleNamespace(**d)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss, correct = 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    total_loss, correct = 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [val]  ", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader.dataset)
    return total_loss / n, correct / n


def main(cfg):
    logger = get_logger("train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_ds = SkinLesionDataset(cfg.train_csv, cfg.data_root, transform=train_transforms)
    val_ds   = SkinLesionDataset(cfg.val_csv,   cfg.data_root, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.workers)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)

    model = build_model(num_classes=cfg.num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_acc = 0.0
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_pbar = tqdm(range(1, cfg.epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg.epochs)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device, epoch, cfg.epochs)
        epoch_pbar.set_postfix(train_acc=f"{train_acc:.4f}", val_acc=f"{val_acc:.4f}")
        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, output_dir / "best.pth")
            logger.info(f"  → saved best model (val_acc={best_val_acc:.4f})")

    save_checkpoint(model, output_dir / "last.pth")
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml", help="Path to YAML config")
    # CLI overrides
    parser.add_argument("--train_csv",   default=None)
    parser.add_argument("--val_csv",     default=None)
    parser.add_argument("--output_dir",  default=None)
    parser.add_argument("--epochs",      type=int, default=None)
    parser.add_argument("--batch_size",  type=int, default=None)
    parser.add_argument("--lr",          type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Apply any CLI overrides
    for key in ("train_csv", "val_csv", "output_dir", "epochs", "batch_size", "lr"):
        val = getattr(args, key)
        if val is not None:
            setattr(cfg, key, val)

    main(cfg)
