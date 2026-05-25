import argparse

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import SkinLesionDataset
from src.transforms import val_transforms
from src.utils import get_logger
from model.classifier import build_model


def main(cfg):
    logger = get_logger("evaluate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SkinLesionDataset(cfg.csv, cfg.data_root, transform=val_transforms)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)

    model = build_model(num_classes=cfg.num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(cfg.checkpoint, map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    logger.info("\n" + classification_report(all_labels, all_preds))
    logger.info("Confusion matrix:\n" + str(confusion_matrix(all_labels, all_preds)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",          required=True)
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--data_root",    default="")
    parser.add_argument("--num_classes",  type=int, default=7)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--workers",      type=int, default=4)
    main(parser.parse_args())
