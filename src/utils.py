import logging
from pathlib import Path

import torch


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


def save_checkpoint(model: torch.nn.Module, path: str | Path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str | Path, device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
