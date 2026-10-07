#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:13:30 2026

@author: jbrennan
"""

import argparse
from pathlib import Path
import tomllib
import torch
import torch.nn as nn

from deeponet_chem.trainer import Trainer
from deeponet_chem.train_setup import setup_data_loaders
from deeponet_chem.train_setup import setup_model_and_optimizer



def main(paths, model_cfg, train_cfg):

    # Set compute device
    device = torch.device(
        train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    train_loader, valid_loader = setup_data_loaders(paths, train_cfg)
    
    model, optimizer = setup_model_and_optimizer(model_cfg, train_cfg, device)

    # Initialize Trainer and Run
    trainer = Trainer(
        model        = model,
        optimizer    = optimizer,
        train_loader = train_loader,
        valid_loader = valid_loader,
        loss_fn      = nn.MSELoss(),
        device       = device,
    )

    out_dir = Path(paths.get("output_dir", "."))
    save_path = out_dir / paths.get("model_checkpoint", "best_model.pth")
    print(f"Starting training for {train_cfg['num_epochs']} epochs...")
    trainer.run(
        num_epochs = train_cfg["num_epochs"], 
        save_path  = str(save_path)
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeepONet model on tracer dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("train.toml"),
        help="Path to TOML configuration file (default: train.toml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Load TOML Configuration
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    main(cfg["paths"], cfg["model"], cfg["training"])