#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:31:42 2026

@author: jbrennan
"""

from pathlib import Path

import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import TracerDataset
from .deep_o_net import DeepONet


def setup_data_loaders(paths, train_cfg):
    """
    """
    data_dir = Path(paths.get("data_dir", "."))
    metadata_path = data_dir / paths["metadata_file"]

    train_dataset = TracerDataset(
        data_dir / paths["train_data"],
        metadata_path,
        "num_train_tracers",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size  = train_cfg["batch_size"],
        shuffle     = train_cfg.get("shuffle_train", True),
        num_workers = 2,
        pin_memory  = True
    )

    valid_dataset = TracerDataset(
        data_dir / paths["valid_data"],
        metadata_path,
        "num_valid_tracers",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size  = train_cfg["batch_size"],
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True
    )
    
    return train_loader, valid_loader


def setup_model_and_optimizer(model_cfg, train_cfg, device):
    """
    """
    model = DeepONet(**model_cfg).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr           = train_cfg["learning_rate"],
        weight_decay = train_cfg.get("weight_decay", 0.0),
    )
    
    return model, optimizer