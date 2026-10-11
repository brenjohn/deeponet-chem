#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:31:42 2026

@author: jbrennan
"""

import json
from pathlib import Path

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .trainer import Trainer
from .validators import VALIDATORS
from .dataset import OneStepTracerDataset
from .deep_o_net import DeepONet


def setup_train_loader(paths, train_cfg):
    """
    """
    data_dir = Path(paths.get("data_dir", "."))
    metadata_path = data_dir / paths["metadata_file"]
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    train_dataset = OneStepTracerDataset(
        data_dir / paths["train_data"],
        metadata,
        metadata["num_train_tracers"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size  = train_cfg["batch_size"],
        shuffle     = train_cfg.get("shuffle_train", True),
    )
    return train_loader


def setup_validators(paths, train_cfg):
    validators = []
    
    data_dir = Path(paths.get("data_dir", "."))
    valid_data = data_dir / paths["train_data"]
    metadata_path = data_dir / paths["metadata_file"]
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)
    
    for name in train_cfg['validators']:
        validators.append(
            VALIDATORS[name](valid_data, metadata)
        )
    
    return validators


def setup_model_and_optimizer(model_cfg, train_cfg, device):
    """
    """
    model = DeepONet(**model_cfg).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr           = train_cfg["learning_rate"],
        weight_decay = train_cfg.get("weight_decay", 0.0),
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-8
    )
    
    return model, optimizer, scheduler


def start_training_run(paths, model_cfg, train_cfg):
    """
    """
    torch.set_flush_denormal(True)

    # Set compute device
    device = torch.device(
        train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    train_loader = setup_train_loader(paths, train_cfg)
    validators = setup_validators(paths, train_cfg)
    
    model, optimizer, scheduler = setup_model_and_optimizer(
        model_cfg, train_cfg, device
    )

    # Initialize Trainer and Run
    trainer = Trainer(
        model        = model,
        optimizer    = optimizer,
        scheduler    = scheduler,
        train_loader = train_loader,
        validators   = validators,
        primary_loss = train_cfg['primary_loss'],
        device       = device,
    )

    out_dir = Path(paths.get("output_dir", "."))
    save_path = out_dir / paths.get("model_checkpoint", "best_model.pth")
    print(f"Starting training for {train_cfg['num_epochs']} epochs...")
    trainer.run(
        num_epochs = train_cfg["num_epochs"], 
        save_path  = str(save_path)
    )