#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:36:47 2026

@author: jbrennan
"""

import tomllib
import argparse
from pathlib import Path

import torch
import optuna
import torch.nn as nn

from deeponet_chem.trainer import Trainer
from deeponet_chem.train_setup import setup_data_loaders
from deeponet_chem.train_setup import setup_model_and_optimizer


def sample_param(trial: optuna.Trial, param_name: str, config: dict):
    """Dynamically registers Optuna parameters based on dictionary specs from
    the config TOML file.
    """
    spec = config[param_name]
    param_type = spec["type"]

    if param_type == "float":
        return trial.suggest_float(
            param_name, spec["low"], spec["high"], log=spec.get("log", False)
        )
    
    elif param_type == "int":
        return trial.suggest_int(param_name, spec["low"], spec["high"])
    
    elif param_type == "categorical":
        return trial.suggest_categorical(param_name, spec["choices"])
    
    else:
        raise ValueError(f"Unsupported search parameter type: {param_type}")



def create_objective(config: dict, device):
    
    paths = config['paths']
    model_base = config['model']
    train_base = config['training']
    
    def objective(trial: optuna.Trial) -> float:
        search_cfg = config["search_space"]
        
        train_cfg = train_base | {
            'learning_rate' : sample_param(trial, "learning_rate", search_cfg),
            'weight_decay'  : sample_param(trial, "weight_decay",  search_cfg),
            'batch_size'    : sample_param(trial, "batch_size",    search_cfg)
        }
        
        model_cfg = model_base | {
            'latent_size'   : sample_param(trial, "latent_size",   search_cfg),
            'branch_size'   : sample_param(trial, "branch_size",   search_cfg),
            'trunk_size'    : sample_param(trial, "trunk_size",    search_cfg),
            'branch_layers' : sample_param(trial, "branch_layers", search_cfg),
            'trunk_layers'  : sample_param(trial, "trunk_layers",  search_cfg),
            'activation'    : sample_param(trial, "activation",    search_cfg),
        }
        
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

        # Pass trial to trainer so it can report epoch losses & prune early
        val_loss = trainer.train_with_optuna(trial)
        return val_loss

    return objective



def main(config):
    # Set compute device
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    
    optuna_cfg = config["optuna"]
    
    study = optuna.create_study(
        study_name = optuna_cfg.get("study_name", "deeponet_hpo"),
        direction  = optuna_cfg.get("direction", "minimize"),
        storage    = optuna_cfg.get("storage", None),
        pruner     = optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )

    study.optimize(
        create_objective(config, device), 
        n_trials = optuna_cfg.get("n_trials", 30),
    )

    print("\n--- Optimization Complete ---")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Loss:  {study.best_value:.6f}")
    print("Best Hyperparameters:")
    for key, val in study.best_params.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Optuna HPO for deeponet-chem."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/tune.toml"),
        help="Path to tuning TOML configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    main(config)