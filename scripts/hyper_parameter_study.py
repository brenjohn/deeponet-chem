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

torch.set_flush_denormal(True)

from deeponet_chem.trainer import Trainer
from deeponet_chem.train_setup import setup_train_loader
from deeponet_chem.train_setup import setup_validators
from deeponet_chem.train_setup import setup_model
from deeponet_chem.train_setup import setup_optimizer


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



# TODO: Reduce code duplication with train_setup.start_training_run
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
        
        # train_loader = setup_train_loader(paths, train_cfg)
        # validators = setup_validators(paths, train_cfg)
        # model, optimizer, scheduler = setup_model_and_optimizer(
        #     model_cfg, train_cfg, device
        # )
        
        model                = setup_model(model_cfg, train_cfg, device)
        train_loader         = setup_train_loader(paths, train_cfg)
        validators           = setup_validators(paths, train_cfg)
        optimizer, scheduler = setup_optimizer(model, train_cfg, device)

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

        # Pass trial to trainer so it can report epoch losses & prune early
        val_loss = trainer.train_with_optuna(trial, train_base['num_epochs'])
        return val_loss

    return objective



def main(config):
    # Set compute device
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    
    optuna_cfg = config["optuna"]
    
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            optuna_cfg.get("storage", "./optuna_journal.log")
        ),
    )
    
    study = optuna.create_study(
        study_name = optuna_cfg.get("study_name", "deeponet_hpo"),
        direction  = optuna_cfg.get("direction", "minimize"),
        storage    = storage,
        pruner     = optuna.pruners.MedianPruner(),
        load_if_exists = True,
    )
    
    # try:
    #     study = optuna.create_study(
    #         study_name = optuna_cfg.get("study_name", "deeponet_hpo"),
    #         direction  = optuna_cfg.get("direction", "minimize"),
    #         storage    = storage,
    #         pruner     = optuna.pruners.MedianPruner(),
    #         load_if_exists = True,
    #     )
    # except Exception:
    #     study = optuna.load_study(
    #         study_name = optuna_cfg.get("study_name", "deeponet_hpo"),
    #         storage    = storage,
    #     )

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
        default=Path("./tune_config.toml"),
        help="Path to tuning TOML configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    main(config)