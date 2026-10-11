#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:13:30 2026

@author: jbrennan
"""

import tomllib
import argparse

from pathlib import Path

from deeponet_chem.train_setup import start_training_run


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeepONet model on tracer dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./train_config.toml"),
        help="Path to TOML configuration file (default: train.toml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Load TOML Configuration
    args = parse_args()
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    start_training_run(cfg["paths"], cfg["model"], cfg["training"])