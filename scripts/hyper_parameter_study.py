#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:36:47 2026

@author: jbrennan
"""

import tomllib
import argparse
from pathlib import Path

from deeponet_chem import start_hpo_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Optuna HPO for deeponet-chem."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("../example/tune_config.toml"),
        help="Path to tuning TOML configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    start_hpo_study(config)