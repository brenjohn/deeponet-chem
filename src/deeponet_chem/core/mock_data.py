#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:31:24 2026

@author: jbrennan
"""

import json
import numpy as np

from pathlib import Path


# TODO: use the prepare dataset pipeline here
def generate_mock_dataset(
        data_path:       Path,
        num_tracers:     int, 
        num_hydro_steps: int,
        chem_per_hydro:  int,
        n_params:        int, 
        n_species:       int
    ):
    """Generates mock numpy files and metadata for testing datasets.
    """
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Create dummy tensors for branch input (u), trunk input (t), and target 
    # outputs (y)
    input_size  = n_params + n_species
    rows_per_tracer = num_hydro_steps * chem_per_hydro + 1
    num_samples = num_tracers * rows_per_tracer
    
    mock_dataset = np.random.randn(
        num_samples, input_size + 1
    ).astype(np.float32)
    
    # Save split datasets
    np.save(data_path / "train_dataset.npy", mock_dataset)
    np.save(data_path / "valid_dataset.npy", mock_dataset)

    # Write metadata json
    bounds = {
        "dt_min":  0.2,
        "dt_max":  1.0,
        "y_min": [-1.0] * n_species,
        "y_max":  [1.0] * n_species,
        "p_min": [-1.0] * n_params,
        "p_max":  [1.0] * n_params,
    }

    # Save metadata including dataset split counts
    metadata = {
        "n_species":         n_species,
        "n_params":          n_params,
        "num_tracers":       num_tracers,
        "num_train_tracers": num_tracers,
        "num_valid_tracers": num_tracers,
        "rows_per_tracer":   rows_per_tracer,
        "num_hydro_steps":   num_hydro_steps,
        "chem_per_hydro":    chem_per_hydro,
        "bounds": bounds,
    }
    
    with open(data_path / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)