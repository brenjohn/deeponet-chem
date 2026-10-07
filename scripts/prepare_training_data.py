#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 22:08:41 2026

@author: jbrennan
"""

import json
import tomllib
import numpy as np

from pathlib import Path


def compute_bounds_from_ids(
        data, 
        train_tracer_ids, 
        rows_per_tracer, 
        n_params, 
        n_species
    ):
    """Computes min and max bounds for parameters and species of tracers used
    in training data."""
    # Get mask for rows belonging to training tracers
    all_tracer_ids = np.repeat(
        np.arange(data.shape[0] // rows_per_tracer), rows_per_tracer
    )
    train_mask = np.isin(all_tracer_ids, train_tracer_ids)
    train_data = data[train_mask]

    t_all = train_data[:, 1]
    p_all = train_data[:, 2 : 2 + n_params]
    y_all = train_data[:, 2 + n_params : 2 + n_params + n_species]
    
    t_diffs = t_all[1:6] - t_all[0]
    dt_min = np.min(t_diffs[t_diffs > 0])
    dt_max = np.max(t_diffs)

    y_min = np.min(y_all, axis=0)
    y_max = np.max(y_all, axis=0)

    p_min = np.min(p_all, axis=0)
    p_max = np.max(p_all, axis=0)

    return {
        "dt_min": float(dt_min),
        "dt_max": float(dt_max),
        "y_min": y_min.tolist(),
        "y_max": y_max.tolist(),
        "p_min": p_min.tolist(),
        "p_max": p_max.tolist(),
    }


def normalize_dataset(tracer_data, n_params, n_species, bounds):
    """Normalizes physical parameters and species into [-1, 1] range in 
    log-scale."""
    dataset = tracer_data[:, 2 : 2 + n_params + n_species]
    dataset = np.log10(dataset)

    lower = np.concat([bounds['p_min'], bounds['y_min']])
    upper = np.concat([bounds['p_max'], bounds['y_max']])
    lower = np.log10(lower)
    upper = np.log10(upper)

    dataset = 2.0 * (dataset - lower) / (upper - lower + 1e-8) - 1.0
    # Append the raw time array as the last column
    dataset = np.hstack([dataset, tracer_data[:, 1, None]])
    return dataset


def density_stratified_split(
        tracer_data, 
        rows_per_tracer, 
        density_column=2, 
        num_bins=10, 
        train_ratio=0.8, 
        seed=42
    ):
    """Splits tracer IDs into train and validation sets using a density 
    stratification approach."""
    num_tracers = tracer_data.shape[0] // rows_per_tracer
    
    # Extract start and end row indices for each tracer trajectory
    starts = np.arange(0, tracer_data.shape[0], rows_per_tracer)
    ends   = starts + rows_per_tracer - 1
    
    tracer_ids      = np.arange(num_tracers)
    initial_density = tracer_data[starts, density_column]
    final_density   = tracer_data[ends, density_column]

    # Filter out invalid or non-positive density tracers
    has_valid_start     = initial_density > 0
    has_valid_end       = np.isfinite(final_density) & (final_density > 0)
    valid_mask          = has_valid_start & has_valid_end
    useable_ids         = tracer_ids[valid_mask]
    valid_final_density = final_density[valid_mask]

    # Partition log10 density into equal-width bins
    log_density = np.log10(valid_final_density)
    edges = np.linspace(log_density.min(), log_density.max(), num_bins + 1)
    labels = np.digitize(log_density, edges[1:-1], right=True)

    rng = np.random.default_rng(seed)
    train_ids = []
    valid_ids = []

    # Stratify split per bin
    for bin_idx in range(num_bins):
        candidates = useable_ids[labels == bin_idx]
        if len(candidates) == 0:
            continue
        
        candidates = rng.permutation(candidates)
        n_train = int(np.round(len(candidates) * train_ratio))
        
        train_ids.extend(candidates[:n_train])
        valid_ids.extend(candidates[n_train:])

    return (
        np.array(train_ids, dtype=np.int64), 
        np.array(valid_ids, dtype=np.int64)
    )


if __name__ == '__main__':
    
    config_path = Path("dataset_config.toml")
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
        
    paths = cfg["paths"]
    dims = cfg["data_dimensions"]
    split_cfg = cfg["split_parameters"]
    
    out_dir = Path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tracer_data = np.load(paths["raw_data"], mmap_mode="r")
    num_tracers = tracer_data.shape[0] // dims["rows_per_tracer"]
    num_hydro_steps = (dims["rows_per_tracer"] - 1) // dims["chem_per_hydro"]

    # Stratified split into train and validation IDs
    train_ids, valid_ids = density_stratified_split(
        tracer_data     = tracer_data,
        rows_per_tracer = dims["rows_per_tracer"],
        density_column  = dims["density_col_idx"],
        num_bins        = split_cfg["num_bins"],
        train_ratio     = split_cfg["train_ratio"],
        seed            = split_cfg["seed"]
    )

    # Compute bounds exclusively from the training tracers
    bounds = compute_bounds_from_ids(
        data             = tracer_data,
        train_tracer_ids = train_ids,
        rows_per_tracer  = dims["rows_per_tracer"],
        n_params         = dims["n_params"],
        n_species        = dims["n_species"],
    )

    # Save metadata including dataset split counts
    metadata = {
        "n_species": dims["n_species"],
        "n_params": dims["n_params"],
        "num_tracers": num_tracers,
        "num_train_tracers": len(train_ids),
        "num_valid_tracers": len(valid_ids),
        "rows_per_tracer": dims["rows_per_tracer"],
        "num_hydro_steps": num_hydro_steps,
        "chem_per_hydro": dims["chem_per_hydro"],
        "bounds": bounds,
    }
    with open(out_dir / paths["metadata_file"], "w") as file:
        json.dump(metadata, file, indent=2)

    # Normalize full dataset
    normalized_data = normalize_dataset(
        tracer_data, dims["n_params"], dims["n_species"], bounds
    )

    # Extract rows corresponding to train and validation sets
    all_tracer_ids = np.repeat(np.arange(num_tracers), dims["rows_per_tracer"])
    train_mask = np.isin(all_tracer_ids, train_ids)
    valid_mask = np.isin(all_tracer_ids, valid_ids)

    train_dataset = normalized_data[train_mask]
    valid_dataset = normalized_data[valid_mask]

    # Save split datasets
    np.save(out_dir / paths["train_output"], train_dataset.astype(np.float32))
    np.save(out_dir / paths["valid_output"], valid_dataset.astype(np.float32))
    
    print("Data successfully prepared:")
    print(f"Train shape: {train_dataset.shape} {len(train_ids)} tracers")
    print(f"Valid shape: {valid_dataset.shape} {len(valid_ids)} tracers")