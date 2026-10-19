#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:14:45 2026

@author: jbrennan
"""

import torch
import numpy as np

from torch import tensor
from torch.utils.data import Dataset


class ZipDataset(Dataset):
    """
    """
    def __init__(self, *datasets):
        self.datasets = datasets
        
        lengths = [len(d) for d in datasets]
        self.length = min(lengths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return tuple(dataset[idx] for dataset in self.datasets)



class OneStepTracerDataset(Dataset):
    """
    """
    def __init__(
            self, 
            dataset_path,
            metadata,
            num_tracers
        ):
        self.dataset_path = dataset_path
        self.metadata     = metadata
        self.num_tracers  = num_tracers
        
        self.data = np.load(dataset_path, mmap_mode="r")
        
        self.n_params        = self.metadata['n_params']
        self.rows_per_tracer = self.metadata['rows_per_tracer']
        self.num_hydro_steps = self.metadata['num_hydro_steps']
        self.chem_per_hydro  = self.metadata['chem_per_hydro']
        
        dt_min = self.metadata['bounds']['dt_min']
        dt_max = self.metadata['bounds']['dt_max']
        self.time_step = dt_min / dt_max
        self.dt = [
            tensor([i * self.time_step], dtype=torch.float32)
            for i in range(6)
        ]


    def __len__(self):
        return self.num_tracers * self.num_hydro_steps * self.chem_per_hydro


    def __getitem__(self, idx):
        hydro_step = idx // self.chem_per_hydro
        chem_step  = idx  % self.chem_per_hydro + 1
        
        block = np.asarray(
            self.data[hydro_step : hydro_step + self.chem_per_hydro + 1]
        )
        
        t = self.dt[chem_step]
        u = tensor(block[0, :-1], dtype=torch.float32)
        y = tensor(block[chem_step, self.n_params:-1], dtype=torch.float32)
        
        return u, t, y
    
    
    
class FiveStepRolloutDataset(Dataset):
    """
    """
    def __init__(
            self, 
            dataset_path,
            metadata,
            num_tracers
        ):
        self.dataset_path = dataset_path
        self.metadata     = metadata
        self.num_tracers  = num_tracers
        
        self.data = np.load(dataset_path, mmap_mode="r")
        
        self.n_params               = self.metadata['n_params']
        self.rows_per_tracer        = self.metadata['rows_per_tracer']
        self.chem_per_hydro         = self.metadata['chem_per_hydro']
        self.hydro_steps_per_tracer = self.metadata['num_hydro_steps']
        self.hydro_steps_per_sample = 5
        
        dt_min = self.metadata['bounds']['dt_min']
        dt_max = self.metadata['bounds']['dt_max']
        self.time_step = dt_min / dt_max
        self.dt = [
            tensor([i * self.time_step], dtype=torch.float32)
            for i in range(6)
        ]
        
        self.samples_per_tracer   = self.hydro_steps_per_tracer
        self.samples_per_tracer //= self.hydro_steps_per_sample
        self.num_samples = self.num_tracers * self.samples_per_tracer
        
        self.rows_per_sample  = self.hydro_steps_per_sample
        self.rows_per_sample *= self.metadata['chem_per_hydro']


    def __len__(self):
        return self.num_samples
    
    
    def __getitem__(self, sample_ind):
        start = self.rows_per_sample * sample_ind
        block = np.asarray(self.data[start : start + self.rows_per_sample + 1])
        
        u_init = tensor(block[0, :-1], dtype=torch.float32)
        t_step = self.dt[-1]
        
        params_seq = []
        target_seq = []
        
        for step in range(self.hydro_steps_per_sample):
            
            ind = (step + 1) * self.chem_per_hydro
            p_step = tensor(block[ind, :self.n_params], dtype=torch.float32)
            y_step = tensor(block[ind, self.n_params:-1], dtype=torch.float32)
            
            params_seq.append(p_step)
            target_seq.append(y_step)
            
        # Stack into tensors: shape becomes [num_steps, feature_dim]
        params_tensor = torch.stack(params_seq)
        target_tensor = torch.stack(target_seq)
        
        initial_state = (u_init, t_step)
        rollout_sequence = (params_tensor, target_tensor)
        
        return initial_state, rollout_sequence



def read_tracer_trajectory(
        path, 
        index, 
        n_species, 
        rows_per_tracer, 
        n_params=7
    ):
    mm = np.load(path, mmap_mode="r")
    start = index * rows_per_tracer
    block = np.asarray(mm[start:start + rows_per_tracer])
    
    if block.shape[0] == 0 or block[0, 2] == 0.0:     # unwritten tracer
        return np.empty(0), np.empty((n_species, 0)), np.empty((0, n_params))
    
    t = block[:, 1]
    y = block[:, 2 + n_params:2 + n_params + n_species].T
    return t, y, block[:, 2:2 + n_params]