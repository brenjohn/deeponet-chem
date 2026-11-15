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
    

class RolloutDataset(Dataset):
    """
    """
    def __init__(
            self, 
            dataset_path,
            metadata,
            rollout_steps
        ):
        self.dataset_path  = dataset_path
        self.metadata      = metadata
        self.rollout_steps = rollout_steps
        self.data          = np.load(dataset_path, mmap_mode="r")
        
        self.n_params               = self.metadata['n_params']
        self.rows_per_tracer        = self.metadata['rows_per_tracer']
        self.chem_per_hydro         = self.metadata['chem_per_hydro']
        self.hydro_steps_per_tracer = self.metadata['num_hydro_steps']
        
        self.samples_per_tracer  = self.hydro_steps_per_tracer
        self.samples_per_tracer -= self.rollout_steps - 1
        self.rows_per_sample     = self.rollout_steps
        self.rows_per_sample    *= self.metadata['chem_per_hydro']
        
        self.num_tracers = self.data.shape[0] // self.rows_per_tracer
        self.num_samples = self.num_tracers * self.samples_per_tracer


    def __len__(self):
        return self.num_samples
    
    
    def __getitem__(self, sample_ind):
        tracer_ind   = sample_ind // self.samples_per_tracer
        tracer_start = tracer_ind * self.rows_per_tracer
        
        sample_ind   = sample_ind % self.samples_per_tracer
        sample_start = tracer_start + sample_ind * self.chem_per_hydro
        
        block = np.asarray(
            self.data[sample_start : sample_start + self.rows_per_sample + 1]
        )
        
        y_init = tensor(block[0, self.n_params:-1], dtype=torch.float32)
        p_init = tensor(block[0, :self.n_params], dtype=torch.float32)
        
        params_seq, target_seq = [], []
        for step in range(self.rollout_steps):
            
            ind = (step + 1) * self.chem_per_hydro
            p_step = tensor(block[ind, :self.n_params], dtype=torch.float32)
            y_step = tensor(block[ind, self.n_params:-1], dtype=torch.float32)
            
            params_seq.append(p_step)
            target_seq.append(y_step)
            
        # Stack into tensors: shape becomes [num_steps, feature_dim]
        params_tensor = torch.stack(params_seq)
        target_tensor = torch.stack(target_seq)
        
        initial_state = (y_init, p_init)
        rollout_sequence = (params_tensor, target_tensor)
        
        return initial_state, rollout_sequence
