#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:14:45 2026

@author: jbrennan
"""

import json
import torch
import numpy as np

from torch.utils.data import Dataset


class OneStepTracerDataset(Dataset):
    """
    """
    def __init__(
            self, 
            dataset_path,
            metadata_path,
            num_tracers_key
        ):
        self.dataset_path    = dataset_path
        self.metadata_path   = metadata_path
        self.num_tracers_key = num_tracers_key
        
        self.data = np.load(dataset_path, mmap_mode="r")
        with open(metadata_path, 'r') as file:
            self.metadata = json.load(file)
        
        self.n_params        = self.metadata['n_params']
        self.num_tracers     = self.metadata[num_tracers_key]
        self.rows_per_tracer = self.metadata['rows_per_tracer']
        self.num_hydro_steps = self.metadata['num_hydro_steps']
        
        dt_min = self.metadata['bounds']['dt_min']
        dt_max = self.metadata['bounds']['dt_max']
        self.time_step = dt_min / dt_max
        self.dt = [
            torch.tensor([i * self.time_step], dtype=torch.float32)
            for i in range(6)
        ]


    def __len__(self):
        return self.num_tracers * self.num_hydro_steps


    def __getitem__(self, idx):
        tracer = self.rows_per_tracer * idx // self.num_hydro_steps
        block = np.asarray(self.data[tracer : tracer + self.rows_per_tracer])
        t_ind = np.random.randint(1, 6)
        
        dt = self.dt[t_ind]
        u = torch.tensor(block[0, :-1], dtype=torch.float32)
        y = torch.tensor(block[t_ind, self.n_params:-1], dtype=torch.float32)
        
        return u, dt, y
    
    
    
class FiveStepRolloutDataset(Dataset):
    """
    """
    def __init__(
            self, 
            dataset_path,
            metadata_path,
            num_tracers_key
        ):
        self.dataset_path    = dataset_path
        self.metadata_path   = metadata_path
        self.num_tracers_key = num_tracers_key
        
        self.data = np.load(dataset_path, mmap_mode="r")
        with open(metadata_path, 'r') as file:
            self.metadata = json.load(file)
        
        self.n_params               = self.metadata['n_params']
        self.num_tracers            = self.metadata[num_tracers_key]
        self.rows_per_tracer        = self.metadata['rows_per_tracer']
        self.hydro_steps_per_tracer = self.metadata['num_hydro_steps']
        self.hydro_steps_per_sample = 5
        
        dt_min = self.metadata['bounds']['dt_min']
        dt_max = self.metadata['bounds']['dt_max']
        self.time_step = dt_min / dt_max
        self.dt = [
            torch.tensor([i * self.time_step], dtype=torch.float32)
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
        # Get the tracer block
        start  = self.rows_per_sample * sample_ind
        block = np.asarray(self.data[start : start + self.rows_per_sample + 1])
        
        # Get the target states
        y = torch.tensor(block[1:, self.n_params:-1], dtype=torch.float32)
        
        # Get the initial states
        chem_per_hydro = self.metadata['chem_per_hydro']
        u = torch.tensor(block[:-1:chem_per_hydro, :-1], dtype=torch.float32)
        u = u.repeat_interleave(chem_per_hydro, 0)
        
        # Get the time steps
        dt = torch.stack(self.dt[1:])
        dt = dt.repeat(chem_per_hydro, 1)
        
        return u, dt, y



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