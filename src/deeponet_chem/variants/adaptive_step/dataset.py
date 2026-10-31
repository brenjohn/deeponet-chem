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