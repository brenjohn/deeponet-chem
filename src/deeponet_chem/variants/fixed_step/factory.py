#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 13:12:03 2026

@author: jbrennan
"""

from .dataset import RolloutDataset
from .validators import VALIDATORS


class FixedStepFactory:
    
    def __init__(self):
        pass
    
    
    def get_dataset(self, data_path, metadata, train_cfg):
        rollout_steps = train_cfg.get('rollout_steps', 5)
        return RolloutDataset(data_path, metadata, rollout_steps)
    
    
    def get_validator(self, name, valid_data, metadata):
        return VALIDATORS[name](valid_data, metadata)