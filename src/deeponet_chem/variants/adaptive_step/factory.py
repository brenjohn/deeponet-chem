#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 13:36:00 2026

@author: jbrennan
"""

from .dataset import OneStepTracerDataset
from .validators import VALIDATORS
from .loss import OneStepLoss


class OneStepFactory:
    
    def __init__(self):
        pass
    
    
    def get_dataset(self, data_path, metadata, train_cfg):
        num_tracers = metadata['num_train_tracers']
        return OneStepTracerDataset(data_path, metadata, num_tracers)
    
    
    def get_validator(self, name, valid_data, metadata):
        return VALIDATORS[name](valid_data, metadata)
    
    
    def get_loss_function(self):
        return OneStepLoss()