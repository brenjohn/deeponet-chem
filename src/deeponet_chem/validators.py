#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:16:36 2026

@author: jbrennan
"""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from .dataset import OneStepTracerDataset


class OneStepValidator:
    """
    """
    
    def __init__(self, valid_data_path, metadata):
        self.valid_data_path = valid_data_path
        self.metadata = metadata
        
        self.loss_fn = nn.MSELoss()

        valid_dataset = OneStepTracerDataset(
            valid_data_path,
            metadata,
            metadata["num_valid_tracers"],
        )
        self.data_loader = DataLoader(
            valid_dataset,
            batch_size  = 256,
            shuffle     = False,
        )
        
        
    def validate(self, model, device):
        """
        """
        num_batches = len(self.data_loader)
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                u, t, y = [item.to(device) for item in batch]
                
                y_pred = model(u, t)
                total_loss += self.loss_fn(y_pred, y).item()
                
                if (batch_idx % 350 == 0):
                    message = f'\r[Valid batch {batch_idx}/{num_batches}]'
                    message += f'[One step = {total_loss/(batch_idx+1):0.4e}]'
                    print(message, end='')
                
            print('\r', end='')
        
        one_step_loss  = total_loss / num_batches
        valid_metadata = {'one_step_loss' : one_step_loss}
        valid_message  = f"[One Step Loss: {one_step_loss:.6e}]"
        return valid_metadata, valid_message
    
    
VALIDATORS = {
    'OneStep' : OneStepValidator
}