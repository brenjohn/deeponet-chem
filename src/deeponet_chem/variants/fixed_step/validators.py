#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:16:36 2026

@author: jbrennan
"""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from .dataset import RolloutDataset



class RolloutValidator:
    """
    """
    
    def __init__(self, valid_data_path, metadata):
        self.valid_data_path = valid_data_path
        self.metadata = metadata

        valid_dataset = RolloutDataset(
            dataset_path  = valid_data_path,
            metadata      = metadata,
            rollout_steps = metadata['num_hydro_steps']
        )
        
        self.data_loader = DataLoader(
            valid_dataset,
            batch_size = 256,
            shuffle    = False,
        )
        
        self.num_batches = len(self.data_loader)
    
    
    def validate(self, model, device):
        """
        """
        loss_monitor = LossMonitor(self.num_batches)
        
        with torch.no_grad():
            for batch in self.data_loader:
                initial_state, rollout_sequence = batch
                u = initial_state[0].to(device)
                p = initial_state[1].to(device)
                
                batch_size = u.shape[0]
                loss_monitor.batch_starting(batch_size)
                
                params_tensor, target_tensor = rollout_sequence
                
                num_steps = params_tensor.shape[1]
                for step in range(num_steps):
                    y_pred = model(u, p)
                    y_targ = target_tensor[:, step, :].to(device)
                    
                    loss_monitor.record(y_pred, y_targ)
                    
                    p = params_tensor[:, step, :].to(device)
                    u = y_pred
                    
                loss_monitor.batch_ending()
        
        return loss_monitor.report()

    
    
class LossMonitor:
    
    def __init__(self, num_batches):
        self.num_batches  = num_batches
        self.num_rollouts = 0
        self.curr_batch   = 0
        
        self.loss_fn = nn.MSELoss()
        self.average_l2 = 0.0
        self.l_inf = 0.0
    
    
    def batch_starting(self, batch_size):
        self.num_rollouts += batch_size
        self.running_l2 = 0.0
        self.num_steps = 0
    
    
    def record(self, y_pred, y_targ):
        self.num_steps += 1
        self.running_l2 += self.loss_fn(y_pred, y_targ).item()
        
        l_inf = torch.max(torch.abs(y_pred - y_targ)).item()
        if l_inf > self.l_inf:
            self.l_inf = l_inf
    
    
    def batch_ending(self):
        self.average_l2 += self.running_l2 / self.num_steps
        self.curr_batch += 1
        
        curr_batch = self.curr_batch
        if (curr_batch % 2 == 0):
            message = f'\r[Valid batch {self.curr_batch}/{self.num_batches}]'
            message += f'[L2 = {self.average_l2/(curr_batch):0.4e}]'
            message += f'[L_inf = {self.l_inf:0.4e}]'
            print(message, end='')
    
    
    def report(self):
        print('\r', end='')
        
        self.average_l2 /= self.num_rollouts
        valid_message  = f"[Rollout Loss: {self.average_l2:.6e}]"
        valid_metadata = {
            'rollout_average_loss' : self.average_l2,
            'rollout_max_loss' : self.l_inf,
        }
        return valid_metadata, valid_message



VALIDATORS = {
    'Rollout' : RolloutValidator
}