#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 02:54:36 2026

@author: john
"""

import torch.nn as nn


class RolloutLoss:
    
    def __init__(self):
        self.loss_fn = nn.MSELoss()


    def __call__(self, model, batch, device):
        initial_state, rollout_sequence = batch
        u = initial_state[0].to(device)
        p = initial_state[1].to(device)
        params_tensor, target_tensor = rollout_sequence
        
        num_steps = params_tensor.shape[1]
        loss = 0
        for step in range(num_steps):
            y_pred = model(u, p)
            
            y_grnd = target_tensor[:, step, :].to(device)
            loss += self.loss_fn(y_pred, y_grnd) / num_steps
            
            p = params_tensor[:, step, :].to(device)
            u = y_pred
            
        return loss