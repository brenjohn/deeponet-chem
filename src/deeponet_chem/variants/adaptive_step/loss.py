#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 03:15:47 2026

@author: john
"""

import torch.nn as nn


class OneStepLoss:
    
    def __init__(self):
        self.loss_fn = nn.MSELoss()
        
    def __call__(self, model, batch, device):
        u, t, y_true = [item.to(device) for item in batch]
        y_pred = model(u, t)
        return self.loss_fn(y_pred, y_true)