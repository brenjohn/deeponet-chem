#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:09:07 2026

@author: jbrennan
"""

import os
import torch

from collections import defaultdict

class Trainer:
    """
    A simple Trainer class to encapsulate the training and validation logic.
    """
    
    def __init__(
            self, 
            model, 
            optimizer, 
            train_loader, 
            valid_loader, 
            loss_fn, 
            device
        ):
        """Initializes the Trainer with all necessary components.

        Args:
            model (torch.nn.Module): 
                The PyTorch model to train.
                
            optimizer (torch.optim.Optimizer): 
                The optimizer for updating model weights.
                
            train_loader (torch.utils.data.DataLoader): 
                DataLoader for the training set.
                
            valid_loader (torch.utils.data.DataLoader): 
                DataLoader for the validation set.
                
            loss_fn (callable): 
                The loss function.
                
            device (torch.device): 
                The device (e.g., 'cpu' or 'cuda') to run training on.
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_fn = loss_fn
        self.device = device
        self.best_valid_loss = float('inf')
        self.history = defaultdict(list)
        
        # Move the model to the specified device
        self.model.to(self.device)


    def train_epoch(self):
        """Performs one full training epoch.
        """
        self.model.train()
        dataset_size = len(self.train_loader)
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Unpack the batch data.
            u, t, y_true = [item.to(self.device) for item in batch]

            # Compute loss and update model
            self.optimizer.zero_grad()
            
            y_pred = self.model(u, t)
            loss = self.loss_fn(y_pred, y_true)
            
            # u2 = self.model(u, t/2)
            # y_pred_half = self.model(u2, t/2)
            # loss += self.loss_fn(y_pred_half, y_true)
            
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            
            if (batch_idx % 140 == 0):
                message = f'\rBatch {batch_idx+1}/{dataset_size}'
                message += f' loss = {loss.item()}'
                print(message, end='')
            
        print('\r', end='')
        return {'train_loss' : running_loss / len(self.train_loader)}


    def validate_epoch(self):
        """Performs one full validation epoch.
        """
        self.model.eval()
        dataset_size = len(self.valid_loader)
        loss_pred_true = 0.0
        # loss_iter_true = 0.0
        # loss_pred_iter = 0.0
        
        with torch.no_grad():
            for batch_idx, example in enumerate(self.valid_loader):
                us, ts, ys = [item.to(self.device) for item in example]
                
                # num_time_points = ts.shape[1]
                
                # u_repeated = u.unsqueeze(1)
                # u_repeated = u.repeat(1, num_time_points, 1)
                # u_repeated = u.reshape(-1, u.shape[-1])
                
                # ts_reshaped = ts.reshape(-1, 1)
                # ys_reshaped = ys.reshape(-1, ys.shape[-1])
    
                y_pred = self.model(us, ts)
                
                # y_iter = self.model(u_repeated, ts_reshaped/2)
                # y_iter = self.model(y_iter, ts_reshaped/2)
    
                # Compute the loss for the entire vectorized batch
                loss = self.loss_fn(y_pred, ys)
                loss_pred_true += loss.item()
                
                # loss = self.loss_fn(y_iter, ys_reshaped.T)
                # loss_iter_true += loss.item()
                
                # loss = self.loss_fn(y_pred, y_iter)
                # loss_pred_iter += loss.item()
                
                if (batch_idx % 140 == 0):
                    message = f'\rBatch {batch_idx+1}/{dataset_size}'
                    message += f' loss = {loss.item()}'
                    print(message, end='')
                
            print('\r', end='')
        
        return {
            'valid_loss' : loss_pred_true / len(self.valid_loader),
            'loss_pred_true' : loss_pred_true / len(self.valid_loader),
            # 'loss_iter_true' : loss_iter_true / len(self.valid_loader),
            # 'loss_pred_iter' : loss_pred_iter / len(self.valid_loader),
            # 'pred_sample' : y_pred.cpu().numpy(),
            # 'iter_sample' : y_iter.cpu().numpy(),
            # 'u_sample' : u.cpu().numpy()
        }


    def run(self, num_epochs, save_path=None):
        """Runs the training and validation process for a specified number of 
        epochs.

        Args:
            num_epochs (int): 
                The number of epochs to train for.
                
            save_path (str, optional): 
                Path to save the best model.
        """
        
        for epoch in range(num_epochs):
            epoch_metadata = {}
            epoch_metadata |= self.train_epoch()
            epoch_metadata |= self.validate_epoch()
            
            self.update_history(epoch_metadata)
            train_loss = epoch_metadata['train_loss']
            valid_loss = epoch_metadata['valid_loss']
            message = f"\r[Epoch {epoch+1}/{num_epochs}]"
            message += f"[Train Loss: {train_loss:.6f}]"
            message += f"[Valid Loss: {valid_loss:.6f}]"
            print(message)
            
            # Save the best model based on validation loss
            if save_path and valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.save_checkpoint(epoch, save_path)
                print(f"Saved model with validation loss: {valid_loss}")
                
                
    def update_history(self, epoch_metadata):
        for key, value in epoch_metadata.items():
            self.history[key].append(value)
                
    
    def save_checkpoint(self, epoch, path):
        """
        Saves the model and optimizer state to a file.
        """
        torch.save({
            'history': self.history,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)


    def load_checkpoint(self, path):
        """
        Loads the model and optimizer state from a checkpoint file.
        """
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.history = checkpoint['history']
            return True
        else:
            print(f"No checkpoint found at '{path}'.")
            return False