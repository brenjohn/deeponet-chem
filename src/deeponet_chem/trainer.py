#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:09:07 2026

@author: jbrennan
"""

import torch
import optuna

from collections import defaultdict

class Trainer:
    """
    A simple Trainer class to encapsulate the training and validation logic.
    """
    
    def __init__(
            self, 
            model, 
            optimizer,
            scheduler,
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
        self.scheduler = scheduler
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
            
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            
            if (batch_idx % 350 == 0):
                message = f'\r[Train batch {batch_idx}/{dataset_size}'
                message += f' loss = {loss.item():0.4e}]'
                print(message, end='')
            
        print('\r', end='')
        return {'train_loss' : running_loss / len(self.train_loader)}


    def validate_epoch(self):
        """Performs one full validation epoch.
        """
        self.model.eval()
        dataset_size = len(self.valid_loader)
        loss_pred_true = 0.0
        # rollout_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, example in enumerate(self.valid_loader):
                us = example[0].to(self.device)[0]
                ts = example[1].to(self.device)[0]
                ys = example[2].to(self.device)[0]
    
                y_pred = self.model(us, ts)
    
                # Compute the loss for the entire vectorized batch
                loss = self.loss_fn(y_pred, ys)
                loss_pred_true += loss.item()
                
                #==========================#
                # u = us[0:1, :]
                # dt = ts[5:6, :]
                # for step in range(1, 5):
                #     y_pred = self.model(u, dt)
                    
                #     start = step * 5
                #     u = torch.concat([us[start, :7], y_pred])[None, :]
                    
                # y_target = ys[-1, :]
                # loss = self.loss_fn(y_pred, y_target)
                # rollout_loss += loss.item()
                #==========================#
                
                if (batch_idx % 350 == 0):
                    message = f'\r[Valid batch {batch_idx}/{dataset_size}'
                    message += f' loss = {loss.item():0.4e}]'
                    print(message, end='')
                
            print('\r', end='')
        
        return {
            'valid_loss' : loss_pred_true / len(self.valid_loader),
            'one_step_loss' : loss_pred_true / len(self.valid_loader),
            # 'rollout_loss' : rollout_loss / len(self.valid_loader)
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
            message += f"[Train Loss: {train_loss:.6e}]"
            message += f"[Valid Loss: {valid_loss:.6e}]"
            print(message)
            
            # Save the best model based on validation loss
            if save_path and valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.save_checkpoint(epoch, save_path)
                print(f"Saved model with validation loss: {valid_loss}")
                
                
    def train_with_optuna(self, trial, epochs):
        """Trains the model while reporting losses to Optuna for pruning."""
        best_valid_loss = float("inf")

        for epoch in range(epochs):
            self.train_epoch()
            results = self.validate_epoch()
            valid_loss = results['valid_loss']

            # Report intermediate validation loss to Optuna
            trial.report(valid_loss, step=epoch)

            # Check if Optuna thinks this trial should be pruned early
            if trial.should_prune():
                raise optuna.TrialPruned()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                
            if self.scheduler is not None:
                self.scheduler.step(valid_loss)
                
            print(f'[Epoch: {epoch}][valid loss: {valid_loss}]')

        return best_valid_loss
                
                
    def update_history(self, epoch_metadata):
        for key, value in epoch_metadata.items():
            self.history[key].append(value)
                
    
    def save_checkpoint(self, epoch, path):
        """Saves the model and optimizer state to a file.
        """
        checkpoint_data ={
            'history': self.history,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        if self.scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint_data, path)


    def load_checkpoint(self, path):
        """Loads the trainer state from a checkpoint file.
        """
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        if getattr(self, 'scheduler', None) is not None:
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(
                    checkpoint['scheduler_state_dict']
                )