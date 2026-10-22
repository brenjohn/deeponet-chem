#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:09:07 2026

@author: jbrennan
"""

import json
import torch
import optuna
import torch.nn as nn

from collections import defaultdict


class Trainer:
    """A Trainer class to encapsulate DeepONet training logic.
    """
    
    def __init__(
            self, 
            model, 
            optimizer,
            scheduler,
            train_loader, 
            validators, 
            primary_loss, 
            device
        ):
        """Initializes the Trainer with all necessary components.

        Args:
            model        : The PyTorch model to train.
            optimizer    : The optimizer for updating model weights.
            train_loader : DataLoader for the training set.
            validators   : List of validators.
            primary_loss : Which loss to prioritize when saving checkpoints.
            device       : The device to run training on.
        """
        self.model        = model
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.train_loader = train_loader
        self.validators   = validators
        self.primary_loss = primary_loss
        self.device       = device
        
        self.model.to(self.device)
        self.loss_fn = nn.MSELoss()
        self.best_valid_loss = float('inf')
        self.history = defaultdict(list)


    #=========================================================================#
    # Training methods
    #=========================================================================#
    
    def train_epoch(self):
        """Performs one full training epoch.
        """
        self.model.train()
        dataset_size = len(self.train_loader)
        total_loss = 0.0
        
        for batch_idx, (one_batch, five_batch) in enumerate(self.train_loader):
            # Unpack the batch data.
            u, t, y_true = [item.to(self.device) for item in one_batch]

            # Compute loss and update model
            self.optimizer.zero_grad()
            
            # One step loss
            y_pred = self.model(u, t)
            loss = self.loss_fn(y_pred, y_true)
            
            # Five step loss
            initial_state, rollout_sequence = five_batch
            u = initial_state[0].to(self.device)
            t = initial_state[1].to(self.device)
            params_tensor, target_tensor = rollout_sequence
            num_steps = params_tensor.shape[1]
            for step in range(num_steps):
                p = params_tensor[:, step, :].to(self.device)
                y = target_tensor[:, step, :].to(self.device)
                y_pred = self.model(u, t)
                u = torch.concat([p, y_pred.detach()], dim=1)
                loss += self.loss_fn(y_pred, y) / num_steps
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
            if (batch_idx % 350 == 0):
                message = f'\r[Train batch {batch_idx}/{dataset_size}'
                message += f' loss = {loss.item():0.4e}]'
                print(message, end='')
            
        print('\r', end='')
        train_loss     = total_loss / len(self.train_loader)
        train_metadata =  {'train_loss' : train_loss}
        train_message  = f"[Train Loss: {train_loss:.6e}]"
        return train_metadata, train_message


    def run_epoch(self):
        """Runs the training and validation process for a specified number of 
        epochs.
        """
        train_metadata, train_message = self.train_epoch()
        valid_metadata, valid_message = self.validate()
        
        if self.scheduler is not None:
            primary_loss = valid_metadata[self.primary_loss]
            self.scheduler.step(primary_loss)
        
        epoch_metadata = train_metadata | valid_metadata
        message = train_message + valid_message
        return epoch_metadata, message
    
    
    def run(self, num_epochs, save_path=None):
        """Runs the training and validation process for a specified number of 
        epochs.

        Args:
            num_epochs (int): The number of epochs to train for.
            save_path (str, optional): Path to save the best model.
        """
        
        for epoch in range(num_epochs):
            epoch_metadata, message = self.run_epoch()
            print(f"\r[Epoch {epoch}/{num_epochs}]" + message)
            
            primary_loss = epoch_metadata[self.primary_loss]
            self.update_history(epoch_metadata)
            
            # Save the best model based on validation loss
            if primary_loss < self.best_valid_loss:
                self.best_valid_loss = primary_loss
                if save_path is not None:
                    checkpoint = save_path / f'best_model_{primary_loss:0.4e}'
                    checkpoint.mkdir(parents=True, exist_ok=True)
                    self.save_checkpoint(checkpoint)
                    print(f"Saved model with validation loss: {primary_loss}")
            
            # Save the current model
            checkpoint = save_path / 'current_model'
            checkpoint.mkdir(parents=True, exist_ok=True)
            self.save_checkpoint(checkpoint)
                
        return epoch_metadata
                
                
    def train_with_optuna(self, trial, epochs):
        """Trains the model while reporting losses to Optuna for pruning."""
        best_valid_loss = float("inf")

        for epoch in range(epochs):
            epoch_metadata, epoch_message = self.run_epoch()
            
            primary_loss = epoch_metadata[self.primary_loss]
            trial.report(primary_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if primary_loss < best_valid_loss:
                best_valid_loss = primary_loss
            
            message  = f'[Epoch: {epoch}][valid loss: {primary_loss}]'
            message += epoch_message
            print(message)

        return best_valid_loss
    
    
    #=========================================================================#
    # Training utilities
    #=========================================================================#
         
    def validate(self):
        """Performs one full validation epoch.
        """
        self.model.eval()
        
        valid_metadata = {}
        valid_message = ''
        
        for validator in self.validators:
            metadata, message = validator.validate(self.model, self.device)
            valid_metadata |= metadata
            valid_message += message
            
        return valid_metadata, valid_message
    
    
    def update_history(self, epoch_metadata):
        for key, value in epoch_metadata.items():
            self.history[key].append(value)
    
    
    #=========================================================================#
    # Saving checkpoints
    #=========================================================================#
    
    def save_checkpoint(self, checkpoint_path):
        """Saves the model and optimizer state to a file.
        """
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save(checkpoint_path / 'model.pt')
        
        checkpoint_data = {
            'scheduler_state_dict': self.scheduler.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint_data, checkpoint_path / 'optimizer.pt')
        
        with open(checkpoint_path / 'history.json', 'w') as file:
            json.dump(self.history, file, indent=4)