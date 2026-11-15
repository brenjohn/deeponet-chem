#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:01:25 2026

@author: jbrennan
"""

import torch
import torch.nn as nn

ACTIVATIONS = {
    'tanh' : nn.Tanh,
    'gelu' : nn.GELU,
    'silu' : nn.SiLU,
    'selu' : nn.SELU
}


class DeepONet(nn.Module):
    
    def __init__(
            self,
            output_size,
            latent_size,
            branch_input_size,
            trunk_input_size,
            branch_size=70, 
            branch_layers=4,
            branch_out_act=False,
            trunk_size=70, 
            trunk_layers=4,
            trunk_out_act=True,
            activation='tanh',
            residual_offset=None,
        ):
        """
        DeepONet for solving IVPs.
        
        Args:
            input_size (int): Dimension of initial condition vector.
            branch_size (int): Number of neurons in the branch network layers.
            branch_layers (int): The number of hidden layers in branch network.
            trunk_size (int): Number of neurons in the trunk network layers.
            trunk_layers (int): The number of hidden layers in trunk network.
            activation (str): The name of the activation function to use.
        """
        super(DeepONet, self).__init__()
        
        self.output_size     = output_size
        self.latent_size     = latent_size
        self.residual_offset = residual_offset
        
        self.config = {
            'output_size'       : output_size,
            'latent_size'       : latent_size,
            'branch_input_size' : branch_input_size,
            'trunk_input_size'  : trunk_input_size,
            'branch_size'       : branch_size, 
            'branch_layers'     : branch_layers,
            'branch_out_act'    : branch_out_act,
            'trunk_size'        : trunk_size, 
            'trunk_layers'      : trunk_layers,
            'trunk_out_act'     : trunk_out_act,
            'activation'        : activation,
            'residual_offset'   : residual_offset,
        }

        # Define the Branch Network
        self.branch_net = self._build_network(
            branch_input_size, 
            branch_size, 
            branch_layers, 
            latent_size * output_size,
            activation,
            branch_out_act
        )
        
        # Zero the final layer if a residual connection is used so the model
        # starts at the identity map.
        if residual_offset is not None:
            nn.init.zeros_(self.branch_net[-1].weight)
            if self.branch_net[-1].bias is not None:
                nn.init.zeros_(self.branch_net[-1].bias)

        # Define the Trunk Network
        self.trunk_net = self._build_network(
            trunk_input_size, 
            trunk_size, 
            trunk_layers, 
            latent_size, 
            activation,
            trunk_out_act
        )


    def _build_network(
            self, 
            input_size, 
            layer_size, 
            num_layers, 
            output_size,
            activation,
            final_activation
        ):
        """Helper function to build a multi-layer perceptron."""
        Activation = ACTIVATIONS[activation]
        layers = []
        
        curr_size = input_size
        next_size = layer_size
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_size, next_size))
            layers.append(Activation())
            curr_size = layer_size
            
        layers.append(nn.Linear(curr_size, output_size))
        if final_activation:
            layers.append(Activation())

        return nn.Sequential(*layers)


    def forward(self, u, t):
        """
        Forward pass of the DeepONet.

        Args:
            u (torch.Tensor): 
                Initial conditions. Shape: [batch_size, branch_input_size].
            
            t (torch.Tensor): 
                Time coordinates. Shape: [batch_size, trunk_input_size].

        Returns:
            torch.Tensor: The predicted solution at the query points.
        """
        # Pass inputs through the respective networks
        branch_output = self.branch_net(u)
        trunk_output = self.trunk_net(t)
        
        # Element-wise product
        branch_output = branch_output.view(
            -1, self.output_size, self.latent_size
        )
        
        output = torch.matmul(branch_output, trunk_output[:, :, None])
        output = output.squeeze(-1)
        
        if self.residual_offset is not None:
            offset = self.residual_offset
            output = output + u[:, offset:offset + self.output_size]
        
        return output
    
    
    def save(self, path):
        """Saves the model architecture configuration and weights."""
        checkpoint = {
            'config': self.config,
            'state_dict': self.state_dict()
        }
        torch.save(checkpoint, path)


    @classmethod
    def load(cls, path, map_location=None):
        """Class method to load a DeepONet instance from file."""
        checkpoint = torch.load(
            path, map_location=map_location, weights_only=False
        )
        config = checkpoint['config']
        # TODO: remove this, it was meant to be tempoary so old checkpoints
        # would load for comparison to new models.
        config.setdefault('branch_final_activation', True)
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'])
        return model
