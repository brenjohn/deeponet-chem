#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:01:25 2026

@author: jbrennan
"""

import torch
import torch.nn as nn


class DeepONet(nn.Module):
    
    def __init__(
            self,
            output_size,
            latent_size,
            input_size, 
            branch_size=70, 
            branch_layers=4,
            trunk_size=70, 
            trunk_layers=4
        ):
        """
        DeepONet for solving IVPs.
        
        Args:
            input_size (int): Dimension of initial condition vector.
            branch_size (int): Number of neurons in the branch network layers.
            branch_layers (int): The number of hidden layers in branch network.
            trunk_size (int): Number of neurons in the trunk network layers.
            trunk_layers (int): The number of hidden layers in trunk network.
        """
        super(DeepONet, self).__init__()
        
        self.output_size = output_size
        self.latent_size = latent_size
        self.input_size = input_size

        # Define the Branch Network
        self.branch_net = self._build_network(
            input_size, branch_size, branch_layers, latent_size * output_size
        )

        # Define the Trunk Network
        self.trunk_net = self._build_network(
            1, trunk_size, trunk_layers, latent_size
        )


    def _build_network(self, input_size, layer_size, num_layers, output_size):
        """Helper function to build a multi-layer perceptron."""
        layers = []
        
        curr_size = input_size
        next_size = layer_size
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_size, next_size))
            layers.append(nn.Tanh())
            curr_size = layer_size
            
        layers.append(nn.Linear(curr_size, output_size))
        layers.append(nn.Tanh())

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
        output = torch.squeeze(output)
        # [batch, output, latent] x [batch, latent] -> [batch, output]
        # output = torch.einsum('bol,bl->bo', branch_output, trunk_output)
        
        return output