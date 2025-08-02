#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 14:23:32 2025

@author: qb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# LoRA class definition
class LoRA(nn.Module):
    def __init__(self, layer, r=4):
        """
        Apply LoRA to a given layer.
        `r` is the rank of the low-rank update (this can be adjusted)
        """
        super(LoRA, self).__init__()
        self.layer = layer
        self.r = r
        
        # Create low-rank matrices (for LoRA)
        self.A = nn.Parameter(torch.randn(self.layer.weight.size(0), self.r))
        self.B = nn.Parameter(torch.randn(self.r, self.layer.weight.size(1)))
        
        # Freeze original weights during training
        self.layer.weight.requires_grad = False
        
    def forward(self, x):
        # Perform the LoRA low-rank update
        return F.linear(x, self.layer.weight + torch.matmul(self.A, self.B))

# Function to apply LoRA to layers (odd/even) of the model
def apply_lora_to_layers(student_model, odd_even="odd"):
    """
    Apply LoRA to odd/even layers of the student model
    `odd_even` should be "odd" or "even"
    """
    layers_to_update = []
    layer_count = 0  # To track the layer order (odd/even)
    
    # Iterate over the modules to collect layers to apply LoRA to
    for name, module in student_model.named_modules():
        if isinstance(module, nn.Linear):
            # Increment the layer count for each Linear layer
            layer_count += 1
            
            # Apply LoRA to odd or even layers based on layer_count
            if odd_even == "odd" and layer_count % 2 == 1:
                layers_to_update.append((name, module))
            elif odd_even == "even" and layer_count % 2 == 0:
                layers_to_update.append((name, module))
    
    # Now, replace the layers
    for name, module in layers_to_update:
        student_model._modules[name] = LoRA(module)
    
    return student_model