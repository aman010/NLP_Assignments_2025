#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 14:37:12 2025

@author: qb
"""
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer
import torch.nn.functional as F
import streamlit as st
from lora import LoRA

# LoRA BERT Model with modifications
class LoRA_BertModel(nn.Module):
    def __init__(self, original_model, r=4):
        super(LoRA_BertModel, self).__init__()
        self.bert = original_model
        
        # Example: Apply LoRA to the attention layers (this can be adjusted based on your needs)
        self.bert.encoder.layer = nn.ModuleList([
            self._apply_lora_to_layer(layer, r) for layer in self.bert.encoder.layer
        ])
    
    def _apply_lora_to_layer(self, layer, r):
        # Apply LoRA to the attention layers (you can apply LoRA to other parts of the model as needed)
        layer.attention.self.query = LoRA(layer.attention.self.query, r)
        layer.attention.self.key = LoRA(layer.attention.self.key, r)
        layer.attention.self.value = LoRA(layer.attention.self.value, r)
        
        # You can apply LoRA to other layers as well (e.g., output, feedforward, etc.)
        return layer
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
