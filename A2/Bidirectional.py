#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 04:01:00 2025

@author: qb
"""
import torch
from torch import nn


class BiLSTM_model(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, output_dim, pad_idx, dropout_prob=0.6):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=pad_idx)
        
        # LSTM layer (BiLSTM means bidirectional, so 2 * hidden_dim outputs)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Fully connected layer to map the LSTM output to the vocab size
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # hidden_dim * 2 for bidirectional LSTM

    def forward(self, text):
        # Get the embeddings of the input sequence
        embedded = self.embedding(text)  # Shape: [batch_size, seq_len, embed_dim]
        
        # Pass the embeddings through the BiLSTM
        lstm_out, _ = self.lstm(embedded)  # lstm_out: [batch_size, seq_len, hidden_dim * 2]
        
        # Apply dropout to the LSTM outputs for regularization
        lstm_out = self.dropout(lstm_out)
        
        # Pass the LSTM output through the fully connected layer
        output = self.fc(lstm_out)  # Output shape: [batch_size, seq_len, vocab_size]
        
        return output

    def compute_loss(self, output, target):
        """
        Calculates the cross-entropy loss for next-word prediction.
    
        Args:
        - output (torch.Tensor): The model's predicted word scores [batch_size, seq_len, vocab_size]
        - target (torch.Tensor): The true next words [batch_size, seq_len]
        
        Returns:
        - loss (torch.Tensor): The cross-entropy loss
        """
        # Flatten the output and target for cross-entropy loss calculation
        output_flat = output.view(-1, output.size(-1))  # Shape: [batch_size * seq_len, vocab_size]
        target_flat = target.view(-1)  # Shape: [batch_size * seq_len]
        
        # Cross-entropy loss expects the target to be the index of the correct class
        loss = nn.CrossEntropyLoss()(output_flat, target_flat)  # Compute cross-entropy loss
    
        return loss