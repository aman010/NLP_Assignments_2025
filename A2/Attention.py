#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 04:02:23 2025

@author: qb
"""
import torch
from torch import nn
import  torch.functional as F
pad_idx =0

class LSTM_GAtt(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=pad_idx)
        
        # let's use pytorch's LSTM
        self.lstm = nn.LSTM(embed_dim, 
                           hidden_dim, 
                           num_layers=4, 
                           bidirectional=True, 
                           dropout=.5,
                           batch_first=True)
        
        # Linear Layer for binary classification 
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def attention_net(self, lstm_output, hn):
        
        h_t      = hn.unsqueeze(2)
        H_keys   = torch.clone(lstm_output)
        H_values = torch.clone(lstm_output)
        
        alignment_score   = torch.bmm(H_keys, h_t).squeeze(2) # SHAPE : (bs, seq_len, 1)
        
        soft_attn_weights = torch.nn.functional.softmax(alignment_score, 1) # SHAPE : (bs, seq_len, 1)
        
        context           = torch.bmm(H_values.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2) # SHAPE : (bs, hidden_size * num_directions)
        
        return context

    def forward(self, text, text_lengths):

        embedded = self.embedding(text) # SHAPE : (batch_size, seq_len, embed_dim)

        lstm_output, (hn, cn) = self.lstm(embedded)
        
        # This is how we concatenate the forward hidden and backward hidden from Pytorch's BiLSTM
        hn = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)

        attn_output = self.attention_net(lstm_output, hn)
        output = self.fc(attn_output)
        return output