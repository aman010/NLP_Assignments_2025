#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:09:07 2025

@author: qb
"""
import torch
import torch.nn as nn
from Bert import *
import numpy as np

device = "cpu"

class SiameseNetworkWithBERT(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_labels=3):
        super(SiameseNetworkWithBERT, self).__init__()
        # Load the pre-trained BERT model
        self.bert = pretrained_model_name
        for param in self.bert.parameters():
                param.requires_grad = True  #
        # Classifier to map concatenated embeddings to final output
        self.classifier = nn.Linear(d_model * 3, num_labels)  # [u, v, |u - v|]

    def mean_pooling(self, token_embeddings, attention_mask):
        # We create the attention mask to ignore the padding tokens while calculating the mean
        token_embeddings = token_embeddings.cpu().detach().numpy()
        token_embeddings = token_embeddings * attention_mask[:,:,np.newaxis]
        # Sum the embeddings along the sequence length axis (dim=1) and divide by the number of tokens
        pooled_output = token_embeddings.sum(axis=1) / np.sum(attention_mask, axis=1, keepdims=True)
        return pooled_output

    def forward_one(self, input_ids, attention_mask):
        # Get batch size and sequence length
        batch_size, max_len = input_ids.shape
        
        # Create segment_ids (assuming all inputs belong to the same segment)
        segment_ids = torch.zeros(batch_size, max_len, dtype=torch.int32).to(device) 
        
        # Get the output embeddings from the pre-trained BERT model
        output = self.bert.get_last_hidden_state(torch.tensor(input_ids).to(device), segment_ids)
        # Apply mean pooling to the token embeddings
        pooled_embedding = self.mean_pooling(output, attention_mask)
        return pooled_embedding

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        # Forward pass through the pre-trained BERT for both sentences
        embedding1 = self.forward_one(input_ids=input_ids1, attention_mask=attention_mask1)
        embedding2 = self.forward_one(input_ids=input_ids2, attention_mask=attention_mask2)
        # print(embedding1)
        # print(embedding2)
        # Compute the absolute difference between embeddings
        abs_diff = np.abs(embedding1 - embedding2)
        
        # Concatenate embeddings: [u, v, |u - v|]
        combined_embedding = np.concatenate((embedding1, embedding2, abs_diff), axis=1)
        # print(combined_embedding.shape)
        # Forward pass through the classifier to get logits
        logits = self.classifier(torch.tensor(combined_embedding).float())
        return logits