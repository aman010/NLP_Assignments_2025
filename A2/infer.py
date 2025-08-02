#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 06:02:30 2025

@author: qb
"""
import torch

class inference:
    def infer_with_sliding_window(self,model, input_sentence, vocab, device, seq_len=3, pad_idx=0):
        print(model, type(model))
        model.eval()
        
        input_tensor = torch.tensor([vocab.get(word, vocab['<unk>']) for word in input_sentence], dtype=torch.long).to(device)
        
        # Pad the sequence to the required length (if necessary)
        if len(input_tensor) < seq_len:
            padding = torch.full((seq_len - len(input_tensor),), pad_idx, dtype=torch.long).to(device)
            input_tensor = torch.cat((input_tensor, padding), dim=0).to(device)
    
        # Sliding window: create chunks of size `seq_len` with stride of 1
        chunks = [input_tensor[i:i+seq_len] for i in range(0, len(input_tensor) - seq_len + 1)]
        
        predicted_words = []
    
        # For each chunk, predict the next word
        for chunk in chunks:
            # Pad the chunk if it's smaller than `seq_len` (i.e., the last chunk)
            if len(chunk) < seq_len:
                chunk = torch.cat((chunk, torch.full((seq_len - len(chunk),), pad_idx, dtype=torch.long).to(device)))
    
            # Create a tensor for the current chunk and its length
            chunk = chunk.unsqueeze(0)  # Add batch dimension (batch_size = 1)
            lengths = torch.tensor([len(chunk)], dtype=torch.long)
            
            # Pack the input sequence (for variable-length sequences)
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(chunk, lengths, batch_first=True, enforce_sorted=False).to(device)
    
            # Get the model's prediction for this chunk
            with torch.no_grad():
                output = model(packed_input, lengths)
            
            # Convert the output to a list of predicted tokens for each time step
            output = output.squeeze(0)  # Remove batch dimension (batch_size = 1)
            
            # For the last token in the chunk, predict the next word
            _, predicted_idx = output[-1].max(dim=-1)  # Get the predicted index for the last token
            predicted_word = list(vocab.keys())[list(vocab.values()).index(predicted_idx.item())]
            predicted_words.append(predicted_word)
    
        return predicted_words
    
    def infer_last_word(self,model, input_sentence, vocab, device, pad_idx=0): 
        # Ensure the model is in evaluation mode
        model.eval()
    
        # Tokenize and pad the input sequence
        input_tensor = torch.tensor([vocab.get(word, vocab['<unk>']) for word in input_sentence], dtype=torch.long).to(device)
        
        # Pad the sequence to the required length (if necessary)
        # if len(input_tensor) < seq_len:
        #     padding = torch.full((seq_len - len(input_tensor),), pad_idx, dtype=torch.long).to(device)
        #     input_tensor = torch.cat((input_tensor, padding), dim=0).to(device)
    
        # Create tensor for the full input sentence
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension (batch_size = 1)
        lengths = torch.tensor([len(input_tensor)], dtype=torch.long)
        
        # Pack the input sequence (for variable-length sequences)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input_tensor, lengths, batch_first=True, enforce_sorted=False).to(device)
    
        # Get the model's prediction for the entire input sequence
        with torch.no_grad():
            output = model(packed_input, lengths)
        
        # Get the output corresponding to the last token (the last word in the sentence)
        output = output.squeeze(0)  # Remove batch dimension (batch_size = 1)
        
        # For the last token in the sequence, predict the next word
        # We get the output for the last token position
        _, predicted_idx = output[-1].max(dim=-1)  # Get the predicted index for the last token
        
        # Convert the predicted index back to a word
        predicted_word = list(vocab.keys())[list(vocab.values()).index(predicted_idx.item())]
        
        return predicted_word
    
    
    def pad_batch_to_fixed_length(self,seq, seq_len, pad_idx=0, device='cpu'):
        padded_batch = []
    
        # Convert the sequence to a tensor
        seq_tensor = torch.tensor(seq, dtype=torch.long).to(device)

        # If sequence length is shorter than `seq_len`, pad it
        if seq_tensor.size(0) < seq_len:
            padding = torch.full((seq_len - seq_tensor.size(0),), pad_idx, dtype=torch.long).to(device)
            padded_seq = torch.cat((seq_tensor, padding), dim=0)
        else:
            padded_seq = seq_tensor[:seq_len]  # Truncate if longer than seq_len

        padded_batch.append(padded_seq)

# Stack the padded sequences into a tensor
        padded_batch = torch.stack(padded_batch, dim=0)
        return padded_batch
