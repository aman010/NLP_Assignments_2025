#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 11:58:57 2025

@author: qb
"""

import streamlit as st
from datasets import load_dataset
from transformers import BertTokenizer
import torch
from Bert import Embedding, EncoderLayer, MultiHeadAttention, ScaledDotProductAttention, BERT, PoswiseFeedForwardNet, get_attn_pad_mask
from Bert import *
from Siames import SiameseNetworkWithBERT
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from huggingface_hub import login



dataset = load_dataset("multi_nli")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
login("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")


# Set page title
st.title("Predict the similarity")



# Text input boxes for entering the two inputs
input1 = st.text_input("Enter Text 1", placeholder="Enter the first input text here")
input2 = st.text_input("Enter Text 2", placeholder="Enter the second input text here")

# Select a subset of the dataset (e.g., first 1000 samples of the training set)
dataset = dataset.shuffle()
train_data = dataset['train'].select(range(100))  
validation_data = dataset['validation_matched'].select(range(20))

# Load the custom tokenizer
re_ = {'premise': [], 'hypothesis':[]}
def load_dataset(dataset):
    re_['premise'].extend(dataset['premise'])
    re_['hypothesis'].extend(dataset['hypothesis'])
    
    
#load model 
device = "cpu"
# Assuming you've already defined your BERT architecture like this
n_layers = 12  # number of Encoder of Encoder Layer
n_heads  = 12   # number of heads in Multi-Head Attention
d_model  = 768  # Embedding Size
d_ff = d_model * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2
vocab_size = 187368
max_len = 1059
device = "cpu"
# Initialize your custom BERT model
bert_ = BERT(
    n_layers, 
    n_heads, 
    d_model, 
    d_ff, 
    d_k, 
    n_segments, 
    vocab_size, 
    max_len, 
    device
).to(device)  # Move model to GPU if available

# Load the pretrained model weights (ensure 'BERT' is the correct path)
# pretrained_model = torch.load('./requires/BERT2', weights_only=False)  # Make sure the path is correct
#load from huggt face the model 
@st.cache_resource()
def load_model():
    model_id = "Aman010/Bert-Siames"
    model_path = hf_hub_download(repo_id= model_id, filename="pytorch_model.bin")
    pretrained_model = torch.load(model_path, weights_only=False)
    pretrained_model_state_dict = pretrained_model.state_dict()
    
    # Get the model's state dict
    model_state_dict = bert_.state_dict()
    
    
    # Iterate through the pretrained model's state dict and load weights
    for name, param in pretrained_model_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param)  # Copy weights if the name matches
    
    # Now load the updated state dict into your model
    
    # pretrained_model = torch.load('./requires/Saimes', weights_only=False)  # Make sure the path is correct
    model_path = hf_hub_download(repo_id = model_id , filename='Saimes')
    pretrained_model = torch.load(model_path,weights_only=False)
    pretrained_model_state_dict = pretrained_model.state_dict()
    model = SiameseNetworkWithBERT(num_labels=3, pretrained_model_name=bert_)  # 3 classes: entailment, contradiction, neutral
    
    # Get the model's state dict
    model_state_dict = model.state_dict()
    
    # Iterate through the pretrained model's state dict and load weights
    for name, param in pretrained_model_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param)  # Copy weights if the name matches
    
    # Now load the updated state dict into your model
    model.load_state_dict(model_state_dict)
    
    return model

model = load_model()

def inference(model, tokenizer, sentence1, sentence2, device):
    # Tokenize the input sentences
    inputs = tokenizer(sentence1, sentence2, padding=True, truncation=True, return_tensors="pt", max_length=max_len)
    
    # Move the tensors to the same device as the model (CPU or GPU)
    input_ids1 = inputs['input_ids'].detach().numpy()
    attention_mask1 = inputs['attention_mask'].detach().numpy()
    input_ids2 = inputs['input_ids'].detach().numpy()
    attention_mask2 = inputs['attention_mask'].detach().numpy()
    # Set the model to evaluation mode
    model.eval()
    # Disable gradients for inference
    with torch.no_grad():
        # Forward pass through the model
        logits = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
    # Convert logits to probabilities using softmax
    print(logits)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    print(probabilities)
    # Get the predicted class (the index with the highest probability)
    predicted_class = torch.max(probabilities, dim=1)
    print(predicted_class)
    
    return predicted_class, probabilities

re_=train_data.map(load_dataset)
option1 = list(range(len(re_['premise'])))
option2 = list(range(len(re_['hypothesis'])))
premise = st.selectbox("select data options", option1, index = None)

    
def get_sentences():
    # Dropdown to select the model
    sentence1 = re_['premise'][premise]
    sentence2 = re_['hypothesis'][premise]

    return sentence1, sentence2
   
def get_label(label):
    if label == 0:
        return "entailment"
    if label == 1:
        return "contradiction"
    if label == 2:
        return "neutral"

# Button to trigger model prediction
if st.button("Predict"):
    # Logic for handling the button click
   
    if premise == None:
        if not input1 or not input2:
            st.warning("inputs are empty")
        else:
            out=inference(model, tokenizer, input1, input2, device)
            print('label:', out[0][1])
            label=out[0][1].detach().numpy()
            label=get_label(label)
            st.write(label)
            print('out',out[1])
            fig, ax=plt.subplots()
            ax.bar(['entailment', 'contradiction', 'neutral'] , out[1].detach().numpy().flatten())
            st.pyplot(fig)
            
    else:
        sentence1, sentence2 = get_sentences()
        st.write(f"premise: {sentence1}")
        st.write(f"hypothesis: {sentence2}")
        model.eval()
        fig, ax=plt.subplots()

        out=inference(model, tokenizer, sentence1, sentence2, device)
        print('label:', out[0][1])
        label=out[0][1].detach().numpy()
        label=get_label(label)
        st.write(label)
        print('out',out[1])
        ax.bar(['entailment', 'contradiction', 'neutral'] , out[1].detach().numpy().flatten())
        st.pyplot(fig)
    
        
    # st.write(f"Text 1: {input1}")
    # st.write(f"Text 2: {input2}")
    
    
    # st.write(f"Selected Model: {selected_model}")
    
    # Here you can add the logic to pass the inputs to the selected model and show predictions.
    # For example:
    # prediction = some_model_predict_function(input1, input2, selected_model)
    # st.write(f"Prediction: {prediction}")
    
    # Just for demo purposes
