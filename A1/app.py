#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 06:50:31 2025

@author: qb
"""
from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
import numpy as np
import torch 
from Models import Glove
from Models import SkipgramNeg
import pandas as pd
import json
from preprocess import process
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim_pro import genism_model

app = Flask(__name__)

# df=pd.read_parquet('Model_corpus/0.parquet')
# df['text'] = df['text'].astype('unicode')
# df=df['text'][:1000]


with open('Model_corpus/corpus.json', 'r') as fp:
    corpus = json.load(fp)
    
def pop_numeric(tokens):
    """
    Pops numeric tokens from the list and keeps only non-numeric tokens.
    """
    index = 0
    while index < len(tokens):
        if tokens[index].isdigit():  # Check if the token is numeric
            tokens.pop(index)  # Remove numeric token
        else:
            index += 1  # Only increment index if we don't pop

    return tokens

for doc in corpus:
    corpus[doc]=pop_numeric(corpus[doc])

#load the model 
# model = SkipgramNeg(8743, 2)
# model.load_state_dict(torch.load('Model_corpus/neg_samples', weights_only=True, map_location=torch.device('cpu')))

# with open('Model_corpus/word2index.json', 'r') as fp:
#     word2index = json.load(fp)
word2index = np.load('Model_corpus/word2index.npy', allow_pickle=True).item()

def rpuncst(x):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
     # Remove punctuation and stopwords
    # tokens = [token.lower() for token in tokens]
    # Apply stemming to each token
    tokens = [token for token in x if token.isalnum() and token.lower() not in stop_words]    
    return tokens

# Callback function for handling search
def search_callback(query, model_type):
    # query = str(query).split(' ')
    # query = "traveling to the place"
    if model_type == 'skipGram':
        model = SkipgramNeg(8743, 2)
        model.load_state_dict(torch.load('Model_corpus/neg_samples', weights_only=True, map_location=torch.device('cpu')))
        model.eval()  # Set the model to evaluation mode
        p  = process(corpus, model, word2index)
        re=p.find_most_similar_documents(str(query), corpus)
        

    if model_type == 'Glove':
        model = Glove(8743, 2)
        model.load_state_dict(torch.load('Model_corpus/glove', weights_only=True))
        model.eval()  # Set the model to evaluation mode
        p  = process(corpus, model, word2index)
        re=p.find_most_similar_documents(str(query), corpus)
        
    return re
    #create a search corpus of the given text

# Route for the home page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    # print(data)
    if not data or "query" not in data:
       return jsonify({"error": "Missing query parameter"}), 400
    query = data["query"]
    model_type = data.get("model", "word2vec")
    results = search_callback(query, model_type)
    # print("results", results)
    return jsonify({"result": results})

if __name__ == "__main__":
    app.run(debug=True)

    