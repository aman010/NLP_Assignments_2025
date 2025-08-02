#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:42:59 2025

@author: qb
"""
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import torch
import numpy as np
from numpy.linalg import norm

class genism_model():
    def __init__(self, corpus, model, word2index):
        self.corpus= corpus
        self.model = model
        self.word2index = word2index
        
    def rpuncst(self, x):
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
         # Remove punctuation and stopwords
        # tokens = [token.lower() for token in tokens]
        # Apply stemming to each token
        tokens = [token for token in x if token.isalnum() and token.lower() not in stop_words]    
        return tokens
    
    def get_sim(self, query, top_k):
        query_words = query.split()
        query_words = self.rpuncst(query_words)
        
        query_vector = []
        
        for word in query_words:
            if word in self.word2index.keys():
                # wd=torch.tensor([self.word2index[word]], dtype=torch.long)
                try:
                    vector =self.model.get_vector(word)
                except KeyError:
                    continue
                
               
                query_vector.append((word,vector))
        print(query_vector)

                
        # query_vector=torch.stack(query_vector).sum(dim=0)
        flatten = lambda l: [item for sublist in l for item in sublist]
        #assign unique integer
        vocabs = list(set(flatten(self.corpus.values())))
        
        re = {}
        for v in vocabs:
            try:
                emv =self.model.get_vector(word)
            except KeyError:
                continue
            re[v] = emv
        def cos_sim(a, b):
            cos_sim = (a @ b.T)/(norm(a)*norm(b))
            return cos_sim
             
        sim = {}
        for q in query_vector:
            re_ = {}
            for v in vocabs:
                re_[v] = cos_sim(re[v],q[1])
                
            sim[q[0]] = dict(list(sorted(re_.items(), key=lambda item: item[1], reverse=True))[:10])
        print(sim)

        # for q in query_vector:
        #     sim[q[0]] = {key:values.tolist()[0] for key, values in sim[q[0]].items()}
        # print(sim)
            
        return sim
            
        
            
        
            