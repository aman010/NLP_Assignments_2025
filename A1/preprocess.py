#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 08:41:03 2025

@author: qb
"""
import json
import torch 
from Models import Glove
from Models import SkipgramNeg
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
#load the corpus
from numpy import dot
from numpy.linalg import norm
import requests
import itertools
import re as reg

class process:
    def __init__(self, corpus, model, word2index):
        # self.df = df
        self.corpus = corpus
        self.model = model
        self.word2index = word2index
        #load word to index
        self.document_embeddings = {}
    
    # #load the corpus
    
    # # compute the document vectors for each document in the corpus
        # for doc in corpus:
        #     word_vectors = []
        #     for i,word in enumerate(self.corpus[doc]):
        #         if word in word2index.keys():  
        #             wd=torch.tensor([self.word2index[word]], dtype=torch.long)
        #             class_name = self.model.__class__.__name__
        #             if class_name == 'SkipgramNeg':
        #                 v = self.model.embedding_center(wd)
        #                 u = self.model.embedding_outside(wd)
        #             if class_name == 'Glove':
        #                 v = self.model.center_embedding(wd)
        #                 u = self.model.center_embedding(wd)
        #             vector = (v + u)/2
        #             word_vectors.append(vector.detach())
                
        #     if word_vectors:
        #         doc_vector=torch.stack(word_vectors).mean(dim=0)
        #         self.document_embeddings[doc] = (doc_vector)
        #         # doc_vector = np.mean(word_vectors, axis=0)
        #         # print(doc_vector)
    
        #     else:
        #         pass
        
    # print(document_embeddings)
    
    def rpuncst(self, x):
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
         # Remove punctuation and stopwords
        # tokens = [token.lower() for token in tokens]
        # Apply stemming to each token
        tokens = [token for token in x if token.isalnum() and token.lower() not in stop_words]    
        return tokens
    
    def get_embed(self,word):
        id_tensor=torch.tensor([self.word2index[word]], dtype=torch.long)

        
        class_name = self.model.__class__.__name__
        if class_name == 'SkipgramNeg':
            v = self.model.embedding_center(id_tensor)
            u = self.model.embedding_outside(id_tensor)            
            word_embed  = (u+ v)/2
            
        if class_name == 'Glove':
            v = self.model.center_embedding(id_tensor)
            u = self.model.center_embedding(id_tensor)
            word_embed  = (u+ v)/2

        # x, y = word_embed[0][0].item(), word_embed[0][1].item()
    
        # return x, y
        return word_embed 
    
    
    def cos_sim(self,a, b):
        cos_sim = dot(a, b)/(norm(a)*norm(b))
        return cos_sim
      

    
    def find_most_similar_documents(self, query, corpus,top_k=5):
        query_words = query.split()
        query_words = self.rpuncst(query_words)
        
        query_vector = []  # Initialize a zero vector
        # with open('Model_corpus/vocab.json', 'r') as fp:
        #     vocabs = json.load(fp)
        # Sum the word vectors for the query
        for word in query_words:
            if word in self.word2index.keys():
                vector=self.get_embed(word)
                query_vector.append((word,vector))
        
  
        def remove_stopwords(x, _stopwords):
            '''x is sentence'''
            stems = [w for w in x if not w.lower() in _stopwords]
            return stems
        # query_vector=torch.stack(query_vector).sum(dim=0)
        flatten = lambda l: [item for sublist in l for item in sublist]
        #assign unique integer
        stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
        stopwords_ = set(stopwords_list.decode().splitlines())
        vv = [remove_stopwords(i, stopwords_) for i in list(self.corpus.values())]
        vocabs = list(set(flatten(vv)))
        
        for index,token in enumerate(vocabs):
            k=reg.findall(r'\d+', token)
            if k:
                vocabs.pop(index)
        
        
        re = {}
        for v in vocabs:
            emv =self.get_embed(v)
            re[v] = emv
             
        sim = {}
        rep = {}
        for q in query_vector:
            re_ = {}
            for v in vocabs:
                re_[v] = self.cos_sim(re[v].detach().numpy(),q[1].detach().numpy().T)
                
            sim[q[0]] = dict(list(sorted(re_.items(), key=lambda item: item[1], reverse=True))[:10])
          

            
        for q in query_vector:
            sim[q[0]] = {key:values[0].tolist() for key, values in sim[q[0]].items()}
        
        rp = {}
        for i in sim:
            rp[i] = list(sim[i].keys())
        # print(rp)
            
        # print(self.word_pair_relationships(query_vector))
        in_ = self.word_pair_relationships(query_vector)
        print(in_)
        # print("merged_dictionary:",{**sim, **in_})
        
        # Compute similarity with each document
        # similarities = []
        # for i, doc_vector in enumerate(self.document_embeddings):
        #     self.document_embeddings[doc_vector]
        #     sim = query_vector @ self.document_embeddings[doc_vector].T / (torch.norm(query_vector) * torch.norm(self.document_embeddings[doc_vector]))
        #     similarities.append((doc_vector, sim.item()))
    
        # # # Sort documents by similarity (highest first)
        # similarities=sorted(similarities, key = lambda x:x[1], reverse=True)
        # # # Get the top K most similar documents
        
        # top_documents = [self.corpus[i[0]] for i, _ in similarities[:top_k]]
        # res =[self.df.loc[int(i[0])] for i, _ in similarities[:top_k]]
        
    
        return {**sim, **in_}
      
    def word_pair_relationships(self,word_vectors):
        relationships = {}
        def remove_stopwords(x, _stopwords):
            '''x is sentence'''
            stems = [w for w in x if not w.lower() in _stopwords]
            return stems
        
        stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
        stopwords_ = set(stopwords_list.decode().splitlines())
        flatten = lambda l: [item for sublist in l for item in sublist]
        
        vv = [remove_stopwords(i, stopwords_) for i in list(self.corpus.values())]
        vocabs = list(set(flatten(vv)))
        re = {}
        for v in vocabs:
            emv =self.get_embed(v)
            re[v] = emv

        # Generate all combinations of word pairs (without repetition)
        word_pairs = itertools.combinations(word_vectors, 2)
        
        in_p = {}
        in_pl = []
        for word1, word2 in word_pairs:
            # Get vectors for both words
            # print(word1, word2)
            vec1 = word1[1]
            vec2 = word2[1]
            
            # Calculate the relationship vector (difference)
            relationship_vector = vec2 * vec1
            relationship_vector2 = vec2 - vec1

            # Compute cosine similarity of the relationship vector with the original vectors
            
            
            # sim1 = self.cos_sim(a=vec1.detach().numpy(), b=relationship_vector.detach().numpy().T)
            # sim2 = self.cos_sim(a=vec2.detach().numpy(), b=relationship_vector.detach().numpy().T)

            # print(word1[0], sim1)
            # print(word2[0], sim2)
            re_ = {}
            for v in vocabs:
                re_[(word1[0], word2[0], v,1)] = self.cos_sim(re[v].detach().numpy(),relationship_vector.detach().numpy().T).tolist()
                re_[(word1[0], word2[0], v,2)] = self.cos_sim(re[v].detach().numpy(),relationship_vector2.detach().numpy().T).tolist()

            
            in_p[word1[0]+'_'+word2[0]] = list(sorted(re_.items(), key=lambda item: item[1], reverse=True))[:10]
            # print('relationships',sim)
    
        return in_p




            
    