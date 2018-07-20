#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 06:48:37 2018

@author: aadarsh
"""

import torch
import nltk
import re
import numpy as np
import torch.utils.data as data_utils
import torch.nn.functional as F
import sklearn as sk
from scipy import spatial
import os

device = torch.device('cpu')
CONTEXT_SIZE=2
EMBEDDING_SIZE=50
num_epochs = 1000
batch_size = 50
learning_rate = 0.001
is_trained=False
#data="This is a test sentence"

data="""When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold. """

def preprocess(data):
    sentences=nltk.sent_tokenize(data)
    sentences=[re.sub('[^A-Za-z0-9]+',' ',sentence) for sentence in sentences]
    sentences=[sentence.lower() for sentence in sentences]
    
    ## preprocess to delete the words which doesn't have more than 2 letters
    sentences=[list(filter(lambda word:len(word)>2,sentence.split(' '))) for sentence in sentences]
    return sentences

def generate_pair_from_sentence(sentence,CONTEXT_SIZE):
    length=len(sentence)
    
    pairs=[]
    pairs+=([(sentence[i],sentence[i+j])  for i in range(length) for j in range(-1*CONTEXT_SIZE,CONTEXT_SIZE+1) if j!=0 and i+j>=0 and i+j<length])
    return pairs

def word_to_id(vocab):
    word_to_idx={word:i for i, word in enumerate(vocab)}
    return word_to_idx

def id_to_word(word_to_id_dict):
    keys=list(word_to_id_dict.keys())
    values=list(word_to_id_dict.values())
    return dict(zip(values,keys))
    

def get_vocabulory(sentences):
    summed_vocab=[]
    for sentence in sentences:
        for word in sentence:
            summed_vocab.append(word)
        
    vocab=set(summed_vocab)
    return vocab

def word_to_onehot(word,vocab_size,word_to_id):
    idx=word_to_id[word]
    zeros=[0]*vocab_size
    zeros[idx]=1
    return zeros

def create_tensor_dataloader(XY_Pairs,vocab):
    vocab_size=len(vocab)
    word_id_dict=word_to_id(vocab)
    X_Train=[]
    Y_Train=[]
    
    for pair in XY_Pairs:
        X_Train+=word_to_onehot(pair[0],vocab_size,word_id_dict)
        Y_Train+=word_to_onehot(pair[1],vocab_size,word_id_dict)
    
    X_Train=np.array(X_Train).reshape(-1,vocab_size)
    Y_Train=np.array(Y_Train).reshape(-1,vocab_size)
    
    print(X_Train.shape)
    print(Y_Train.shape)
    Dataset = data_utils.TensorDataset(torch.tensor(X_Train,dtype=torch.float), torch.tensor(Y_Train,dtype=torch.long))
    DataLoader = data_utils.DataLoader(Dataset, batch_size=batch_size, shuffle=True)
    
    return DataLoader

def generate_all_pairs(sentences):
    training_pairs=[]
    for sentence in sentences:
        training_pairs+=generate_pair_from_sentence(sentence,CONTEXT_SIZE)
    return training_pairs
    

def get_nearest_word_idx(emb_mat,word_idx):
    word_vec=emb_mat[:,word_idx]
    most_simi_idx=0;
    min_distance=1
    for i in range(emb_mat.shape[1]):
        if i!=word_idx:
            distance = spatial.distance.cosine(word_vec, emb_mat[:,i])
            if(distance<min_distance):
                min_distance=distance
                most_simi_idx=i
                
    return most_simi_idx

class skipgram(torch.nn.Module):
    def __init__(self,vocab_size,embedding_size):
        super(skipgram,self).__init__();
        self.hidden_layer=torch.nn.Linear(vocab_size,embedding_size)
        self.output=torch.nn.Linear(embedding_size,vocab_size)
        
    def forward(self,inputs):
        hidden=self.hidden_layer(inputs)
        
        out=self.output(hidden)
        softmax_out=F.log_softmax(out,dim=0)
        return softmax_out
    
    def get_embedded_mat(self):
        return self.hidden_layer.weight.data
    
    
if __name__=='__main__':
    sentences=preprocess(data)
    vocab=get_vocabulory(sentences)
    pairs=generate_all_pairs(sentences)
    word_to_id_dict=word_to_id(vocab)
    id_to_word_dict=id_to_word(word_to_id_dict)
    
    dataloader=create_tensor_dataloader(pairs,vocab)
    vocab_size=len(vocab)
    
    model=skipgram(vocab_size,EMBEDDING_SIZE)
    
    model_name='model.ckpt'
    for f in os.listdir():
        if f==model_name:
            model.load_state_dict(torch.load(model_name))
            is_trained=True;
    
    if not is_trained:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        total_step = len(dataloader)
        for epoch in range(num_epochs):
            for i,(X_Train,Y_Train) in enumerate(dataloader):
                X_Train=X_Train.reshape(-1,vocab_size).to(device)
                Y_Train=Y_Train.reshape(-1,vocab_size).to(device)
                
                output=model(X_Train)
                loss=criterion(output,torch.max(Y_Train,1)[1])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i%2==0:
                    print('Epoch {}, Batch {}, Loss {:.4f}'.format(epoch,i,loss.item()))
                    if(i%10==0):
                        torch.save(model.state_dict(), 'model.ckpt')
                    
    embedded_mat=model.get_embedded_mat();
    
    #Predicting the most closest word 
    print(id_to_word_dict[get_nearest_word_idx(embedded_mat,word_to_id_dict['asked'])])
                    


            
    
    
    
    
        
        
    