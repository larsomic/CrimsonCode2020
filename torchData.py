import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
    
from math import sqrt

from torchtext import data
from torchtext import vocab
from torchtext import datasets

import spacy
import numpy as np

import time
import random


#Most of this code was written with the help of https://github.com/bentrevett/pytorch-pos-tagging/blob/master/1%20-%20BiLSTM%20PoS%20Tagger.ipynb


#Change based on system
device = torch.device('cpu')

#Parse the data into "fields"
TEXT = data.Field(lower = True)
UD_TAGS = data.Field(unk_token= None)
PTB_TAGS = data.Field(unk_token = None)

#Split the data set into train and test data
fields = (("text", TEXT), ("udtags", UD_TAGS), ("ptbtags", PTB_TAGS))
train_data,test_data = datasets.UDPOS.splits(fields)

MIN_FREQ = 2
#Build the vocab using a wikipedia vector and initialize with tensors
TEXT.build_vocab(train_data, min_freq=MIN_FREQ, vectors = 'glove.6B.100d',unk_init = torch.Tensor.normal_)
UD_TAGS.build_vocab(train_data)
PTB_TAGS.build_vocab(train_data)

#Define the iterator to access data
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data), 
    batch_size = 1,
    device = device)
'''
This was the hardest part of the code. After previously working with Keras, the module format
was a bit hard to learn at first. Linear is essentially the Dense function and LSTM is the same
'''

class BILSTM (nn.Module):
    def __init__ (self, input_dim, embed_dim, hid_dim, out_dim, layers, dropout, pad_idk):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=pad_idk)
        self.lstm = nn.LSTM (embed_dim, hid_dim, layers, bidirectional= True, dropout = dropout if layers > 1 else 0)
        self.fc = nn.Linear(hid_dim *2,out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward (self, text):
        embedded = self.dropout(self.embedding(text))
        outputs, (hidden,cell) = self.lstm(embedded)
        predictions = self.fc(self.dropout(outputs))
        return predictions



inputd = len(TEXT.vocab)
embed = 100     #corresponds to the number of tensors
hidden = 1      #these are currently low for performance reasons
layers = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

testnetwork = BILSTM(inputd, embed, hidden, len(UD_TAGS.vocab),layers, .1, PAD_IDX) 
for name, param in testnetwork.named_parameters():
    nn.init.normal_(param.data, mean = 0, std = 0.1)

pretrained_embeddings = TEXT.vocab.vectors
testnetwork.embedding.weight.data.copy_(pretrained_embeddings)
testnetwork.embedding.weight.data[PAD_IDX] = torch.zeros (embed)

optimizer = optim.Adam(testnetwork.parameters())

TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index= TAG_PAD_IDX)

testnetwork = testnetwork.to(device)
criterion = criterion.to(device)

#Training
def train(testnetwork):
    epoch_loss = 0

    testnetwork.train()
    for batch in train_iterator:
        text = batch.text
        tags = batch.udtags

        optimizer.zero_grad()
        predictions = testnetwork(text)
        predictions = predictions.view (-1, predictions.shape[-1])

        tags = tags.view(-1)
        loss = criterion(predictions, tags)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return(epoch_loss/len(train_iterator))


#Evaluate
def evaluate(testnetwork):
    epoch_loss = 0
    testnetwork.eval()

    with torch.no_grad():
        for batch in test_iterator:
            text = batch.text
            tags = batch.udtags

            predictions = testnetwork(text)
            predictions = predictions.view(-1,predictions.shape[-1])

            loss = criterion (predictions, tags)
            epoch_loss += loss.item()


        return(epoch_loss/len(test_iterator))


N_EPOCHS = 10
for epoch in range (1,N_EPOCHS):
    train_loss = train(testnetwork)
    test_loss = evaluate(testnetwork)
    print (train_loss, test_loss)


def testRealSentence(testnetwork,text_field,sentence,tag_field):
    testnetwork.eval()
    if (isinstance(sentence,str)):
        nlp = spacy.load('en')
        tokens = [token.text for token in nlp(sentence)]
    else:
        tokens = [token for token in sentence]

    if text_field.lower:
        tokens = [t.lower() for t in tokens]
        
    numericalized_tokens = [text_field.vocab.stoi[t] for t in tokens]
    unk_idx = text_field.vocab.stoi[text_field.unk_token]
    unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
    token_tensor = torch.LongTensor(numericalized_tokens)
    token_tensor = token_tensor.unsqueeze(-1).to(device)        
    predictions = testnetwork(token_tensor)
    top_predictions = predictions.argmax(-1)    
    predicted_tags = [tag_field.vocab.itos[t.item()] for t in top_predictions]    
    return tokens, predicted_tags, unks


sentence = "Here is a common sentence that is used frequentely."

tokens, predicted_tags, unks = testRealSentence(testnetwork,TEXT,sentence,UD_TAGS)
print (tokens, predicted_tags, unks)

pd = Dataframe(zip(tokens, predicted_tags))
