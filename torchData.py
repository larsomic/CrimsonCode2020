import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torchtext import vocab
from torchtext import datasets

import spacy
import numpy as np

import time
import random


TEXT = data.Field(lower = True, include_lengths=True, batch_first=True)
UD_TAGS = data.Field(unk_token= None)
PTB_TAGS = data.Field(unk_token = None)

fields = (("text", TEXT), ("udtags", UD_TAGS), ("ptbtags", PTB_TAGS))
train_data, valid_data, test_data = datasets.UDPOS.splits(fields)

MIN_FREQ = 2

TEXT.build_vocab(train_data, min_freq=MIN_FREQ, vectors = 'glove.6B.100d',unk_init = torch.Tensor.normal_)
UD_TAGS.build_vocab(train_data)
PTB_TAGS.build_vocab(train_data)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = 128,
    device = device)

