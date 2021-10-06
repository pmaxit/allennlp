from torch.utils.data import Dataset
import torch.nn.functional as F
from collections import Counter
from os.path import exists
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import math
import re



### Transformer architecture
def attention(q, k, v , mask = None, dropout=None):
    scores = q.matmul(k.transpose(-2,-1))
    scores /= math.sqrt(q.shape[-1])
    
    # mask
    scores = scores if mask is None else scores.masked_fill(mask == 0, -1e3)
    
    scores = F.softmax(scores, dim=-1)
    
    scores = dropout(scores) if dropout is not None else scores
    output = scores.matmul(v)
    
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, out_dim, dropout=0.1):
        super().__init__()
        
        self.linear = nn.Linear(out_dim, out_dim*3)
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.out_dim_per_head = out_dim // n_heads
        self.out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, t):
        return t.reshape(t.shape[0], -1, self.n_heads, self.out_dim_per_head)
    
    def forward(self, x, y=None, mask=None):
        # in decoder, y comes from encoder, In encoder y = x
        y = x if y is None else y
        
        qkv = self.linear(x)      # BS * seq_len * (3*EMBED_SIZE_L)
        q = qkv[:,:,:self.out_dim]
        k = qkv[:,:,self.out_dim:self.out_dim*2]
        v = qkv[:,:,self.out_dim*2:]
        
        # break into heads
        q, k , v = [self.split_heads(t) for t in (q,k,v)]   # BS * SEQ_LEN * HEAD * EMBED_SIZE_P_HEAD
        q, k, v = [t.transpose(1,2) for t in (q,k , v)]     # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
        
        # n_heads => attention => merge the heads => mix information
        scores = attention(q, k , v, mask, self.dropout)      # BS * HEAD * SEQ_LEN * EMBED_SIZE_PHEAD
        scores = scores.transpose(1,2 ).contiguous().view(scores.shape[0], -1, self.out_dim) # BS * SEQ_LEN * EMBED_SIZE_L
        out = self.out(scores)
        
        
        return out
    
class Feedforward(nn.Module):
    def __init__(self, inp_dim, inner_dim, dropout=0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(inp_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, inp_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
    
class EncoderLayer(nn.Module):
    def __init__(self, n_heads, inner_transformer_size, inner_ff_size, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, inner_transformer_size, dropout)
        self.ff = Feedforward(inner_transformer_size, inner_ff_size, dropout)
        self.norm1 = nn.LayerNorm(inner_transformer_size)
        self.norm2 = nn.LayerNorm(inner_transformer_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.mha(x2, mask=mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x

        