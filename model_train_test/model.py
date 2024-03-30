# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:06:58 2024

@author: tianchilu4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold
import collections
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, precision_score
import random
from sklearn.metrics import auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#import esm
from tqdm import tqdm
import time
import seaborn as sns
from argparse import Namespace
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, matthews_corrcoef, recall_score, f1_score, precision_score
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True



class TransHLA(nn.Module):
    def __init__(self,parameters):
        super(TransHLA, self).__init__()
        max_len = parameters.max_len
        n_layers = parameters.n_layers
        n_head = parameters.n_head
        d_model = parameters.d_model
        d_ff = parameters.dim_feedforward
        cnn_padding_index = parameters.cnn_padding_index
        cnn_num_channel = parameters.cnn_num_channel
        region_embedding_size = parameters.region_embedding_size
        cnn_kernel_size = parameters.cnn_kernel_size
        cnn_padding_size = parameters.cnn_padding_size
        cnn_stride = parameters.cnn_stride
        pooling_size = parameters.pooling_size

        self.esm, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.region_cnn1 = nn.Conv1d(
            d_model, cnn_num_channel, region_embedding_size)
        self.region_cnn2 = nn.Conv1d(
            max_len, cnn_num_channel, region_embedding_size)
        self.padding1 = nn.ConstantPad1d((1, 1), 0)
        self.padding2 = nn.ConstantPad1d((0, 1), 0)
        self.relu = nn.ReLU()
        self.cnn1 = nn.Conv1d(cnn_num_channel, cnn_num_channel, kernel_size=cnn_kernel_size,
                              padding=cnn_padding_size, stride=cnn_stride)
        self.cnn2 = nn.Conv1d(cnn_num_channel, cnn_num_channel, kernel_size=cnn_kernel_size,
                              padding=cnn_padding_size, stride=cnn_stride)
        self.maxpooling = nn.MaxPool1d(kernel_size=pooling_size)
        self.transformer_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layers, num_layers=n_layers)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(cnn_num_channel)
        self.bn3 = nn.BatchNorm1d(cnn_num_channel)
        self.fc_task = nn.Sequential(
            nn.Linear(d_model+2*cnn_num_channel, d_model // 4),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(d_model // 4, 64),
        )
        self.classifier = nn.Linear(64, 2)

    def cnn_block1(self, x):
        return self.cnn1(self.relu(x))

    def cnn_block2(self, x):
        x = self.padding2(x)
        px = self.maxpooling(x)
        x = self.relu(px)
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.cnn1(x)
        x = px + x
        return x

    def structure_block1(self, x):
        return self.cnn2(self.relu(x))

    def structure_block2(self, x):
        x = self.padding2(x)
        px = self.maxpooling(x)
        x = self.relu(px)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x = px + x
        return x

    def forward(self, x_in):
        with torch.no_grad():
            results = self.esm(x_in, repr_layers=[33], return_contacts=True)
            emb = results["representations"][33]
            structure_emb = results["contacts"]
        output = self.transformer_encoder(emb)
        representation = output[:, 0, :]
        representation = self.bn1(representation)
        cnn_emb = self.region_cnn1(emb.transpose(1, 2))
        cnn_emb = self.padding1(cnn_emb)
        conv = cnn_emb + self.cnn_block1(self.cnn_block1(cnn_emb))
        while conv.size(-1) >= 2:
            conv = self.cnn_block2(conv)
        cnn_out = torch.squeeze(conv, dim=-1)
        cnn_out = self.bn2(cnn_out)

        structure_emb = self.region_cnn2(structure_emb.transpose(1, 2))
        structure_emb = self.padding1(structure_emb)
        structure_conv = structure_emb + \
            self.structure_block1(self.structure_block1(structure_emb))
        while structure_conv.size(-1) >= 2:
            structure_conv = self.structure_block2(structure_conv)
        structure_cnn_out = torch.squeeze(structure_conv, dim=-1)
        structure_cnn_out = self.bn3(structure_cnn_out)
        representation = torch.concat(
            (representation,cnn_out,structure_cnn_out), dim=1)
        reduction_feature = self.fc_task(representation)
        reduction_feature = reduction_feature.view(
            reduction_feature.size(0), -1)
        logits_clsf = self.classifier(reduction_feature)
        logits_clsf = torch.nn.functional.softmax(logits_clsf, dim=1)
        return logits_clsf, reduction_feature
    
    


class TextCNN(nn.Module):
    def __init__(self, vocab_size=40, embedding_dim=128, windows_size=[2,4,3], max_len=21, feature_size=256, n_class=2, dropout=0.4):
        super(TextCNN, self).__init__()
        # embedding层
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # 卷积层特征提取
        self.conv1 = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=embedding_dim, out_channels=feature_size, kernel_size=h),
                          nn.LeakyReLU(),
                          nn.MaxPool1d(kernel_size=max_len-h+1),
                          )
            for h in windows_size]
        )
        # 全连接层
        self.fc = nn.Linear(feature_size*len(windows_size), n_class)
        # dropout防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embed(x) # [batch, seq_len, embed_dim]
        x = x.permute(0, 2, 1) # [batch, embed_dim, seq_len]
        x = [conv(x) for conv in self.conv1]
        x = torch.cat(x, 1)
        x = x.view(-1, x.size(1)) # [batch, feature_size*len(windows_size)]
        x = self.dropout(x)
        representation=x
        x = self.fc(x)# [batch, n_class]
        return x,representation



#-*- coding:utf-8 -*-






 
class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d,self).__init__()
 
    def forward(self, x):
        return torch.max_pool1d(x,kernel_size=x.shape[-1])
 
    
 
    
 
    
 
class TextRCNN(nn.Module):
    def __init__(self,vocab_size=40,embedding_dim=128,hidden_size=50,num_labels=2):
        super(TextRCNN,self).__init__()
        self.embed = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,
                            batch_first=True,bidirectional=True)
        self.globalmaxpool = GlobalMaxPool1d()
        self.dropout = nn.Dropout(.5)
        self.linear1 = nn.Linear(embedding_dim+2*hidden_size,256)
        self.linear2 = nn.Linear(256,num_labels)
 
    def forward(self, x):#x: [batch,L]
        x_embed = self.embed(x) #x_embed: [batch,L,embedding_size]
        last_hidden_state,(c,h) = self.lstm(x_embed) #last_hidden_state: [batch,L,hidden_size * num_bidirectional]
        out = torch.cat((x_embed,last_hidden_state),2)#out: [batch,L,embedding_size + hidden_size * num_bidirectional]
        #print(out.shape)
        out = F.relu(self.linear1(out))
        out = out.permute(dims=[0,2,1]) #out: [batch,embedding_size + hidden_size * num_bidirectional,L]
        representation = self.globalmaxpool(out).squeeze(-1) #out: [batch,embedding_size + hidden_size * num_bidirectional]
        #print(out.shape)
        out = self.dropout(representation) #out: [batch,embedding_size + hidden_size * num_bidirectional]
        out = self.linear2(out) #out: [batch,num_labels]
        return out,representation







class RNN_ATTs(nn.Module):
    def __init__(self, vocab_size=40, embedding_dim=256, hidden_dim=128, output_dim=2,
                 n_layers=2, bidirectional=True, dropout=0.2, pad_idx=0, hidden_size2=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(hidden_dim * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_size2)
        self.fc = nn.Linear(hidden_size2, output_dim)

    def forward(self, x):
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)
        #representation = out# [128, 256]
        out = F.relu(out)
        representation = self.fc1(out)
        out = self.fc(representation)  # [128, 64]
        return out, representation






DPargs = Namespace(
    num_vocab=1000,
    embedding_dim=256,
    padding_index=0,
    region_embedding_size=3,             
    cnn_num_channel=256,
    cnn_kernel_size=3,              
    cnn_padding_size=1,
    cnn_stride=1,
    pooling_size=2,         
    num_classes=2,
)





class DPCNN(nn.Module):
    def __init__(self):
        super(DPCNN, self).__init__()
        self.embedding = nn.Embedding(DPargs.num_vocab, DPargs.embedding_dim, padding_idx=DPargs.padding_index)
        self.region_cnn = nn.Conv1d(DPargs.embedding_dim, DPargs.cnn_num_channel, DPargs.region_embedding_size)
        self.padding1 = nn.ConstantPad1d((1, 1), 0)          
        self.padding2 = nn.ConstantPad1d((0, 1), 0)       
        self.relu = nn.ReLU()
        self.cnn = nn.Conv1d(DPargs.cnn_num_channel, DPargs.cnn_num_channel, kernel_size=DPargs.cnn_kernel_size,
                             padding=DPargs.cnn_padding_size, stride=DPargs.cnn_stride)
        self.maxpooling = nn.MaxPool1d(kernel_size=DPargs.pooling_size)
        self.fc = nn.Linear(DPargs.cnn_num_channel, DPargs.num_classes)
    def _block1(self, x):
        return self.cnn(self.relu(x))          

    def _block2(self, x):
        x = self.padding2(x)
        px = self.maxpooling(x)
        x = self.relu(px)
        x = self.cnn(x)
        x = self.relu(x)
        x = self.cnn(x)
        x = px + x
        return x
    
    
    def forward(self, x_in):
        emb = self.embedding(x_in)           
        emb = self.region_cnn(emb.transpose(1, 2))         
        emb = self.padding1(emb)         

        conv = emb + self._block1(self._block1(emb))

        while conv.size(-1) >= 2:
            conv = self._block2(conv)      
        representation = torch.squeeze(conv, dim=-1)
        out = torch.nn.functional.softmax(self.fc(torch.squeeze(conv, dim=-1)),dim=1)
        return out, representation


