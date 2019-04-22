import os, sys, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LSTMNet(nn.Module):
    def __init__(self, opt):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        # comment notation
        #  b: batch size, s: sequence length, e: embedding dim, c : num of class
        super(LSTMNet, self).__init__()
        # Define the layers
        self.embed_x = nn.Embedding(opt.n_words, opt.embed_size) 
        # self.embed_y = nn.Embedding(opt.num_class, opt.embed_size)
        # self.att_conv = nn.Conv1d(opt.num_class,opt.num_class,kernel_size=opt.ngram,padding=opt.ngram//2)


        # self.word_embedding = nn.Embedding(opt.n_words, opt.embed_size)
        self.lstm = nn.LSTM(opt.embed_size, opt.embed_size)

        self.dropout = nn.Dropout(opt.dropout)
        self.H1_x = nn.Linear(opt.embed_size, opt.H_dis)
        self.H2_x = nn.Linear(opt.H_dis, opt.num_class)

        # Init the weights
        self.embed_x.weight.data = self.embed_x.weight.data + torch.tensor(opt.W_emb)
        
    
    def forward(self, x, opt):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary (differentiable) operations on Tensors.
        """
        # Embeddings
        embedded = self.embed_x(x) 
        
        # H_enc = torch.sum(embedded, dim=1)
        # H_enc = torch.squeeze(H_enc)
        
        # lstm_out, _ = self.lstm(H_enc)

        # # 2 layer nn classification
        # H1_out_x = self.H1_x(self.dropout(lstm_out))
        # logits = self.H2_x(self.dropout(H1_out_x))
        
        # return logits 

        lstm_out, _ = self.lstm(embedded.view(len(x), 1, -1))
        # 2 layer nn classification
        H1_out_x = self.H1_x(self.dropout(lstm_out))
        logits = self.H2_x(self.dropout(H1_out_x))
        
        return logits 

        # tag_space = self.hidden2tag(lstm_out.view(len(x), -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        # return tag_scores


class RNN(nn.Module):
    def __init__(self, opt):
        
        super().__init__()
        hidden_dim = 300

        self.embedding = nn.Embedding(opt.n_words, opt.embed_size)
        
        self.rnn = nn.LSTM(opt.embed_size, hidden_dim)
        
        self.H1_x = nn.Linear(opt.embed_size, opt.H_dis)
        self.H2_x = nn.Linear(opt.H_dis, opt.num_class)

        self.dropout = nn.Dropout(opt.dropout)

        self.fc = nn.Linear(hidden_dim, 4)
        
    def forward(self, text, opt):

        #text = [sent len, batch size]
        # print(text.shape)
        embedded = self.embedding(text)
        # print(embedded.shape)
        #embedded = [sent len, batch size, emb dim]

        output, hidden = self.rnn(embedded)

        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        # assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        # print(output.shape, hidden.shape)
        H_enc = torch.sum(output, dim=1)
        H_enc = torch.squeeze(H_enc)

        H1_out_x = self.H1_x(self.dropout(H_enc))
        H1_out_x = nn.ReLU()(H1_out_x)
        # print(H1_out_x.shape)
        logits = self.H2_x(self.dropout(H1_out_x))
        logits = nn.Sigmoid()(logits)
        # print(logits.shape)

        return logits
