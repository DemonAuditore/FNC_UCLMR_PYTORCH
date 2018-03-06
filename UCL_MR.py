import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

import torch.utils.data as Data
import torchvision

import numpy as np
from torch.utils.data import TensorDataset
import random
import time

import sys

# torch.manual_seed(1)    # reproducible


class UCL_NN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(UCL_NN, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer
        self.softmax = nn.Softmax()
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.WD = l2_alpha

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        output = self.softmax(x)
        return output

    def data_process(self, train_dataset):
        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        return train_loader

    def train(self, train_dataset):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.LR, weight_decay=self.WD)
        loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

        train_loader = self.data_process(train_dataset)

        for epoch in range(self.EPOCH):
            for stem, (x, y) in enumerate(train_loader):
                b_x = Variable(x)  # batch x
                b_y = Variable(y)  # batch y

                output = self.forward(b_x)  # cnn output

                loss = loss_func(output, b_y)  # mean squared error loss
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()

    def predict(self, test_data):
        test_output = self.forward(test_data).data.numpy()[0]
        return test_output


lim_unigram = 1
n_feature = 2*lim_unigram+1
hidden_size = 10
n_output = 2
EPOCH = 90
BATCH_SIZE = 10
LR = 0.02
l2_alpha = 0.0001
clip_ratio = 5
train_ratio = 0.8

if __name__ == "__main__":

    # make fake data
    n_data = torch.ones(100, n_feature)
    x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
    y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
    x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
    y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
    data_x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
    data_y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

    # torch can only train on Variable, so convert them to Variable
    train_dataset = TensorDataset(data_x, data_y)
    # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
    # plt.show()



    ucl_nn = UCL_NN(n_feature, hidden_size, n_output)     # define the network
    print(net)  # net architecture

    net.train(train_dataset)
