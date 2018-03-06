# coding:utf-8

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
from utils.dataset import DataSet
from utils.score import report_score, LABELS, score_submission

import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        optimizer = torch.optim.SGD(self.parameters(), lr=self.LR, weight_decay=self.WD)
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
        test_output = self.forward(test_data)
        prediction = torch.max(test_output, 1)[1]
        return prediction

def normalize_word(w):
    return _wnl.lemmatize(w).lower()

def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

def word_tf_features(headlines, bodies, length):

    X = []

    # Create vectorizers and BOW and TF arrays for train set
    bow_vectorizer = CountVectorizer(max_features=lim_unigram, stop_words=feature_extraction.text.ENGLISH_STOP_WORDS)
    bow = bow_vectorizer.fit_transform(headlines + bodies) # Train set only

    tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)
    tfreq = tfreq_vectorizer.transform(bow).toarray()  # Train set only

    tfidf_vectorizer = TfidfVectorizer(max_features=lim_unigram, stop_words=feature_extraction.text.ENGLISH_STOP_WORDS).fit(headlines + bodies)  # Train and test sets

    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        head_tf = tfreq[i].reshape(1, -1)
        body_tf = tfreq[i+length].reshape(1, -1)
        head_tfidf = tfidf_vectorizer.transform([headline]).toarray()
        body_tfidf = tfidf_vectorizer.transform([body]).toarray()
        tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
        features = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
        X.append(features)

    return X



def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X,y

lim_unigram = 5000
n_feature = 2*lim_unigram+1
hidden_size = 100
n_output = 4
EPOCH = 90
BATCH_SIZE = 500
LR = 0.01
l2_alpha = 0.0001
total_len = 49972
train_ratio = 0.8
train_len = int(total_len * train_ratio)
dev_len = int(total_len - train_len - 1)

_wnl = nltk.WordNetLemmatizer()

if __name__ == "__main__":

    # # make fake data
    # n_data = torch.ones(100, n_feature)
    # x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
    # y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
    # x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
    # y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
    # data_x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
    # data_y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer
    #
    # # torch can only train on Variable, so convert them to Variable
    # train_dataset = TensorDataset(data_x, data_y)
    # # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
    # # plt.show()



    # ucl_nn = UCL_NN(n_feature, hidden_size, n_output)     # define the network
    # print(ucl_nn)  # net architecture
    #
    # ucl_nn.train(train_dataset)



    dataset = DataSet()
    h_train, b_train, y_train = [], [], []
    for train_stance in dataset.stances[:train_len]:
        y_train.append(LABELS.index(train_stance['Stance']))
        h_train.append(train_stance['Headline'])
        b_train.append(dataset.articles[train_stance['Body ID']])

    X = word_tf_features(h_train, b_train, train_len)
    print("type of tf feature: ", type(X))
    print("length of tf feature: ", len(X))

    data_x = torch.FloatTensor(X)
    data_y = torch.LongTensor(y_train)

    print("type of data_x: ", type(data_x))
    print("type of data_y: ", type(data_y))

    train_dataset = TensorDataset(data_x, data_y)

    print("type of train_dataset: ", type(train_dataset))

    ucl_nn = UCL_NN(n_feature, hidden_size, n_output)     # define the network
    print(ucl_nn)  # net architecture

    ucl_nn.train(train_dataset)


    # ==============================================
    #   dev dataset
    # ==============================================

    h_dev, b_dev, y_dev = [], [], []
    for dev_stance in dataset.stances[train_len+1:total_len]:
        y_dev.append(LABELS.index(dev_stance['Stance']))
        h_dev.append(dev_stance['Headline'])
        b_dev.append(dataset.articles[dev_stance['Body ID']])

    X_dev = word_tf_features(h_dev, b_dev, dev_len)
    print("type of tf feature: ", type(X_dev))
    print("length of tf feature: ", len(X_dev))

    data_x_dev = Variable(torch.FloatTensor(X_dev))

    predicted = ucl_nn.predict(data_x_dev).data.numpy().tolist()

    # # print score
    # dev
    actual = y_dev

    predicted = [LABELS[int(a)] for a in predicted]
    actual = [LABELS[int(a)] for a in y_dev]

    report_score(actual, predicted)


    # data_x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
    # data_y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer
    #
    # # torch can only train on Variable, so convert them to Variable
    # train_dataset = TensorDataset(data_x, data_y)
    # # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
    # # plt.show()



    # ucl_nn = UCL_NN(n_feature, hidden_size, n_output)     # define the network
    # print(ucl_nn)  # net architecture
    #
    # ucl_nn.train(train_dataset)










