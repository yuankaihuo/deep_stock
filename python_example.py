# -*- coding: utf-8 -*-
"""
Created on Jan 2017

@author: Yuankai Huo @ Vanderbilt Univerisity
"""


import deep_stock as ds

# the directory that save the csv tables. (contains all csv filesfrom table1,,2 and 3)
csv_dir = '/fs4/masi/huoy1/DeepLearning/stock/tables/tables'
# parameter setting
look_back       = 90 # look back to past 90 days
predict_len     = 5  # predict the 5th day in the future 
sample_interval = 1  # intervals of downsampling training data
train_epoch     = 1 # epoches of training

# read training and testing dataset
trainX,trainY,testX,testY = ds.generate_dataset(csv_dir, look_back, predict_len, sample_interval)
print("size of train = [%d %d %d]" % (trainX.shape[0],trainX.shape[1],trainX.shape[2]))
print("size of test = [%d %d]" % (testX.shape[0],testX.shape[1]))

# train network 1 (lstm RNN only)    
lstm_model = ds.lstm_train(trainX, trainY, look_back, predict_len,train_epoch,1)
lstm_acc   = ds.lstm_test(lstm_model, testX, testY)
print("LSTM RNN network's accuracy = %f\n" % lstm_acc)

# train network 2 (CNN + lstm RNN only)    
cnn_lstm_model = ds.cnn_lstm_train(trainX, trainY, look_back, predict_len, train_epoch,1)
cnn_lstm_acc   = ds.cnn_lstm_test(cnn_lstm_model, testX, testY)
print("CNN + LSTM RNN network's accuracy = %f\n" % cnn_lstm_acc)