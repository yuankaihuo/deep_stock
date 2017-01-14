# -*- coding: utf-8 -*-
"""
Created on Jan 2017

@author: Yuankai Huo @ Vanderbilt University
"""

import pandas
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout, Embedding, Activation, Convolution1D, MaxPooling1D
import numpy as np
import glob, os


# binarize the testing data to see if the price of the 5th future day increases or decreases 
def binarize_data(trainX,trainY,col=4):
    trainY_binary = np.zeros((trainY.shape[0],1))
    for i in range(0,trainY.shape[0]):
        if trainX.ndim == 2:
            lastNum = trainX[i][-1]
        else:
            lastNum = trainX[i][-1][0]
        YNum = trainY[i][col]
        if YNum>=lastNum:
            trainY_binary[i][0] = 1
        else:
            trainY_binary[i][0] = 0
    return trainY_binary

# normlize each path to [0,1]
def normalize_input(trainX,trainY):
    for i in range(0,trainX.shape[0]):
        rawRow = np.concatenate((trainX[i,:,0],trainY[i,:]))
        normRow = (rawRow-min(rawRow))/(max(rawRow)-min(rawRow))
        trainX[i,:,0] = normRow[0:trainX.shape[1]]
        trainY[i,:] = normRow[trainX.shape[1]:trainX.shape[1]+trainY.shape[1]]
    return trainX,trainY
    
# generate training data for one company
def generate_one_train(oneTrain,examples,y_examples,interval):
    data = np.asarray(oneTrain)
    nb_samples = len(data) - examples - y_examples
    inputX_list = [np.expand_dims(np.atleast_2d(data[i:examples+i,]), axis=2) for i in range(0,nb_samples,interval)]
    inputY_list = [np.atleast_2d(data[i+examples:examples+i+y_examples,]) for i in range(0,nb_samples,interval)]
    
    try:
        inputX_mat = np.concatenate(inputX_list, axis=0)
        inputY_mat = np.concatenate(inputY_list, axis=0)
    except:
        print('error')
    return inputX_mat,inputY_mat

# generate training data for all companies
def generate_all_train(Train,look_back,predict_len,interval):
    TrainX_list = []
    TrainY_list = []
    for i in range(len(Train)):
        oneTrain = Train[i]
        if len(oneTrain)-look_back-predict_len>0:
            inputX_mat,inputY_mat = generate_one_train(oneTrain,look_back,predict_len,interval)
            TrainX_list.append(inputX_mat)
            TrainY_list.append(inputY_mat)
    TrainX_mat = np.concatenate(TrainX_list,axis=0)
    TrainY_mat = np.concatenate(TrainY_list,axis=0)
    return TrainX_mat,TrainY_mat

# read particular column from csv file
def read_csv(path,colNum):
    dataframe = pandas.read_csv(path, usecols=[colNum],engine='python',header=None)
    dataset = dataframe.values.astype(float)
    return dataset

# read all raw data from csv files
def read_raw_data(files):
    Train = []
    Test = []    
    for fi in files:
        times = read_csv(fi,0)
        prices = read_csv(fi,5)
        trainSingle = []
        testSingle = []
        for i in range(0,len(times)):
            time = times[i][0]
            if time <= 20091231:
                trainSingle.append(prices[i][0])
            else:
                testSingle.append(prices[i][0])   
        Train.append(trainSingle)
        Test.append(testSingle)
    return Train,Test
    

# generate train and test dataset from csv files
def generate_dataset(csv_dir, look_back, predict_len, interval_train=1, interval_test=1):
    files = glob.glob(os.path.join(csv_dir,"*.csv"))
    files.sort()        
    print('=== Start generate train and test dataset from csv files ===')
    train_fileX = ('trainDataX_interval%d.npy' % interval_train)
    train_fileY = ('trainDataY_interval%d.npy' % interval_train)
    test_fileX = ('testDataX_interval%d.npy' % interval_test)
    test_fileY = ('testDataY_interval%d.npy' % interval_test)
    if os.path.exists(train_fileX) and os.path.exists(train_fileY) and os.path.exists(test_fileX) and os.path.exists(test_fileY):
        trainX = np.load(train_fileX)
        trainY = np.load(train_fileY)
        testX = np.load(test_fileX)
        testY = np.load(test_fileY)
    else:
        Train,Test = read_raw_data(files)
        trainX,trainY = generate_all_train(Train,look_back,predict_len,interval_train)
        trainX, trainY = normalize_input(trainX,trainY)
        trainX = np.float32(trainX)
        trainY = np.float32(trainY) 
        np.save(train_fileX,trainX)
        np.save(train_fileY,trainY)
        testX,testY = generate_all_train(Test,look_back,predict_len,interval_test)
        testX,testY = normalize_input(testX,testY)
        testX = np.float32(testX)
        testY = np.float32(testY) 
        np.save(test_fileX,testX)
        np.save(test_fileY,testY)
    print('=== Finish generate train and test dataset from csv files ===')
    return trainX, trainY, testX, testY
    

# train lstm RNN network
def lstm_train(trainX, trainY, look_back, predict_len, epoch=1, verbose=2):
    print('=== Start training LSTM-RNN network ===')      
    features = trainX.shape[2]
    hidden = 128
    trainY_binary = binarize_data(trainX,trainY,4)
    model = Sequential()        
    model.add(LSTM(hidden,input_shape=(look_back,features)))
    model.add(Dropout(.2))
    model.add(Dense(trainY_binary.shape[1]))        
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.fit(trainX, trainY_binary, nb_epoch=epoch, verbose=verbose)    
    print('=== Finish training LSTM-RNN network ===')  
    return model

# test lstm RNN network   
def lstm_test(model, testX, testY):
    print('=== Start testing LSTM-RNN network ===')   
    test_trials = testX.shape[0]
    testY_binary = binarize_data(testX,testY,4)
    predictY = model.predict(testX)
    predictY_binary = (predictY>0.5)        
    acc = float(sum(testY_binary[:,0]==predictY_binary[:,0]))/test_trials
    print('=== Finish testing LSTM-RNN network ===')   
    return acc

# train CNN + lstm RNN network    
def cnn_lstm_train(trainX, trainY, look_back, predict_len, epoch=1,verbose=2):
    print('=== Start training CNN + LSTM-RNN network ===')      
    trials = trainX.shape[0]
    features = trainX.shape[2]
    hidden = 128
#    trainX = trainX.reshape(trials,look_back)
    trainY_binary = binarize_data(trainX,trainY,4)
    model = Sequential()
#    model.addEm(bedding(trials, 128, input_length=look_back))
    
    model.add(Convolution1D(nb_filter=64,
                            filter_length = 5,
                            border_mode = 'valid',
                            activation = 'relu',
                            input_shape = (look_back,features)))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Dropout(0.2))
    model.add(LSTM(hidden))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.fit(trainX, trainY_binary, nb_epoch=epoch, verbose=verbose) 
    print('=== Finish training CNN + LSTM-RNN network ===')             
    return model

# test CNN + lstm RNN network                 
def cnn_lstm_test(model, testX, testY):
    print('=== Start testing CNN + LSTM-RNN network ===')  
    test_trials = testX.shape[0]
    test_times = testX.shape[1]
#    testX = testX.reshape(test_trials,test_times)
    testY_binary = binarize_data(testX,testY,4)
    predictY = model.predict(testX)
    predictY_binary = (predictY>0.5)        
    acc = float(sum(testY_binary[:,0]==predictY_binary[:,0]))/test_trials
    print('=== Start testing CNN + LSTM-RNN network ===')  
    return acc
            

