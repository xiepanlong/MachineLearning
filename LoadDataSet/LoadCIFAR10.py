#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:xiepanlong 
@file: LoadCIFAR10.py 
@time: 2017/11/{DAY} 
"""

import cPickle
import numpy as np
import os

def load_CIFAR_batch(fileName):
    """ load single batch of cifar """
    with open(fileName, 'rb') as f:
        dataDict = cPickle.load(f)
        X = dataDict['data']
        Y = dataDict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    print ('Xtr: ', Xtr.shape, 'Ytr:', Ytr.shape, 'Xte:', Xte.shape, 'Yte:', Yte.shape)
    return Xtr, Ytr, Xte, Yte

if __name__ == '__main__':
    path = '/Users/xiepanlong/Documents/DeepLearning/DataSet/cifar-10-batches-py/'
    Xtr, Ytr , Xte, Yte = load_CIFAR10(path)
    print len(Xtr), len(Ytr)
    print Xtr[0], Ytr[0]