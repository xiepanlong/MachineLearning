#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:xiepanlong 
@file: NearestNeighbor.py 
@time: 2017/11/{DAY} 
"""

import numpy as np

class NN(object):
    def __init__(self, X, Y):
        self.Xtr = X
        self.Ytr = Y

    def train(self, X, Y):
        self.Xtr = X
        self.Ytr = Y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.Ytr.dtype)
        print('The data %d is training...')
        for i in xrange(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            #min_index = np.argmin(distances)
            min_index = np.where(distances == np.min(distances))
            one_index = np.min(min_index)
            Ypred[i] = self.Ytr[one_index]
            print('The data is i, the index is %d, the label is %d' % one_index, self.Ytr[one_index])
        return Ypred



