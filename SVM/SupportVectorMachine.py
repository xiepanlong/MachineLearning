#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:xiepanlong 
@file: SupportVectorMachine.py 
@time: 2017/11/06
"""

import numpy as np

class SVM(object):
    def __init__(self,Xtr, Ytr, W, Reg):
        self.xtr = Xtr
        self.ytr = Ytr
        self.w = W
        self.reg = Reg

    def svm_loss_naive(self, W, X, y, reg ):
        '''
        # 非向量化，naive
        :param W: D * C weights
        :param X: N * D train data
        :param y: N * 1 labels of data
        :param reg: regularization
        :return:
        - loss as single float
        - gradient with respect to W, a array like W
        '''
        dW = np.zeros(W.shape)
        num_classes = W.shape[1]
        num_train = X.shape[0]
        loss = 0.0
        for i in xrange(num_train):
            scores = X[i].dot(W) # Xi 分成C类，存在广播
            correct_class_score = scores[y[i]] # 获得分类正确的分数
            for j in xrange(num_classes):
                if j == y[i]:
                    continue # 等于正确分类时不管
                margin = scores[j] - correct_class_score + 1 # 令delta = 1
                if margin > 0:
                    loss += margin
                    dW[:, y[i]] += -X[i, :]
                    dW[:, j] += X[i, :] # why??
        loss /= num_train
        dW /= num_train
        loss += 0.5 * reg * np.sum(W * W)
        dW += reg * W
        return loss, dW

    def svm_loss_vectorized(self, W, X, y, reg):
        '''
        :param W:
        :param X:
        :param y:
        :param reg:
        :return:
        like naived, but vectorized.
        '''
        loss = 0.0
        dW = np.zeros(W.shape)
        scores = X.dot(W)
        num_train = X.shape[0]
        num_classes = W.shape[1]
        scores_correct = scores[np.arange(num_train),y]
        scores_correct = np.reshape(scores_correct, (num_train, 1))
        margins = scores - scores_correct + 1.0
        margins[np.arange(num_train), y] = 0.0
        margins[margins <= 0] = 0.0
        loss += np.sum(margins) / num_train
        loss += 0.5 * reg * np.sum(W * W)
        margins[margins > 0] = 1.0
        row_sum = np.sum(margins, axis=1)
        margins[np.arange(num_train), y] = -row_sum
        dW += np.dot(X.T, margins) / num_train + reg * W
        return loss, dW
