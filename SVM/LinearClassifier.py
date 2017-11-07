#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:xiepanlong 
@file: LinearClassifier.py 
@time: 2017/11/06 
"""

import numpy as np

class LinearClassifier(object):
    def __init__(self, Xtr, Ytr, W=None):
        self.w = W
        self.xtr = Xtr
        self.ytr = Ytr

    def train(self, Xtr, Ytr, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, vis_opt_process=False):
        num_train, dim = Xtr.shape
        num_classes = np.max(Ytr) + 1
        if self.w is None:
            self.w = 0.001 * np.random.randn(dim, num_classes)
        loss_history = []
        for it in xrange(num_iters):
            X_batch = None
            Y_batch = None

            sample_index = np.random.choice(num_train, batch_size, replace=False)
            X_batch = Xtr[sample_index, :]
            Y_batch = Ytr[sample_index]
            loss, grad = self.loss(X_batch, Y_batch, reg)
            loss_history.append(loss)

            self.w += -learning_rate * grad

            if vis_opt_process and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
        return loss_history

    def predict(self, Xte):
        Y_pred = np.zeros(Xte.shape)
        score = Xte.dot(self.w)
        Y_pred = np.argmax(score, axis=1)
        return Y_pred

    def loss(self, X_batch, Y_batch):
        pass



