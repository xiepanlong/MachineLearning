#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:xiepanlong 
@file: Softmax.py 
@time: 2017/11/06 
"""

from LinearClassifier import LinearClassifier
from SupportVectorMachine import SVM

class SM(LinearClassifier):
    def loss(self, X_batch, Y_batch, reg):
        return SVM.svm_loss_vectorized(self.w, X_batch, Y_batch, reg)