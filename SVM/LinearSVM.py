#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:xiepanlong 
@file: LinearSVM.py 
@time: 2017/11/06 
"""
from SupportVectorMachine import SVM
from LinearClassifier import LinearClassifier

class LSVM(LinearClassifier):
    def loss(self, X_batch, Y_batch, reg):
        return SVM.svm_loss_vectorized(self.w, X_batch, Y_batch, reg)
