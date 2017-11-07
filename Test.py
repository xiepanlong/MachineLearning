#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:xiepanlong 
@file: Test.py 
@time: 2017/11/{DAY} 
"""

import KNN.KNearestNeighbor
import numpy as np

from DeepLearning.LoadDataSet import LoadCIFAR10
from DeepLearning.NN import NearestNeighbor
from DeepLearning.KNN import KNearestNeighbor
from DeepLearning.SVM import LinearClassifier

ROOT = '/Users/xiepanlong/Documents/DeepLearning/DataSet/cifar-10-batches-py/'

def testNearestNeighbor():
    Xtr, Ytr, Xte, Yte = LoadCIFAR10.load_CIFAR10(ROOT)
    nn = NearestNeighbor.NN(Xtr, Ytr)
    Ypred = nn.predict(Xte)
    print 'accuracy : % f' % (np.mean(Ypred == Yte))


def testKNearestNeighbor(K):
    Xtr, Ytr, Xte, Yte = LoadCIFAR10.load_CIFAR10(ROOT)
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)
    validation_accuracies = []
    knn = KNearestNeighbor.KNN(Xtr_rows, Xte_rows)
    dists = knn.compute_distances_no_loop(Xte_rows)
    for k in [1, 5, 20, 100]:
        Ypred = knn.predict(dists, k)
        acc = np.mean(Ypred == Yte)
        print 'Acc: %f' % (acc,)
        validation_accuracies.append((k, acc))
    print validation_accuracies

def testLinearClassifier():
    Xtr, Ytr, Xte, Yte = LoadCIFAR10.load_CIFAR10(ROOT)
    lc = LinearClassifier.LinearClassifier(Xtr, Ytr)
    Ypred = lc.predict(Xte)
    print 'accuracy : % f' % (np.mean(Ypred == Yte))

if __name__ == '__main__':
    #print 'NN : '
    #testNearestNeighbor()
    print 'KNN : '
    testKNearestNeighbor(5)

