# -*-coding: utf-8 -*-

import numpy as np
from DeepLearning.LoadDataSet import LoadCIFAR10
from collections import Counter

class KNN(object):
    def __init__(self, X, Y):
        self.Xtr = X
        self.Ytr = Y

    def train(self, X, Y):
        self.Xtr = X
        self.Ytr = Y

    def compute_distances_no_loop(self, Xte):
        num_test = Xte.shape[0]
        num_train = self.Xtr.shape[0]
        dists = np.zeros((num_test, num_train))
        xtr_square = np.sum(np.square(self.Xtr), axis=1)
        xte_square = np.sum(np.square(Xte), axis=1)
        xtr_xte = np.multiply(np.dot(Xte, self.Xtr.T), -2)
        print ('Xte: ', Xte.shape, 'Xtr: ', self.Xtr.T.shape,
               'xtr_square:', xtr_square.shape, 'xte_square: ', xte_square.shape,
               'xtr_xte: ', xtr_xte.shape)

        dists = np.add(xtr_xte.T, xte_square)
        dists = np.add(dists.T, xtr_square)
        return dists

    def predict(self, dists, K=1):
        num_test = dists.shape[0]
        Ypred = np.zeros(num_test)
        for i in xrange(num_test):
            Yknn = []
            Yknn = self.Ytr[np.argsort(dists[i, :])[:K]].flatten()
            counts = Counter(Yknn)
            Ypred[i] = counts.most_common(1)[0][0]
            print('The data %d, Ypred %d' % (i, Ypred[i]))
        return Ypred

