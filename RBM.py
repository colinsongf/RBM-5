from time import time
from functools import wraps
from timeit import timeit


import pprint
__author__ = 'Olek'

import numpy as np
from numpy import random
from scipy.special import expit as Sigmoid
from numpy import multiply as Multiply
from numpy import exp as Exponent
from numpy import matrix as CastToMatrix

CastGeneratorToArray = lambda x: np.fromiter((x), np.float64)
CastToArray = lambda x: np.array((x), np.float64)

class RMB():
    def __init__(self, artistsNumber = 5, ranksNumber = 2, hiddenLayerSize = 2, learningRate = 0.1):
        self.ArtistsNumber = artistsNumber
        self.RanksNumber = ranksNumber
        self.HiddenLayerSize = hiddenLayerSize
        self.LearningRate = np.float64(learningRate)

        self.Ranks = np.arange(ranksNumber, dtype=np.float64).reshape(ranksNumber,1)

        self.HiddenLayer = np.zeros((1, hiddenLayerSize), dtype=np.float64)
        self.VisibleLayer = np.ones((ranksNumber, artistsNumber), dtype=np.float64)

        self.HiddenLayerBiases = np.zeros((hiddenLayerSize), dtype=np.float64)
        self.VisibleLayerBiases = np.random.normal(0.01, 0.01, (ranksNumber, artistsNumber)) #TODO the proportion of training vectors in which unit i is on

        self.Weights = np.random.normal(0, 0.01, (ranksNumber, hiddenLayerSize, artistsNumber))

    def computeProbabilityTheHiddenStates(self):
        # Eq. 2

        expressionInsideTheParentheses = np.vectorize(lambda j: self.HiddenLayerBiases[j] + (Multiply(self.VisibleLayer, self.Weights[:,j])).sum())

        # print(self.VisibleLayer.shape)
        # print(self.Weights.shape)
        #
        # print(np.kron(self.VisibleLayer * self.Weights)+self.HiddenLayer)
        # print(Multiply(self.VisibleLayer, self.Weights))

        # print(self.HiddenLayerBiases[0] + (Multiply(self.VisibleLayer, self.Weights[:,0])).sum())

        # print([expressionInsideTheParentheses(j) for j in range(self.HiddenLayerSize)])
        #
        # test = np.arange(5)
        # print(test)
        #

        return Sigmoid(expressionInsideTheParentheses(np.arange(self.HiddenLayerSize)))

    def computeUpdateTheHiddenStates(self):
        probabilities = self.computeProbabilityTheHiddenStates()
        return np.random.binomial(1,probabilities, size=(1,self.HiddenLayerSize))

    def computeUpdateTheVisibleStates(self):
        # Eq. 1
        product = Exponent(self.VisibleLayerBiases+np.dot(self.HiddenLayer,self.Weights))
        return (product/product.sum(1)).reshape(self.RanksNumber,self.ArtistsNumber) #keep calm and pray it work

    # def learn(self, V = None, T = 1):
    #     #TODO Updating biases
    #     #TODO When Changing T
    #
    #     self.VisibleLayer = V
    #     self.HiddenLayer = self.computeUpdateTheHiddenStates()
    #
    #     positiveGradient = np.kron(self.VisibleLayer, self.HiddenLayer.T)
    #
    #     for i in range(T):
    #         self.VisibleLayer = self.computeUpdateTheVisibleStates()
    #         self.HiddenLayer = self.computeUpdateTheHiddenStates()
    #
    #     negativeGradient = np.kron(self.VisibleLayer, self.HiddenLayer.T)
    #
    #     self.Weights += Multiply(self.LearningRate,np.split((positiveGradient - negativeGradient),self.RanksNumber, 0))

    def learn(self, V = None, T = 1):
        gradient = lambda v,h: Multiply(v, h.T)

        self.VisibleLayer = V
        self.HiddenLayer = self.computeUpdateTheHiddenStates()

        positiveGradient = CastToArray([gradient(self.VisibleLayer[k,:], self.HiddenLayer) for k in range(self.RanksNumber)])

        for i in range(T):
            self.VisibleLayer = self.computeUpdateTheVisibleStates()
            self.HiddenLayer = self.computeUpdateTheHiddenStates()

        negativeGradient = CastToArray([gradient(self.VisibleLayer[k,:], self.HiddenLayer) for k in range(self.RanksNumber)])

        self.Weights += self.LearningRate*(positiveGradient - negativeGradient)

    def prediction(self, V = None):
        self.VisibleLayer = V
        self.HiddenLayer = self.computeProbabilityTheHiddenStates().reshape(1, self.HiddenLayerSize)
        self.VisibleLayer = self.computeUpdateTheVisibleStates()

        return Multiply(self.VisibleLayer, self.Ranks).sum(axis = 0)


    def saveRBM(self):
        import os
        from time import strftime, localtime
        try:
            os.makedirs("Saves")
        except OSError:
            pass

        fileName = strftime("%Y-%m-%d-%H-%M-%S", localtime())

        np.savez("Saves//"+fileName, \
        ArtistsNumber = self.ArtistsNumber, \
        RanksNumber = self.RanksNumber, \
        HiddenLayerSize = self.HiddenLayerSize, \
        LearningRate = self.LearningRate, \
        HiddenLayer = self.HiddenLayer, \
        VisibleLayer = self.VisibleLayer, \
        HiddenLayerBiases = self.HiddenLayerBiases, \
        VisibleLayerBiases = self.VisibleLayerBiases, \
        Weights = self.Weights)
        return fileName+".npz"


def loadRBM(file):
    RBMFile = np.load(file)
    loadedRBM = RMB()
    loadedRBM.ArtistsNumber = RBMFile['ArtistsNumber']
    loadedRBM.RanksNumber = RBMFile['RanksNumber']
    loadedRBM.HiddenLayerSize = RBMFile['HiddenLayerSize']
    loadedRBM.LearningRate = RBMFile['LearningRate']
    loadedRBM.HiddenLayer = RBMFile['HiddenLayer']
    loadedRBM.VisibleLayer = RBMFile['VisibleLayer']
    loadedRBM.HiddenLayerBiases = RBMFile['HiddenLayerBiases']
    loadedRBM.VisibleLayerBiases = RBMFile['VisibleLayerBiases']
    loadedRBM.Weights = RBMFile['Weights']
    return loadedRBM

