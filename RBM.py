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
    def __init__(self, artistsNumber = 5, ranksNumber = 2, hiddenLayerSize = 3, learningRate = 0.1):
        self.ArtistsNumber = artistsNumber
        self.RanksNumber = ranksNumber
        self.HiddenLayerSize = hiddenLayerSize
        self.LearningRate = learningRate


        self.HiddenLayer = np.zeros((hiddenLayerSize))
        self.VisibleLayer = np.ones((ranksNumber, artistsNumber))

        self.HiddenLayerBiases = np.zeros((hiddenLayerSize))
        self.VisibleLayerBiases = np.random.normal(0.01, 0.01, (ranksNumber, artistsNumber)) #TODO the proportion of training vectors in which unit i is on

        self.Weights = np.random.normal(0, 0.01, (ranksNumber, hiddenLayerSize, artistsNumber))

    def computeProbabilityTheHiddenStates(self):
        # Eq. 2
        expressionInsideTheParentheses = lambda j: self.HiddenLayerBiases[j] + (Multiply(self.VisibleLayer, self.Weights[:,j])).sum()
        return Sigmoid(CastGeneratorToArray(expressionInsideTheParentheses(j) for j in range(self.HiddenLayerSize)))

    def computeUpdateTheHiddenStates(self):
        return CastToMatrix([random.choice([0, 1], p=[probability, 1-probability]) for probability in self.computeProbabilityTheHiddenStates()])

    def computeUpdateTheVisibleStates(self):
        # Eq. 1
        numerator = lambda i,k: Exponent(self.VisibleLayerBiases[k, i] + (Multiply(self.HiddenLayer, self.Weights[k, :, i])).sum())
        denominator = lambda i: Exponent([self.VisibleLayerBiases[l, i] + (Multiply(self.HiddenLayer, self.Weights[l, :, i])).sum() for l in range(self.RanksNumber)]).sum()
        return CastToMatrix([[numerator(i,k) for k in range(self.RanksNumber)] / denominator(i) for i in range(self.ArtistsNumber)]).T #keep calm and pray it work


    def learn(self, V = None, T = 1):
        #TODO Updating biases
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
        self.HiddenLayer = self.computeProbabilityTheHiddenStates()
        self.VisibleLayer = self.computeUpdateTheVisibleStates()

        Ranks = CastToMatrix(CastGeneratorToArray(k for k in range(self.RanksNumber))).T #zero rank is on index 0, 1 - 1, etc.

        return [Multiply(self.VisibleLayer[:,i], Ranks).sum() for i in range(self.ArtistsNumber)]


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