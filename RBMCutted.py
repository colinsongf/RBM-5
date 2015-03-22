__author__ = 'Olek'

import numpy as np
from scipy.special import expit as Sigmoid
from numpy import multiply as Multiply
from numpy import exp as Exponent

CastGeneratorToArray = lambda x: np.fromiter((x), np.float64)
CastToArray = lambda x: np.array((x), np.float64)

class RMBCutted():
    def __init__(self, artistsNumber = 5, ranksNumber = 2, hiddenLayerSize = 2, learningRate = 0.1):
        self.ArtistsNumber = artistsNumber                                                      #M
        self.RanksNumber = ranksNumber                                                          #K
        self.HiddenLayerSize = hiddenLayerSize                                                  #F
        self.LearningRate = np.float64(learningRate)

        self.Ranks = np.arange(ranksNumber, dtype=np.float64).reshape(ranksNumber,1)

        self.HiddenLayer = np.zeros((1, hiddenLayerSize), dtype=np.float64)                     #h
        self.VisibleLayer = np.ones((ranksNumber, artistsNumber), dtype=np.float64)             #V

        self.HiddenLayerBiases = np.zeros((1, hiddenLayerSize), dtype=np.float64)               #A
        self.VisibleLayerBiases = None
        self.Weights = None

        self.GlobalVisibleLayerBiases = np.random.normal(0.01, 0.01, (ranksNumber, artistsNumber))    #B  #TODO the proportion of training vectors in which unit i is on
        self.GlobalWeights = np.random.normal(0, 0.01, (ranksNumber, hiddenLayerSize, artistsNumber)) #W

    def computeProbabilityTheHiddenStates(self):
        # Eq. 2
        expressionInsideTheParentheses = np.vectorize(lambda j: self.HiddenLayerBiases[0,j] + (Multiply(self.VisibleLayer, self.Weights[:,j])).sum())
        return Sigmoid(expressionInsideTheParentheses(np.arange(self.HiddenLayerSize)))

    def computeUpdateTheHiddenStates(self):
        probabilities = self.computeProbabilityTheHiddenStates()
        return np.random.binomial(1,probabilities, size=(1,self.HiddenLayerSize))                   #draw a hidden feature

    def computeUpdateTheVisibleStates(self):
        # Eq. 1
        product = Exponent(self.VisibleLayerBiases+np.dot(self.HiddenLayer,self.Weights))           #σ(B+h·W)
        return (product/product.sum(1)).reshape(self.RanksNumber,self.ArtistsNumber)                #keep calm and pray it work

    def learn(self, VVector = None, V = None, T = 1, showLikelihood = False):
        #TODO try learning with erasing visible states
        gradient = lambda v,h: Multiply(v, h.T)


        self.Weights = self.GlobalWeights[:,:,VVector]
        self.VisibleLayerBiases = self.GlobalVisibleLayerBiases[:,VVector]

        self.ArtistsNumber = len(VVector)
        self.VisibleLayer = VisibleData = V
        self.HiddenLayer = HiddenData = self.computeUpdateTheHiddenStates()

        positiveGradient = CastToArray([gradient(self.VisibleLayer[k,:], self.HiddenLayer) for k in range(self.RanksNumber)])

        for i in range(T):
            self.VisibleLayer = self.computeUpdateTheVisibleStates()
            self.HiddenLayer = self.computeUpdateTheHiddenStates()

        negativeGradient = CastToArray([gradient(self.VisibleLayer[k,:], self.HiddenLayer) for k in range(self.RanksNumber)])

        #updating
        self.Weights += self.LearningRate*(positiveGradient - negativeGradient)
        self.GlobalWeights[:,:,VVector] = self.Weights
        self.HiddenLayerBiases += self.LearningRate*(HiddenData - self.HiddenLayer)
        self.VisibleLayerBiases += self.LearningRate*(VisibleData - self.VisibleLayer)
        self.GlobalVisibleLayerBiases[:,VVector] = self.VisibleLayerBiases
        if showLikelihood:
            rsm = (V-self.VisibleLayer)
            return np.mean(np.multiply(rsm,rsm))

    def prediction(self, V = None):
        self.VisibleLayer = V
        self.HiddenLayer = self.computeProbabilityTheHiddenStates().reshape(1, self.HiddenLayerSize)
        self.VisibleLayer = self.computeUpdateTheVisibleStates()

        return Multiply(self.VisibleLayer, self.Ranks).sum(0)


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

