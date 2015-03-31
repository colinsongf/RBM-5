__author__ = 'Olek'

import numpy as np
from scipy.special import expit as Sigmoid
from numpy import multiply as Multiply
from numpy import exp as Exponent
from itertools import starmap
from operator import mul

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
        self.VisibleLayer = np.zeros((ranksNumber, artistsNumber), dtype=np.float64)             #V

        self.HiddenLayerBiases = np.zeros((1, hiddenLayerSize), dtype=np.float64)               #A
        self.VisibleLayerBiases = None
        self.Weights = None

        self.GlobalVisibleLayerBiases = np.random.normal(0.01, 0.01, (ranksNumber, artistsNumber))    #B  #TODO the proportion of training vectors in which unit i is on
        self.GlobalWeights = np.random.normal(0, 0.01, (ranksNumber, hiddenLayerSize, artistsNumber)) #W

    def computeProbabilityTheHiddenStates(self, HiddenLayerBiases, VisibleLayer, Weights):
        # Eq. 2
        expressionInsideTheParentheses = np.vectorize(lambda j: HiddenLayerBiases[0,j] + (Multiply(VisibleLayer, Weights[:,j])).sum())
        return Sigmoid(expressionInsideTheParentheses(np.arange(self.HiddenLayerSize)))

    def computeUpdateTheHiddenStates(self, HiddenLayerBiases, VisibleLayer, Weights):
        probabilities = self.computeProbabilityTheHiddenStates(HiddenLayerBiases, VisibleLayer, Weights)
        return np.random.binomial(1,probabilities, size=(1,self.HiddenLayerSize))                   #draw a hidden feature

    def computeUpdateTheVisibleStates(self, VisibleLayerBiases, HiddenLayer, Weights, ArtistsNumber):
        # Eq. 1
        product = Exponent(VisibleLayerBiases+np.dot(HiddenLayer,Weights))                          #σ(B+h·W)
        return (product/product.sum(1)).reshape(self.RanksNumber,ArtistsNumber)                #keep calm and pray it work

    def learn(self, VVector = None, V = None, T = 1, showLikelihood = False):
        #TODO try learning with erasing visible states
        gradient = lambda v,h: Multiply(v, h.T)


        Weights = self.GlobalWeights[:,:,VVector]
        VisibleLayerBiases = self.GlobalVisibleLayerBiases[:,VVector]

        ArtistsNumber = len(VVector)
        VisibleLayer = VisibleData = V

        HiddenLayerBiases = self.HiddenLayerBiases

        HiddenLayer = HiddenData = self.computeUpdateTheHiddenStates(HiddenLayerBiases, VisibleLayer, Weights)

        positiveGradient = CastToArray([gradient(VisibleLayer[k,:], HiddenLayer) for k in range(self.RanksNumber)])

        # for i in range(T):
        VisibleLayer = self.computeUpdateTheVisibleStates(VisibleLayerBiases, HiddenLayer, Weights, ArtistsNumber)
        HiddenLayer = self.computeUpdateTheHiddenStates(HiddenLayerBiases, VisibleLayer, Weights)

        negativeGradient = CastToArray([gradient(VisibleLayer[k,:], HiddenLayer) for k in range(self.RanksNumber)])

        #updating
        Weights += self.LearningRate*(positiveGradient - negativeGradient)
        self.GlobalWeights[:,:,VVector] = Weights
        self.HiddenLayerBiases += self.LearningRate*(HiddenData - HiddenLayer)
        VisibleLayerBiases += self.LearningRate*(VisibleData - VisibleLayer)
        self.GlobalVisibleLayerBiases[:,VVector] = VisibleLayerBiases
        # if showLikelihood:
        #     rsm = (V-self.VisibleLayer)
        #     return np.mean(np.multiply(rsm,rsm))
        self.VisibleLayer = np.zeros((self.RanksNumber, self.ArtistsNumber), dtype=np.float64)

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
    loadedRBM = RMBCutted()
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
