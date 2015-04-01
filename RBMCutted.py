import threading

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
    def __init__(self, artistsNumber = 5, ranksNumber = 2, hiddenLayerSize = 2, learningRate = 0.1, momentum = 0.9, decay = 0.001):
        self.ArtistsNumber = artistsNumber                                                      #M
        self.RanksNumber = ranksNumber                                                          #K
        self.HiddenLayerSize = hiddenLayerSize                                                  #F
        self.LearningRate = np.float64(learningRate)
        self.momentum = momentum
        self.decay = decay

        self.Ranks = np.arange(ranksNumber, dtype=np.float64).reshape(ranksNumber,1)

        self.HiddenLayer = np.zeros((1, hiddenLayerSize), dtype=np.float64)                     #h
        self.VisibleLayer = np.zeros((ranksNumber, artistsNumber), dtype=np.float64)            #V

        self.HiddenLayerBiases = np.zeros((1, hiddenLayerSize), dtype=np.float64)               #A
        self.VisibleLayerBiases = None
        self.Weights = None
        self.MomentumTable = np.zeros((ranksNumber, hiddenLayerSize, artistsNumber), dtype=np.float64)

        #Synchronizowane miedzy watkiami
        self.GradientsWeights = np.zeros((ranksNumber, hiddenLayerSize, artistsNumber))
        self.GradientsHiddenLayerBiases = np.zeros((1, hiddenLayerSize), dtype=np.float64)
        self.GradientsVisibleLayerBiases = np.zeros((ranksNumber, artistsNumber))

        self.GradientsWeightsCounter = np.zeros((ranksNumber, hiddenLayerSize, artistsNumber), dtype=np.int)
        self.GradientsHiddenLayerBiasesCounter = np.zeros((1, hiddenLayerSize), dtype=np.int)
        self.GradientsVisibleLayerBiasesCounter = np.zeros((ranksNumber, artistsNumber), dtype=np.int)
        #Konice Synchronizowane miedzy watkiami

        self.GlobalVisibleLayerBiases = np.random.normal(0.01, 0.01, (ranksNumber, artistsNumber))    #B  #TODO the proportion of training vectors in which unit i is on
        self.GlobalHiddenLayerBiases = np.zeros((1, hiddenLayerSize), dtype=np.float64)
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
        MomentumTable = self.MomentumTable[:,:,VVector]
        VisibleLayerBiases = self.GlobalVisibleLayerBiases[:,VVector]
        VisibleLayerBiasesInc = VisibleLayerBiases * 0
        HiddenLayerBiases  = self.HiddenLayerBiases
        HiddenLayerBiasesInc = self.HiddenLayerBiases # already zero

        ArtistsNumber = len(VVector)
        VisibleLayer = VisibleData = V
        HiddenLayer = HiddenData = self.computeUpdateTheHiddenStates(HiddenLayerBiases, VisibleLayer, Weights)

        positiveGradient = CastToArray([gradient(VisibleLayer[k,:], HiddenLayer) for k in range(self.RanksNumber)])

        # for i in range(T):
        VisibleLayer = self.computeUpdateTheVisibleStates(VisibleLayerBiases, HiddenLayer, Weights, ArtistsNumber)
        HiddenLayer = self.computeUpdateTheHiddenStates(HiddenLayerBiases, VisibleLayer, Weights)

        negativeGradient = CastToArray([gradient(VisibleLayer[k,:], HiddenLayer) for k in range(self.RanksNumber)])


        #updating
        MomentumTable = self.momentum * MomentumTable + self.LearningRate * (positiveGradient - negativeGradient - self.decay * Weights)
        HiddenLayerBiasesInc = self.momentum * HiddenLayerBiasesInc + self.LearningRate*(HiddenData - HiddenLayer - self.decay * HiddenLayerBiases)
        VisibleLayerBiasesInc = self.momentum * VisibleLayerBiasesInc + self.LearningRate*(VisibleData - VisibleLayer - self.decay * VisibleLayerBiases)
        #Weights += MomentumTable
        #HiddenLayerBiases += HiddenLayerBiasesInc
        #VisibleLayerBiases += VisibleLayerBiasesInc

        lock = threading.Lock() # .RLock() ?
        lock.acquire()
        try:
            self.GradientsWeights[:,:,VVector] += MomentumTable
            self.GradientsWeightsCounter[:,:,VVector] += 1
        except:
            raise
        finally:
            lock.release()
        lock.acquire()
        try:
            self.GradientsHiddenLayerBiases += HiddenLayerBiasesInc
            self.GradientsHiddenLayerBiasesCounter += 1
        except:
            raise
        finally:
            lock.release()

        lock.acquire()
        try:
            self.GradientsVisibleLayerBiases[:,VVector] += VisibleLayerBiasesInc
            self.GradientsVisibleLayerBiasesCounter[:,VVector] += 1
        except:
            raise
        finally:
            lock.release()

        #for sure
        self.VisibleLayer = np.zeros((self.RanksNumber, self.ArtistsNumber), dtype=np.float64)

        # if showLikelihood:
        #     rsm = (V-self.VisibleLayer)
        #     return np.mean(np.multiply(rsm,rsm))

    #fun updatu wag i biasow

    def update(self):
        x = np.where(self.GradientsWeightsCounter >= 100)
        #print(x[0].size)
        #print(x)
        if(x[0].size != 0):
            #print(self.GradientsWeights[x].size)
            print("yolo")
            self.GlobalWeights[x] += self.GradientsWeightsCounter[x] / 100
            self.GradientsWeightsCounter[x] = 0
            self.GradientsWeights[x] = 0

        x = np.where(self.GradientsHiddenLayerBiasesCounter >= 100)
        if (x[0].size != 0):
            self.GlobalHiddenLayerBiases[x] += self.GradientsHiddenLayerBiases[x] / 100
            self.GradientsHiddenLayerBiasesCounter[x] = 0
            self.GradientsHiddenLayerBiases[x] = 0

        x = np.where(self.GradientsVisibleLayerBiasesCounter >= 100)
        if (x[0].size != 0):
            self.GlobalVisibleLayerBiases[x] += self.GradientsVisibleLayerBiases[x] / 100
            self.GradientsVisibleLayerBiasesCounter[x] = 0
            self.GradientsVisibleLayerBiases[x] = 0

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
