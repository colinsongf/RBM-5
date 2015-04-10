__author__ = 'Aleksander Surman, Tomasz Wis'

import numpy as np
import threading
from scipy.special import expit as Sigmoid

# casting method
CastGeneratorToArray = lambda x: np.fromiter((x), np.float64)
CastToArray = lambda x: np.array((x), np.float64)


class RBM():
    """
    Restricted Boltzman Machine class
    """

    def __init__(self, M = 17765, K = 5, F = 100, learningRate = 0.1, momentum = 0.9, wDecay = 0.001, vBiasesInitialization = None, updateFrequency = None):
        """
        Creates RBM
        :param M: number of artists
        :param K: number of grades
        :param F: number of songs features
        :param learningRate: learning rate for weights, hidden biases and visible biases
        :param momentum: momentum for weights, hidden biases and visible biases
        :param wDecay: weight decay
        :param vBiasesInitialization: visible biases initialization method
        :param updateFrequency: maximum number of local updates to update global weights
        :return: RMB class
        """
        # Constants
        self.M = M
        self.K = K
        self.F = F
        self.LearningRate = np.float32(learningRate)
        self.Momentum = np.float32(momentum)
        self.WDecay = np.float32(wDecay)

        # Shared between threads
        # wDelta - local weights
        # wDeltaCounter - update counter for weights
        self.wDeltaLock = threading.Lock()
        self.wDelta = np.zeros((K, F, M), dtype=np.float32)
        self.wDeltaCounter = np.zeros((K, F, M), dtype=np.int)

        self.hBiasesDeltaLock = threading.Lock()
        self.hBiasesDelta = np.zeros((1, F), dtype=np.float32)
        self.hBiasesDeltaCounter = np.zeros((1, F), dtype=np.int)

        self.vBiasesDeltaLock = threading.Lock()
        self.vBiasesDelta = np.zeros((K, M))
        self.vBiasesDeltaCounter = np.zeros((K, M), dtype=np.int)

        # Globals
        # wGlobal - Weights updating after each mini sets
        # vBiasesGlobal - Visible layer biases updating after each mini sets
        # hBiasesGlobal - Hidden layer biases updating after each mini sets
        self.wGlobal = np.random.normal(0, 0.01, (K, F, M))
        self.vBiasesGlobal = vBiasesInitialization
        self.hBiasesGlobal = np.zeros((1, F), dtype=np.float32)


        self.wMomentumTable = np.zeros((K, F, M), dtype=np.float32)
        self.vBiasesMomentumTable = np.zeros((K, M), dtype=np.float32)
        self.hBiasesMomentumTable = np.zeros((1, F), dtype=np.float32)

        # to make predictions faster
        self.Ranks = np.arange(self.K, dtype=np.float32).reshape(self.K,1)

        self.updateFrequency = updateFrequency
        self.wUpdateFrequency = np.ones((K, F, M))
        for i in range(F):
            self.wUpdateFrequency[:,i] = updateFrequency  # I haven't got better idea

    # Eq. 2
    def computeProbabilityTheHiddenStates(self, v, w):
        """
        Computes probabilities of new hidden layer
        :param v: visible layer (artists X grades)
        :param w: weights
        :return: hidden layer as vector of probabilities
        """

        expressionInsideTheParentheses = np.vectorize(lambda j: self.hBiasesGlobal[0,j] + (np.multiply(v, w[:,j])).sum())
        return Sigmoid(expressionInsideTheParentheses(np.arange(self.F)))


    def computeUpdateTheHiddenStates(self, v, w):
        """
        Computes new hidden layer as binary
        :param v: visible layer (artists X grades)
        :param w: weights
        :return: hidden layer as binary vector
        """

        probabilities = self.computeProbabilityTheHiddenStates(v, w)
        return np.random.binomial(1,probabilities, size=(1,self.F))


    # Eq. 1
    def computeUpdateTheVisibleStates(self, vBiases, h, w, artistsNumber):
        """
        Computes new visible layer
        :param vBiases: biases for visible layer
        :param h: hidden layer
        :param w: weights
        :param artistsNumber: number of used artists
        :return: new visible layer
        """
        # σ(B+h·W)
        product = np.exp(vBiases+np.dot(h, w))
        return (product/product.sum(1)).reshape(self.K, artistsNumber)                #keep calm and pray it work

    def learn(self, input=None, T=1):
        """
        Learn from single user
        :param input: visible layer of user
        :param T: accuracy of Contrastive Divergence
        :return: None
        """

        # numpy.krom worked slower
        gradient = lambda v,h: np.multiply(v, h.T)

        (vVector, v) = input

        w = self.wGlobal[:,:,vVector]
        vBiases = self.vBiasesGlobal[:,vVector]

        v = vData = v
        h = hData = self.computeUpdateTheHiddenStates(v, w)

        positiveGradient = CastToArray([gradient(v[k,:], h) for k in range(self.K)])

        for i in range(T):
            v = self.computeUpdateTheVisibleStates(vBiases, h, w, len(vVector))
            h = self.computeUpdateTheHiddenStates(v, w)

        negativeGradient = CastToArray([gradient(v[k,:], h) for k in range(self.K)])

        def uglyButWork(vData):
            vDataToUpdateW = np.zeros((self.K, self.F, len(vVector)))
            for i in range(self.F):
                vDataToUpdateW[:,i] = vData
            return vDataToUpdateW

        with self.wDeltaLock:
            self.wDelta[:,:,vVector] += positiveGradient - negativeGradient
            self.wDeltaCounter[:,:,vVector] += uglyButWork(vData)

        with self.hBiasesDeltaLock:
            self.hBiasesDelta += hData - h
            self.hBiasesDeltaCounter += 1

        with self.vBiasesDeltaLock:
            self.vBiasesDelta[:,vVector] += vData - v
            self.vBiasesDeltaCounter[:,vVector] += vData


    def update(self, verbose=False):
        """
        Updates global weights and biases
        :param verbose: true enables logs
        :return: None
        """
        def log(x):
            if verbose:
                print(x)

        # updating weights

        wDeltaWhere = np.where((self.wDeltaCounter >= self.wUpdateFrequency) & (self.wDeltaCounter != 0))
        self.wMomentumTable[wDeltaWhere] = self.Momentum * self.wMomentumTable[wDeltaWhere] + self.LearningRate * (self.wDelta[wDeltaWhere]/self.wDeltaCounter[wDeltaWhere] - self.WDecay * self.wGlobal[wDeltaWhere])
        self.wGlobal[wDeltaWhere] += self.wMomentumTable[wDeltaWhere]
        self.wDeltaCounter[wDeltaWhere] = 0
        self.wDelta[wDeltaWhere] = 0
        log("Updated {0} Weights".format(wDeltaWhere[0].size))

        # updating hidden biases
        self.hBiasesMomentumTable = self.Momentum * self.hBiasesMomentumTable + self.LearningRate * (self.hBiasesDelta/self.hBiasesDeltaCounter)
        self.hBiasesGlobal += self.hBiasesMomentumTable
        self.hBiasesDeltaCounter[:] = 0
        self.hBiasesDelta[:] = 0
        log("Updated Hidden biases")

        # updating visible biases
        vBiasesWhere = np.where((self.vBiasesDeltaCounter >= self.updateFrequency) & (self.vBiasesDeltaCounter != 0))
        self.vBiasesMomentumTable[vBiasesWhere] = self.Momentum * self.vBiasesMomentumTable[vBiasesWhere] + self.LearningRate * (self.vBiasesDelta[vBiasesWhere]/self.vBiasesDeltaCounter[vBiasesWhere])
        self.vBiasesGlobal[vBiasesWhere] += self.vBiasesMomentumTable[vBiasesWhere]
        self.vBiasesDeltaCounter[vBiasesWhere] = 0
        self.vBiasesDelta[vBiasesWhere] = 0
        log("Updated {0} Visible biases".format(vBiasesWhere[0].size))

    def prediction(self, input=None, isValidation = False):
        """
        Computes prediction of visible layer
        :param input:
        :param isValidation: false compute all artists
                             true  compute relevant artists
        :return: predicted visible layer
        """

        # vVector used artists
        (vVector, v) = input

        if isValidation:
            w = self.wGlobal[:,:,vVector]
            vBiases = self.vBiasesGlobal[:,vVector]

            h = self.computeUpdateTheHiddenStates(v, w)
            v = self.computeUpdateTheVisibleStates(vBiases, h, w, len(vVector))

            return np.multiply(v, self.Ranks).sum(0)
        else:
            h = self.computeUpdateTheHiddenStates(v, self.wGlobal[:,:,vVector])
            v = self.computeUpdateTheVisibleStates(self.vBiasesGlobal, h, self.wGlobal, self.M)

            return np.multiply(v, self.Ranks).sum(0)

    def saveRBM(self):
        """
        Saves data to numpy file
        """
        import os
        from time import strftime, localtime
        saveDir = "Saves//"
        try:
            if not os.path.isdir("Saves//"):
                os.makedir("Saves")
            saveDir = "Saves//"
        except OSError:
            saveDir = "" # if error save in . folder
            pass

        fileName = strftime("%Y-%m-%d-%H-%M-%S", localtime())

        np.savez(saveDir+fileName, \
        M = self.M, \
        K = self.K, \
        F = self.F, \
        learningRate = self.LearningRate, \
        momentum = self.Momentum, \
        wDecay = self.WDecay, \

        wDelta = self.wDelta, \
        hBiasesDelta = self.hBiasesDelta, \
        vBiasesDelta = self.vBiasesDelta, \

        wDeltaCounter = self.wDeltaCounter, \
        hBiasesDeltaCounter = self.hBiasesDeltaCounter, \
        vBiasesDeltaCounter = self.vBiasesDeltaCounter, \

        wGlobal = self.wGlobal, \
        vBiasesGlobal = self.vBiasesGlobal, \
        hBiasesGlobal = self.hBiasesGlobal, \

        wMomentumTable = self.wMomentumTable, \
        vBiasesMomentumTable = self.vBiasesMomentumTable, \
        hBiasesMomentumTable = self.hBiasesMomentumTable, \

        Ranks = self.Ranks)
        return saveDir+fileName+".npz"


def loadRBM(file):
    """
    Load RBM data from file
    :param file: numpy file
    :return: None
    """
    RBMFile = np.load(file)
    loadedRBM = RBM()
    loadedRBM.M = RBMFile['M']
    loadedRBM.K = RBMFile['K']
    loadedRBM.F = RBMFile['F']
    loadedRBM.LearningRate = RBMFile['learningRate']
    loadedRBM.Momentum = RBMFile['momentum']
    loadedRBM.WDecay = RBMFile['wDecay']

    loadedRBM.wDelta = RBMFile['wDelta']
    loadedRBM.hBiasesDelta = RBMFile['hBiasesDelta']
    loadedRBM.vBiasesDelta = RBMFile['vBiasesDelta']

    loadedRBM.wDeltaCounter = RBMFile['wDeltaCounter']
    loadedRBM.hBiasesDeltaCounter = RBMFile['hBiasesDeltaCounter']
    loadedRBM.vBiasesDeltaCounter = RBMFile['vBiasesDeltaCounter']

    loadedRBM.wGlobal = RBMFile['wGlobal']
    loadedRBM.vBiasesGlobal = RBMFile['vBiasesGlobal']
    loadedRBM.hBiasesGlobal = RBMFile['hBiasesGlobal']

    loadedRBM.wMomentumTable = RBMFile['wMomentumTable']
    loadedRBM.vBiasesMomentumTable = RBMFile['vBiasesMomentumTable']
    loadedRBM.hBiasesMomentumTable = RBMFile['hBiasesMomentumTable']

    loadedRBM.Ranks = RBMFile['Ranks']

    loadedRBM.wDeltaLock = threading.Lock()
    loadedRBM.hBiasesDeltaLock = threading.Lock()
    loadedRBM.vBiasesDeltaLock = threading.Lock()
    return loadedRBM

