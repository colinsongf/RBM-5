__author__ = 'Aleksander Surman, Tomasz Wis'

import threading
from time import time

from DataLoader import DataLoader
from RBM import RBM, loadRBM
import numpy as np

def computeRMSE(rbm = None, dataLoader = None, threadsNumber = 10, verbose = False):
    """
    Compute Root mean square error for RBM
    :param rbm: RBM instance
    :param dataLoader: data loader instance
    :param threadsNumber: number of used threads
    :param verbose: true enables logs
    :return:
    """
    startTime = time()

    dataLoader.StartNewValidationSet()

    # sharing between thread
    errors = []
    errorsLock = threading.Lock()

    def threadJob(threadNumber):
        for i in range(int(dataLoader.validationSetSize/threadNumber)):
            (vVector, v, integerV) = dataLoader.GiveVisibleLayerForValidation(threadNumber)
            predictions = rbm.prediction((vVector, v), isValidation=True)
            with errorsLock:
                errors.append(predictions - integerV)

    threads = []
    for i in range(threadsNumber):
        threads.append(threading.Thread(target=threadJob, args=(i, )))
        threads[i].start()

    for i in range(threadsNumber):
        threads[i].join()

    errors = np.hstack(errors)
    RMSE = np.sqrt(np.power(errors, 2).sum()/len(errors))

    endTime = time()

    if verbose:
        print("RMSE: {0} \nTook: {1:0.5f} sec".format(RMSE, endTime - startTime))
    return RMSE

def learnOneEpoch(rbm = None, dataLoader = None, threadsNumber = 10, batchSizeForOneThread = 100, numberOfMiniSets = 319, verbose = False):
    """
    Trains all users from training set once
    :param rbm: RBM instance
    :param dataLoader: data loader instance
    :param threadsNumber: number of used threads
    :param batchSizeForOneThread: number of users trained by single thread at the time
    :param numberOfMiniSets: number of set thread number times batch size in training set
    :param verbose: true enables logs
    :return:
    """
    startTime = time()
    for setNumber in range(numberOfMiniSets):
        if setNumber + 1 == numberOfMiniSets:           # last case computing with 1 thread to reduce problem with division data onto many threads
            startTime = time()

            dataLoader.StartNewTrainingMiniSet()

            for i in range(dataLoader.trainingSetSize - (setNumber + 1) * (threadsNumber * batchSizeForOneThread)):
                rbm.learn(dataLoader.GiveVisibleLayerForTraining(0))

            rbm.update(verbose=verbose)

            endTime = time()

            if verbose:
                print("Finish last mini set no: {0} \nTook: {1:0.5f} sec".format(setNumber, endTime - startTime))
        else:
            startTime = time()

            dataLoader.StartNewTrainingMiniSet()

            def threadJob(threadNumber):
                for i in range(batchSizeForOneThread):
                    rbm.learn(dataLoader.GiveVisibleLayerForTraining(threadNumber))

            threads = []
            for i in range(threadsNumber):
                threads.append(threading.Thread(target=threadJob, args=(i, )))
                threads[i].start()

            for i in range(threadsNumber):
                threads[i].join()

            rbm.update(verbose=verbose) # updating after one mini set

            endTime = time()

            if verbose:
                print("Finish mini set no: {0} \nTook: {1:0.5f} sec".format(setNumber, endTime - startTime))

    endTime = time()
    if verbose:
        print("Epoch took: {0:0.5f} sec".format(endTime - startTime))

if __name__ == "__main__":
    #################################
    #         configuration         #
    #################################
    threadsNumber = 10              #
    batchSizeForOneThread = 100     #
    M = 17765                       #
    K = 5                           #
    F = 100                         #
    learningRate = 0.1              #
    momentum = 0.9                  #
    wDecay = 0.001                  #
    updateFrequencyMAX = 300        #
                                    #
    numberOfEpoch = 1               #
    #################################

    dataLoader = DataLoader(K=K, M=M, batchSizeForOneThread=batchSizeForOneThread, threadsNumber=threadsNumber, verbose=True)

    # updateFrequency decides how often RMB updates global data
    whereUpdateMax = np.where(dataLoader.updateFrequency > updateFrequencyMAX)
    dataLoader.updateFrequency[whereUpdateMax] = updateFrequencyMAX

    rbm = RBM(M, K, F, learningRate, momentum, wDecay, dataLoader.vBiasesInitialization, dataLoader.updateFrequency)
    numberOfMiniSets = int(np.floor(dataLoader.trainingSetSize / (threadsNumber * batchSizeForOneThread)))

    for i in range(numberOfEpoch):
        dataLoader.StartNewEpoch()
        learnOneEpoch(rbm, dataLoader, threadsNumber, batchSizeForOneThread, numberOfMiniSets, verbose=True)
        with open("RBM_RMSEs.txt", "a") as RMSEsFile:
            RMSEsFile.write("Epoch {0}, RMSE {1}\n".format(i, computeRMSE(rbm, dataLoader, threadsNumber, verbose=True)))



