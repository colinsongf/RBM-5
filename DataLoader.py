__author__ = 'Aleksander Surman'

from _testcapi import the_number_three
from time import time

##################################################### USAGE #####################################################
#                                                                                                               #
# First Initialize Object!                                                                                       #
#                                                                                                               #
# For Training:                                                                                                 #
#   StartNewTrainingMiniSet for Initialize new mini set                                                         #
#       (iterate over number of users given in batchSizeForOneThread with thread number given in threadNumber,  #
#           data in each thread are different)                                                                  #
#   GiveVisibleLayerForTraining(N) return one user data for nth training thread                                 #
#                                                                                                               #
# For Validation:                                                                                               #
#   GiveVisibleLayerForValidation return one user data for validation                                           #
#                                                                                                               #
# For Test:                                                                                                     #
#   GiveVisibleLayerForTest return one user data for validation                                                 #
#################################################################################################################

import numpy as np


class DataLoader():

    def makeVisibleFromRanks(self, matrix, K, addRanksToTuple = False, generatevBiasesInitialization  = False, generateUpdateFrequency = False):

        if generatevBiasesInitialization:
            vBiasesInitialization = np.zeros((self.K, self.M), dtype=np.float32)
            counter = 0;

        visibleLayer = []
        for (indexes, ranks) in matrix: # data format (tuple of artist indexes, tuple of ranks)
            Vtmp = np.zeros((K, len(ranks)), dtype=np.float32)
            for index in range(len(ranks)):
                Vtmp[ranks[index]][index] = np.float32(1.0) # not sure whether should be row[index] - 1, because ranks are from 1
                if generatevBiasesInitialization:
                    vBiasesInitialization[ranks[index]][indexes[index]] += 1
                    counter += 1
            if addRanksToTuple:
                visibleLayer.append((indexes, Vtmp, ranks))
            else:
                visibleLayer.append((indexes, Vtmp))
        if generatevBiasesInitialization and generateUpdateFrequency:
            return visibleLayer, vBiasesInitialization/counter, vBiasesInitialization.astype(np.int)
        elif generatevBiasesInitialization:
            return visibleLayer, vBiasesInitialization/counter,
        return visibleLayer

    def __init__(self,
                    trainingSetFile = "Data\TrainingSet.npy",
                    validationSetFile = "Data\ValidationSet.npy",
                    testSetFile = "Data\TestSet.npy",
                    K = 5,
                    M = 17765,
                    batchSizeForOneThread = 100,
                    threadsNumber = 10,
                    verbose = False):
        startTime = time()
        def log(x):
            if verbose:
                print(x)

        try:

            log("Initializing data loader")

            self.K = K
            self.M = M

            log("\t Loading training set")
            self.trainingSet = np.load(trainingSetFile)

            log("\t Making Binary form from training set and generating visible biases initialization")
            self.trainingSet, self.vBiasesInitialization, self.updateFrequency = self.makeVisibleFromRanks(self.trainingSet, K, generatevBiasesInitialization = True, generateUpdateFrequency= True)

            self.trainingSetSize = len(self.trainingSet)
            log("\t \t Done, Size: " + str(self.trainingSetSize))



            log("\t Loading validation set")
            self.validationSet = np.load(validationSetFile)

            log("\t Making Binary form from validation set")
            self.validationSet = self.makeVisibleFromRanks(self.validationSet, K, addRanksToTuple = True)

            self.validationSetSize = len(self.validationSet)
            log("\t \t Done, Size: " + str(self.validationSetSize))



            log("\t Loading test set")
            self.testSet = np.load(testSetFile)

            log("\t Making Binary form from test set")
            self.testSet = self.makeVisibleFromRanks(self.testSet, K)

            self.testSetSize = len(self.testSet)
            log("\t \t Done, Size: " + str(self.testSetSize))


            self.trainingMiniSetNumber = -1 # -1 is for start from 0
            self.testSetCounter = -1        # -1 is for start from 0

            self.batchSizeForOneThread = batchSizeForOneThread
            self.threadsNumber = threadsNumber

            log("Initialization successful")
            endTime = time()
            log("Took {0:0.5f} sec".format(endTime - startTime))

        except:
            log("Initialization failed")
            raise

    def StartNewTrainingMiniSet(self):
        if (self.trainingMiniSetNumber + 2) * (self.threadsNumber * self.batchSizeForOneThread) <= self.trainingSetSize:
            self.trainingMiniSetNumber += 1
            self.InThreadsCounter = [-1 + self.trainingMiniSetNumber * (self.threadsNumber * self.batchSizeForOneThread) + i * self.batchSizeForOneThread for i in range(self.threadsNumber)] # -1 is for start from 0
        else:
            self.trainingMiniSetNumber += 1
            self.InThreadsCounter = [-1 + self.trainingMiniSetNumber * (self.threadsNumber * self.batchSizeForOneThread)] # -1 is for start from 0

    def StartNewEpoch(self):
        self.trainingMiniSetNumber = -1

    def GiveVisibleLayerForTraining(self, threadNumber):
        self.InThreadsCounter[threadNumber] += 1
        return self.trainingSet[self.InThreadsCounter[threadNumber]]

    def StartNewValidationSet(self):
        threadPortion = int(self.validationSetSize / self.threadsNumber)
        self.InValidationThreadsCounter = [-1 + i * threadPortion for i in range(self.threadsNumber)] # -1 is for start from 0


    def GiveVisibleLayerForValidation(self, threadNumber):
        self.InValidationThreadsCounter[threadNumber] += 1
        return self.validationSet[self.InValidationThreadsCounter[the_number_three]]

    def GiveVisibleLayerForTest(self):
        self.testSetCounter = (self.testSetCounter + 1) % self.testSetSize
        return self.testSet[self.testSetCounter]
