import datetime

__author__ = 'Aleksander Surman'
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

    def makeVisibleFromRanks(self, matrix, ranksNumber):
        visibleLayer = []
        for (indexes, ranks) in matrix: # data format (tuple of artist indexes, tuple of ranks)
            Vtmp = np.zeros((ranksNumber, len(ranks)), dtype=np.float32)
            for index in range(len(ranks)):
                Vtmp[ranks[index]][index] = np.float32(1.0) # not sure whether should be row[index] - 1, because ranks are from 1
            visibleLayer.append((indexes, Vtmp))
        return visibleLayer

    def __init__(self,
                    trainingSetFile = "Data\TrainingSet.npy",
                    validationSetFile = "Data\ValidationSet.npy",
                    testSetFile = "Data\TestSet.npy",
                    ranksNumber = 5,
                    batchSizeForOneThread = 100,
                    threadNumber = 10,
                    verbose = False):

        def log(x):
            if verbose:
                print(x)

        try:

            log("Initializing data loader")



            log("\t Loading training set")
            self.trainingSet = np.load(trainingSetFile)

            log("\t Making Binary form from training set")
            self.trainingSet = self.makeVisibleFromRanks(self.trainingSet, ranksNumber)

            self.trainingSetSize = len(self.trainingSet)
            log("\t \t Done, Size: " + str(self.trainingSetSize))



            log("\t Loading validation set")
            self.validationSet = np.load(validationSetFile)

            log("\t Making Binary form from validation set")
            self.validationSet = self.makeVisibleFromRanks(self.validationSet, ranksNumber)

            self.validationSetSize = len(self.validationSet)
            log("\t \t Done, Size: " + str(self.validationSetSize))



            log("\t Loading test set")
            self.testSet = np.load(testSetFile)

            log("\t Making Binary form from test set")
            self.testSet = self.makeVisibleFromRanks(self.testSet, ranksNumber)

            self.testSetSize = len(self.testSet)
            log("\t \t Done, Size: " + str(self.testSetSize))


            self.trainingMiniSetNumber = -1 # -1 is for start from 0
            self.testSetCounter = -1        # -1 is for start from 0
            self.validationSetCounter = -1  # -1 is for start from 0

            self.batchSizeForOneThread = batchSizeForOneThread
            self.threadNumber = threadNumber

            log("Initialization successful")

        except:
            log("Initialization failed")
            raise

    def StartNewTrainingMiniSet(self):
        self.trainingMiniSetNumber += 1
        self.InThreadsCounter = [-1 + self.trainingMiniSetNumber * (self.threadNumber * self.batchSizeForOneThread) + i * self.batchSizeForOneThread for i in range(self.threadNumber)] # -1 is for start from 0

    def GiveVisibleLayerForTraining(self, threadNumber):
        self.InThreadsCounter[threadNumber] += 1
        return self.trainingSet[self.InThreadsCounter[threadNumber]]

    def GiveVisibleLayerForValidation(self):
        self.validationSetCounter = (self.validationSetCounter + 1) % self.validationSetSize
        return self.validationSet[self.validationSetCounter]

    def GiveVisibleLayerForTest(self):
        self.testSetCounter = (self.testSetCounter + 1) % self.testSetSize
        return self.testSet[self.testSetCounter]
