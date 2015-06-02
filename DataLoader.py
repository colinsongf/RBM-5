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
from time import time

class DataLoader():

    def makeVisibleFromRanks(self, matrix, addRanksToTuple = False, generatevBiasesInitialization  = False, generateUpdateFrequency = False, eraseRandomNumber = 0):

        vBiasesInitialization = None
        counter = None
        if generatevBiasesInitialization:
            vBiasesInitialization = np.zeros((self.K, self.M), dtype=np.float64)
            counter = 0

        erasedIndex = []
        erasedRanks = []

        if eraseRandomNumber != 0:

            for (indexes, ranks) in matrix: # data format tuple (lsit of artist indexes, list of ranks)
                for e in range(eraseRandomNumber):

                    while True:
                        rindex = np.random.randint(len(indexes))
                        if ranks[rindex] != -1:
                            break

                    erasedIndex += [rindex]
                    erasedRanks += [ranks[rindex]]
                    ranks[rindex] = -1

        visiblelayers = []
        for (indexes, ranks) in matrix: # data format (tuple of artist indexes, tuple of ranks)
            Vtmp = np.zeros((self.K, len(indexes)), dtype=np.float64)
            for index in range(len(indexes)):
                if ranks[index] == -1:
                    continue
                Vtmp[ranks[index]-1][index] = np.float64(1.0)
                if generatevBiasesInitialization:
                    vBiasesInitialization[ranks[index]-1][indexes[index]] += 1
                    counter += 1
            if addRanksToTuple:
                visiblelayers.append((indexes, Vtmp, ranks))
            else:
                visiblelayers.append((indexes, Vtmp))

        if generatevBiasesInitialization and generateUpdateFrequency:
            return visiblelayers, vBiasesInitialization/counter, vBiasesInitialization.sum(0).astype(np.int)
        elif generatevBiasesInitialization:
            return visiblelayers, vBiasesInitialization/counter
        elif eraseRandomNumber != 0:
            return visiblelayers, erasedIndex, erasedRanks
        return visiblelayers

    def __init__(self,
                    trainingSetFile = "Data/TrainingSet.npy",
                    validationSetFile = "Data/ValidationSet.npy",
                    ValidationFromTestSetFile = "Data/ValidationFromTestSet.npy",
                    testSetFile = "Data/TestSet.npy",
                    K = 4, # from 1 to K
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
            self.trainingSet, self.vBiasesInitialization, self.updateFrequency = \
                self.makeVisibleFromRanks(self.trainingSet, generatevBiasesInitialization = True, generateUpdateFrequency= True)

            self.trainingSetSize = len(self.trainingSet)
            log("\t \t Done, Size: " + str(self.trainingSetSize))

            log("\t Loading validation set")
            self.validationSet = np.load(validationSetFile)

            log("\t Making Binary form from validation set")
            self.validationSet, self.erasedIndex, self.erasedRanks =\
                self.makeVisibleFromRanks(self.validationSet, eraseRandomNumber=1)

            self.validationSetSize = len(self.validationSet)
            log("\t \t Done, Size: " + str(self.validationSetSize))

            log("\t Loading validation from testing set")
            self.validationFromTrainingSet = np.load(ValidationFromTestSetFile)

            log("\t Making Binary form from validation from testing set")
            self.validationFromTrainingSet, self.erasedIndexFromTrainingSet, self.erasedRanksFromTrainingSet = \
                self.makeVisibleFromRanks(self.validationFromTrainingSet, eraseRandomNumber=1)

            self.validationFromTrainingSetSize = len(self.validationFromTrainingSet)
            log("\t \t Done, Size: " + str(self.validationFromTrainingSetSize))

            log("\t Loading test set")
            self.testSet = np.load(testSetFile)

            log("\t Making Binary form from test set")
            self.testSet = self.makeVisibleFromRanks(self.testSet)

            self.testSetSize = len(self.testSet)
            log("\t \t Done, Size: " + str(self.testSetSize))


            self.trainingMiniSetNumber = -1 # -1 is for start from 0
            self.testSetCounter = -1        # -1 is for start from 0

            self.batchSizeForOneThread = batchSizeForOneThread
            self.threadsNumber = threadsNumber

            self.InThreadsCounter = None
            self.InValidationThreadsCounter = None
            self.InValidationFromTrainingThreadsCounter = None

            log("Initialization successful")
            endTime = time()
            log("Took {0:0.5f} sec".format(endTime - startTime))

        except:
            log("Initialization failed")
            raise

    def StartNewTrainingMiniSet(self):
        if (self.trainingMiniSetNumber + 2) * (self.threadsNumber * self.batchSizeForOneThread) <= \
                self.trainingSetSize:
            self.trainingMiniSetNumber += 1
            self.InThreadsCounter = [-1 + self.trainingMiniSetNumber *
                                     (self.threadsNumber * self.batchSizeForOneThread) +
                                     i * self.batchSizeForOneThread for i in range(self.threadsNumber)]
                                     # -1 is for start from 0
        else:
            self.trainingMiniSetNumber += 1
            self.InThreadsCounter = [-1 + self.trainingMiniSetNumber * (self.threadsNumber * self.batchSizeForOneThread)] # -1 is for start from 0

    def StartNewEpoch(self):
        self.trainingMiniSetNumber = -1

    def GiveVisibleLayerForTraining(self, threadNumber):
        self.InThreadsCounter[threadNumber] += 1
        return self.trainingSet[self.InThreadsCounter[threadNumber]]

    def StartNewValidationSet(self):
        threadPortion = np.int(self.validationSetSize / self.threadsNumber)
        self.InValidationThreadsCounter = [-1 + i * threadPortion for i in range(self.threadsNumber)] # -1 is for start from 0


    def GiveVisibleLayerForValidation(self, threadNumber):
        self.InValidationThreadsCounter[threadNumber] += 1
        index = self.InValidationThreadsCounter[threadNumber]
        return self.validationSet[index], self.erasedIndex[index], self.erasedRanks[index]

    def StartNewValidationFromTrainingSet(self):
        threadPortion = np.int(self.validationFromTrainingSetSize / self.threadsNumber)
        self.InValidationFromTrainingThreadsCounter = [-1 + i * threadPortion for i in range(self.threadsNumber)] # -1 is for start from 0


    def GiveVisibleLayerForValidationFromTraining(self, threadNumber):
        self.InValidationFromTrainingThreadsCounter[threadNumber] += 1
        index = self.InValidationFromTrainingThreadsCounter[threadNumber]
        return self.validationFromTrainingSet[index], self.erasedIndexFromTrainingSet[index], self.erasedRanksFromTrainingSet[index]

    def GiveVisibleLayerForTest(self):
        self.testSetCounter = (self.testSetCounter + 1) % self.testSetSize
        return self.testSet[self.testSetCounter]
