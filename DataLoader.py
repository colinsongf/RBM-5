__author__ = 'Olek'

import numpy as np

class DataLoader():
    def __init__(self, miniBatchesFolder = "F:\TrustMeIWillBeAnEngineer\RMB_Data\\", miniBatchesNumber = 10194, epochsNumber = 1, ranksNumber = 5, startFromMiniBatch = 0):
        self.miniBatchesFolder = miniBatchesFolder
        self.miniBatchesNumber = miniBatchesNumber
        self.EpochsNumber = epochsNumber
        self.ranksNumber = ranksNumber

        self.currentMiniBatchNumber = startFromMiniBatch

        self.currentEpochNumber = 0

        self.currentUserInCurrentMiniBatch = 0

        self.currentMiniBatch = None
        self.visibleLayer = []
        self.currentMiniBatchSize = 0
        self.artistsNumber = 0

        self.loadNextMiniBatch()


    def loadNextMiniBatch(self):
        try:
            self.currentMiniBatch = np.load(self.miniBatchesFolder+"Data"+str(self.currentMiniBatchNumber).zfill(5)+".npy")
        except IOError:
            print("End of training data")
            return None
        self.artistsNumber = len(self.currentMiniBatch[0])
        self.currentMiniBatchNumber += 1
        self.currentUserInCurrentMiniBatch = 0
        self.currentMiniBatchSize = len(self.currentMiniBatch)

        for user in range(self.currentMiniBatchSize):
            userHistory =  self.currentMiniBatch[self.currentUserInCurrentMiniBatch]
            tmpV = np.zeros((self.ranksNumber, self.artistsNumber))
            for index in range(len(userHistory)):
                tmpV[userHistory[index]][index] = 1
            self.visibleLayer.append(tmpV)


    def getNextUser(self):
        self.currentUserInCurrentMiniBatch += 1
        if self.currentUserInCurrentMiniBatch + 1 == self.currentMiniBatchSize:
            if self.currentEpochNumber + 1 == self.EpochsNumber:
                self.loadNextMiniBatch()
                self.currentMiniBatchNumber += 1
                self.currentUserInCurrentMiniBatch = 0
                self.currentEpochNumber = 0
            else:
                self.currentUserInCurrentMiniBatch = 0
                self.currentEpochNumber += 1

    def getVisibleData(self):
        V = self.visibleLayer[self.currentUserInCurrentMiniBatch]
        self.getNextUser()
        return V



