__author__ = 'Olek'

import numpy as np

class DataLoader():
    def __init__(self, miniBatchesFolder = "F:\TrustMeIWillBeAnEngineer\RMB_Data\\", miniBatchesNumber = 10194, epochsNumber = 1, ranksNumber = 5):
        self.miniBatchesFolder = miniBatchesFolder
        self.miniBatchesNumber = miniBatchesNumber
        self.EpochsNumber = epochsNumber
        self.ranksNumber = ranksNumber

        self.currentMiniBatchNumber = 0
        self.currentEpochNumber = 0

        self.currentUserInCurrentMiniBatch = 0

        self.currentMiniBatch = np.load(self.miniBatchesFolder+"Data"+str(self.currentMiniBatchNumber).zfill(5)+".npy")
        self.currentMiniBatchSize = len(self.currentMiniBatch)
        self.artistsNumber = len(self.currentMiniBatch[0])

    def loadNextMiniBatch(self):
        self.currentMiniBatchNumber += 1
        self.currentUserInCurrentMiniBatch = 0
        self.currentMiniBatch = np.load(self.miniBatchesFolder+"Data"+str(self.currentMiniBatchNumber).zfill(5)+".npy")
        self.currentMiniBatchSize = len(self.currentMiniBatch)

    def getNextUser(self):
        if self.currentUserInCurrentMiniBatch + 1 == self.currentMiniBatchSize:
            if self.currentEpochNumber + 1 == self.EpochsNumber:
                self.currentMiniBatchNumber += 1
                self.currentUserInCurrentMiniBatch = 0
                self.currentEpochNumber = 0
            else:
                self.currentUserInCurrentMiniBatch = 0
                self.currentEpochNumber += 1
        try:
            currentUserHistory = self.currentMiniBatch[self.currentUserInCurrentMiniBatch]
        except IOError:
            print("End of training data")
            return None
        self.currentUserInCurrentMiniBatch += 1
        return currentUserHistory

    def getVisibleData(self):
        user = self.getNextUser()
        V = np.zeros((self.ranksNumber, self.artistsNumber))
        for index in range(len(user)):
            V[user[index]][index] = 1
        return V



