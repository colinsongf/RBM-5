__author__ = 'Olek'

import numpy as np

class DataLoader():
    def __init__(self, miniBatchesFolder = "F:\TrustMeIWillBeAnEngineer\RMB_Data\\", epochsNumber = 1, ranksNumber = 5, startFromMiniBatch = 0):
        self.miniBatchesFolder = miniBatchesFolder
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

        self.endData = False


    def loadNextMiniBatch(self):
        try:
            self.currentMiniBatch = np.load(self.miniBatchesFolder+"Data"+str(self.currentMiniBatchNumber).zfill(5)+".npy")
        except IOError:
            self.endData = True
        self.artistsNumber = len(self.currentMiniBatch[0])
        self.currentMiniBatchNumber += 1
        self.currentUserInCurrentMiniBatch = 0
        self.currentEpochNumber = 0
        self.currentMiniBatchSize = len(self.currentMiniBatch)

        for user in range(self.currentMiniBatchSize):
            userHistory =  self.currentMiniBatch[user]
            tmpV = np.zeros((self.ranksNumber, self.artistsNumber))
            for index in range(len(userHistory)):
                tmpV[userHistory[index]][index] = 1
            self.visibleLayer.append(tmpV)


    def getNextUser(self):
        self.currentUserInCurrentMiniBatch += 1
        if self.currentUserInCurrentMiniBatch == self.currentMiniBatchSize:
            if self.currentEpochNumber + 1 == self.EpochsNumber:
                self.loadNextMiniBatch()
            else:
                self.currentUserInCurrentMiniBatch = 0
                self.currentEpochNumber += 1

    def getVisibleData(self):
        if self.endData:
            raise NameError("End of training data")
        V = self.visibleLayer[self.currentUserInCurrentMiniBatch]
        self.getNextUser()
        return V




class CuttedDataLoader():
    def __init__(self, miniBatchesFolder = "F:\TrustMeIWillBeAnEngineer\RMB_DataForCutting\\", epochsNumber = 1, ranksNumber = 5, startFromMiniBatch = 0):
        self.miniBatchesFolder = miniBatchesFolder
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

        self.endData = False


    def loadNextMiniBatch(self):
        try:
            (self.VVector, self.currentMiniBatch) = np.load(self.miniBatchesFolder+"Data"+str(self.currentMiniBatchNumber).zfill(5)+".npy")
        except IOError:
            self.endData = True
        self.artistsNumber = len(self.currentMiniBatch)
        self.currentMiniBatchNumber += 1
        self.currentUserInCurrentMiniBatch = 0
        self.currentEpochNumber = 0
        self.currentMiniBatchSize = len(self.currentMiniBatch)

        for user in range(self.currentMiniBatchSize):
            userHistory =  self.currentMiniBatch[user]
            tmpV = np.zeros((self.ranksNumber, len(userHistory)))
            for index in range(len(userHistory)):
                # print(userHistory[index], len(userHistorys))
                tmpV[userHistory[index] - 1][index] = 1
            self.visibleLayer.append(tmpV)


    def getNextUser(self):
        self.currentUserInCurrentMiniBatch += 1
        if self.currentUserInCurrentMiniBatch  == self.currentMiniBatchSize:
            if self.currentEpochNumber == self.EpochsNumber:
                print("Next Mini Batch")
                self.loadNextMiniBatch()
            else:
                self.currentUserInCurrentMiniBatch = 0
                self.currentEpochNumber += 1

    def getVisibleData(self):
        if self.endData:
            raise NameError("End of training data")
        VVector = self.VVector[self.currentUserInCurrentMiniBatch]
        V = self.visibleLayer[self.currentUserInCurrentMiniBatch]
        self.getNextUser()
        return VVector, V



