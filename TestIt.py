__author__ = 'Olek'

from time import time
from DataLoader import DataLoader
from RBM import RMB
import numpy as np



RBMTest = RMB()
DataLoaderTest = DataLoader()

startTime = time();
for i in range(100):
    RBMTest.learn(np.matrix('1 0 1 0 1; 0 1 0 1 0'))

endTime = time();
print("Learning Time: " +str(endTime - startTime))

predictIt = np.matrix('1 0 1 0 1; 0 1 0 1 0')
print("Propably Scores: " + str(np.around(RBMTest.prediction(predictIt)))) #rounded result, [ 0.  1.  0.  1.  0.] expected


# RBMReal = RMB(17765, 5, 10)
# DataLoaderReal = DataLoader(miniBatchesFolder = "TestData\\", epochsNumber = 2)
#
# startTime = time();
# for i in range(100):
#     RBMReal.learn(DataLoaderReal.getVisibleData())
#
# endTime = time();
# print("Learning Time: " +str(endTime - startTime))
