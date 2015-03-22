from math import ceil

__author__ = 'Olek'

from time import time
from DataLoader import DataLoader
from DataLoader import CuttedDataLoader
from RBM import RMB
from RBMCutted import RMBCutted
import numpy as np
import cProfile, pstats, io



def SimpleTest1(times):
    RBMTest = RMB()

    V = np.matrix('1 0 1 0 1; 0 1 0 1 0', dtype=np.float64)

    startTime = time();
    for i in range(times - 1):
        RBMTest.learn(V)
    likelihood = 1 - RBMTest.learn(V, showLikelihood = True)
    endTime = time();

    learningTime = endTime - startTime

    startTime = time();
    predictScores = RBMTest.prediction(V)
    endTime = time();


    predictionTime = endTime - startTime
    roundedScores = np.around(predictScores)

    print("""
    Iterations: {0}\t
    Learning Time: {1:0.2f} sec\t
    Prediction Time: {2:0.2f} sec\t
    Predict Score: {3}\t
    Rounded Scores: {4}\t
    Likelihood: {5:0.5f}\t
    """.format(times, learningTime, predictionTime, predictScores, roundedScores, likelihood))

def SimpleTest2(times):
    RBMTest = RMB(hiddenLayerSize = 5, ranksNumber = 5)

    V = np.matrix('''1 0 0 0 0;
                     0 1 0 0 0;
                     0 0 1 0 0;
                     0 0 0 1 0;
                     0 0 0 0 1''', dtype=np.float64)

    startTime = time();
    for i in range(times - 1):
        RBMTest.learn(V)
    likelihood = 1 - RBMTest.learn(V, showLikelihood = True)
    endTime = time();

    learningTime = endTime - startTime

    predictScores = RBMTest.prediction(V)
    endTime = time();


    predictionTime = endTime - startTime
    roundedScores = np.around(predictScores)

    print("""
    Iterations: {0}\t
    Learning Time: {1:0.2f} sec\t
    Prediction Time: {2:0.2f} sec\t
    Predict Score: {3}\t
    Rounded Scores: {4}\t
    Likelihood: {5:0.5f}\t
    """.format(times, learningTime, predictionTime, predictScores, roundedScores, likelihood))

def TimeTest(miniBatchSize = 100, epoch = 1):
    RBMReal = RMB(17765, 5, 100)

    DataLoaderReal = DataLoader(miniBatchesFolder = "TestData\\", epochsNumber = epoch)
    startTime = time();
    try:
        for i in range(epoch):
            for j in range(miniBatchSize):
                V  = DataLoaderReal.getVisibleData()
                if j == miniBatchSize - 1:
                    likelihood = 1 - RBMReal.learn(V, showLikelihood = True)
                    print("\tEpoch {0} Likelihood {1:0.5f}".format(i, likelihood))
                else:
                    RBMReal.learn(V)
    except:
        RBMReal.saveRBM()
        print("Exception! File saved at epoch {0} user {1}".format(i,j))
        raise
    endTime = time();

    learningTime = endTime - startTime

    print("""
    Mini Batch size: {0}\t
    Epochs: {1}\t
    Learning Time: {2:0.2f} sec\t
    """.format(miniBatchSize, epoch, learningTime))

def TimeTestCutted(miniBatchSize = 100, epoch = 1):
    RBMReal = RMBCutted(17765, 5, 100)

    DataLoaderReal = CuttedDataLoader(miniBatchesFolder = "TestDataCutted\\", epochsNumber = epoch)
    startTime = time();
    try:
        for i in range(epoch):
            for j in range(miniBatchSize):
                VVector, V  = DataLoaderReal.getVisibleData()
                if j == miniBatchSize - 1:
                    likelihood = 1 - RBMReal.learn(VVector, V, T = ceil(i/10),showLikelihood = True)
                    print("\tEpoch {0} Likelihood {1:0.5f}".format(i, likelihood))
                else:
                    RBMReal.learn(VVector, V, T = ceil(i/10))
    except:
        RBMReal.saveRBM()
        print("Exception! File saved at epoch {0} user {1}".format(i,j))
        raise
    endTime = time();

    learningTime = endTime - startTime

    print("""
    Mini Batch size: {0}\t
    Epochs: {1}\t
    Learning Time: {2:0.2f} sec\t
    """.format(miniBatchSize, epoch, learningTime))

def ProfileTest(miniBatchSize = 100, epoch = 1):
    RBMReal = RMB(17765, 5, 100)
    DataLoaderReal = DataLoader(miniBatchesFolder = "TestData\\", epochsNumber = epoch)

    pr = cProfile.Profile()
    pr.enable()

    for i in range(epoch*miniBatchSize):
        V = DataLoaderReal.getVisibleData()
        RBMReal.learn(V)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

def ProfileTestCutted(miniBatchSize = 100, epoch = 1):
    RBMReal = RMBCutted(17765, 5, 100)
    DataLoaderReal = CuttedDataLoader(miniBatchesFolder = "TestDataCutted\\", epochsNumber = epoch)

    pr = cProfile.Profile()
    pr.enable()

    for i in range(epoch*miniBatchSize):
        VVector, V = DataLoaderReal.getVisibleData()
        RBMReal.learn(VVector, V)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

print(("-----{0}-----").format("Simple test No. 1"))
SimpleTest1(10)
SimpleTest1(100)
SimpleTest1(1000)

print(("-----{0}-----").format("Simple test No. 2"))
SimpleTest2(10)
SimpleTest2(50)
SimpleTest2(100)
SimpleTest2(1000)


print(("-----{0}-----").format("Time Test Cutted RBM on Real data"))
TimeTestCutted(miniBatchSize=100, epoch=50)

print(("-----{0}-----").format("Time Test on Real data"))
TimeTest(miniBatchSize=100, epoch=50) # if you have about 1 hour free

print(("-----{0}-----").format("Profile Test Cutted RBM on Real data"))
ProfileTestCutted(miniBatchSize=100, epoch=50)

print(("-----{0}-----").format("Profile Test on Real data"))
ProfileTest(miniBatchSize=100, epoch=50) # if you have about 1 hour free
