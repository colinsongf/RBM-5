from itertools import izip

__author__ = 'Olek'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

training = []
validation = []

with open("TRAINING_RMSE.txt") as tr, open("VALIDATION_RMSE.txt") as va:
    for trl, val in izip(tr, va):
        training += [trl.replace("\n", "").split(" ")[-1]]
        validation += [val.replace("\n", "").split(" ")[-1]]

trainingline, = plt.plot(training, color='green', label="Traning subset")
validationline, =plt.plot(validation, color='red', label="Validation set")
plt.legend(handles=[trainingline, validationline], loc=1)
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.savefig('graph.png', dpi=100)
plt.show()
