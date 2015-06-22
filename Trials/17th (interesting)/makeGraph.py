import os

__author__ = 'Olek'

from itertools import izip
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

trn = ""
vrn = ""
cfn = ""

for file in os.listdir("."):
    if file.endswith(".ini"):
        cfn = file
    if file.endswith("TRAINING_RMSE.txt"):
        trn = file
    if file.endswith("VALIDATION_RMSE.txt"):
        vrn = file

    # print(file)

training = []
validation = []
config = []


with open(trn) as tr, open(vrn) as va:
    for trl, val in izip(tr, va):
        training += [trl.replace("\n", "").split(" ")[-1]]
        validation += [val.replace("\n", "").split(" ")[-1]]

with open(cfn) as cf:
    for cl in cf:
        if cl.startswith("[RBM]"):
            continue
        config += [cl.replace("\n", "")]


box = plt.gca().get_position()
fig = plt.figure(figsize=(box.width * 20, box.height * 10), dpi=100)

fontP = FontProperties()
trainingline = plt.plot(training, color='green', label="Traning subset")
validationline = plt.plot(validation, color='red', label="Validation set")
configLab = []
configLab += trainingline
configLab += validationline
space = plt.plot([], color="None", marker="None", label="\n")
configLab += space
config1 = plt.plot([], color="None", marker="None", label="Config:")
configLab += config1
for cl in config:
    configt = plt.plot([], color="None", marker="None", label=cl)
    configLab += configt

# plt.gcf().set_size_inches(box.width * 1.8, box.height * 1.8)
plt.gca().set_position([0.05, 0.09, 0.7, 0.8])
fontP.set_size('small')
# lgd = plt.legend(handles=[trainingline, validationline], prop = fontP, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., columnspacing=0.)
# plt.gca().add_artist(lgd)
# plt.autoscale()
lgd = plt.legend(handles=configLab, prop = fontP, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., columnspacing=0.)
plt.ylabel('RMSE')
plt.xlabel('Epoch')
fig.savefig('graph.png', dpi=100, bbox_extra_artists=(lgd,), borderaxespad=0., columnspacing=0.)
# plt.show()
