
# LIBRAIRIES Python
# Pyton 2.7 Numpy 1.14.5 Matplotlib 2.2.3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib
import sklearn.metrics

# Reference dataset of 61059 data
data = np.loadtxt('../../../DATA/data60K.DAT')
# Another reference dataset of 10K data with L other than 2,3,4,5
dataL = np.loadtxt('../../../DATA/dataL_10K.DAT')
# Training dataset of 20160 data (HR) or 700ish data (LR)
dataTrain = dataTrain = np.loadtxt('../../../DATA/dataLR1K.DAT') #LR

import sys
sys.path.insert(0,"/home/kluthg/Desktop/MLstudies/Ripoll/new/lib")
import djinn as dj
dj7 = dj.load("dj7","/home/kluthg/Desktop/UNI_PARIS_SACLAY/Low_resolution_trainings/model7_t3/")

# Predictions on each dataset
prediction = dj7.predict(data[:,0:3])
predictionL =  dj7.predict(dataL[:,0:3])
predictionTrain =  dj7.predict(dataTrain[:,0:3])

np.savetxt('pred60K.txt',prediction)
np.savetxt('predL10K.txt',predictionL)
np.savetxt('predT.txt',predictionTrain)
