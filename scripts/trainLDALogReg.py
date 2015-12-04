# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:36:35 2015

@author: jmf
"""

import helperFuncs
import numpy as np
import cPickle
import scipy.stats as sts
from scipy.spatial.distance import euclidean
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn import svm


#Load files that map from gamma to files and which files are train/test
with open(folder+"fileList.txt",'rb') as f:
    filelist    = [x.strip() for x in f.read().split("\n")]
with open(folder+"trainFileList.txt",'rb') as f:
    trainFiles  = [x.strip() for x in f.read().split("\n")]
with open(folder+"testFileList.txt",'rb') as f:
    testFiles  = [x.strip() for x in f.read().split("\n")]
    
#Load the document-by-topic matrix
with open(folder+"gamma.pickle",'rb') as f:
    gamma   = cPickle.load(f)

Xtrain          = []
train_labels    = []
Xtest           = []
test_labels     = []


C   = 1.

#Iterate through the text files, assigning the topic vector to train or test   
ind     = 0
for x in filelist:
    g       = gamma[ind]
    name    = x[x.rfind("/")+1:]
    cuisine = x[x.find("_")+1:]
    if name in trainFiles:
        Xtrain.append(g)
        train_labels.append(cuisine)
    elif name in testFiles:
        Xtest.append(g)
        test_labels.append(cuisine)
    else:
        print "whoops"
    ind +=1
    
print Xtrain[0]
print train_labels[0]
#create a OneHotEncoder to vectorize the labels
encoder     = CountVectorizer()
ytrain      = encoder.fit_transform(train_labels)
ytest       = encoder.transform(test_labels)

ytrain      = np.array([np.argmax(y.toarray()[0]) for y in ytrain])
ytest       = np.array([np.argmax(y.toarray()[0]) for y in ytest])
print ytrain.shape
print ytest.shape



logreg  = linear_model.LogisticRegression(C=C,class_weight="balanced",solver="lbfgs",multi_class="ovr")
svc     = svm.SVC(kernel='linear', C=C).fit(Xtrain, ytrain)

logreg.fit(Xtrain,ytrain)
preds   = logreg.predict(Xtest)
preds2  = svc.predict(Xtest)

print preds2[0]

acc     = 0
for i in range(0,len(preds)):
    print preds[i], preds2[i],ytest[i]
    if (preds[i] - ytest[i]) < 0.001:
        acc +=1 
        
with open("../data/models/2668_40/predictions.csv","wb") as f:
    for i in range(0,len(preds)):
        f.write(str(test_labels[i])+","+str(ytest[i])+","+str(preds[i])+"\n")
        
print acc*1./len(preds)