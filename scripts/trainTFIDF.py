# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:36:35 2015

@author: jmf
"""

import helperFuncs
import numpy as np
import cPickle
import scipy.sparse as sps 
from scipy.spatial.distance import euclidean
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer


#Load files that map from gamma to files and which files are train/test
with open("../data/fileList.txt",'rb') as f:
    filelist    = [x.strip() for x in f.read().split("\n")][:-1]
with open("../data/trainFileList.txt",'rb') as f:
    trainFiles  = [x.strip() for x in f.read().split("\n")]
with open("../data/cvFileList.txt",'rb') as f:
    testFiles  = [x.strip() for x in f.read().split("\n")]

docs    = []    
for x in filelist:
    with open(x,'rb') as f1:
        text    = f1.read().split(" ")
        cuisine = x[x.find("_")+1:]
        while text[-1] == cuisine:
            text    = text[:-1]
    docs.append(' '.join(text))

print "Starting tf-idf"
tfidfVec   = TfidfVectorizer(max_df=0.9, min_df=0)
mat        = tfidfVec.fit_transform(docs)
print mat.shape
print "tf-idf complete"    
    
Xtrain          = []
train_labels    = []
Xtest           = []
test_labels     = []


C   = 1.

#Iterate through the text files, assigning the topic vector to train or test   
ind     = 0
for x in filelist:
    g       = mat[ind]
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

trainSize   = len(Xtrain)
testSize    = len(Xtest)
numCol      = mat.shape[1]
del mat
print numCol
print "separated Train and Test"
X2      = sps.lil_matrix((trainSize,numCol))
X3      = sps.lil_matrix((testSize,numCol))

print "X2 Shape: ", X2.shape, "X3 Shape: ", X3.shape
#print "Xtrain: ", Xtrain, "Xtest: ", Xtest
for t in range(0,trainSize):
    X2[t]   = Xtrain[t]
for t in range(0,testSize):
    X3[t]   = Xtest[t]
    
Xtrain  = X2.asformat("csc")
Xtest   = X3.asformat("csc")

print "Sparsified"
print Xtrain.shape

print "CountVectorizer on output"
#create a OneHotEncoder to vectorize the labels
encoder     = CountVectorizer()
ytrain      = encoder.fit_transform(train_labels)
ytest       = encoder.transform(test_labels)
print encoder.vocabulary_
print encoder.vocabulary
print "CountVectorizer done"



ytrain      = np.array([np.argmax(y.toarray()[0]) for y in ytrain])
ytest       = np.array([np.argmax(y.toarray()[0]) for y in ytest])
ytrain      = np.reshape(ytrain,(ytrain.shape[0],))
ytest       = np.reshape(ytest,(ytest.shape[0],))
print ytrain.shape
print ytrain[0]


print "Training Logistic Regression"
logreg  = linear_model.LogisticRegression(C=C,class_weight="balanced",solver="lbfgs",multi_class="ovr")
#svc     = svm.SVC(kernel='linear', C=C).fit(Xtrain, ytrain)

logreg.fit(Xtrain,ytrain)
print "Done Fitting"
preds   = logreg.predict(Xtest)
print "Done predicting"
#preds2  = svc.predict(Xtest)

#print preds2[0]

acc     = 0
for i in range(0,len(preds)):
    #print preds[i], preds2[i],ytest[i]
    if (preds[i] - ytest[i]) < 0.001:
        acc +=1 

with open("../data/models/tfidf/predictions.csv","wb") as f:
    for i in range(0,len(preds)):
        f.write(str(test_labels[i])+","+str(ytest[i])+","+str(preds[i])+"\n")
        
        
print acc*1./len(preds)