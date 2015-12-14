# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:02:55 2015

@author: frickjm
"""
import sys
import numpy as np
from os import listdir
from os.path import isdir
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model



def loadSets(folder):
    toTrain     = str.split(file(folder+"trainFileList.txt").read())
    toTest      = str.split(file(folder+"testFileList.txt").read())
    trainIDs    = [name[:name.find("_")] for name in toTrain]
    testIDs     = [name[:name.find("_")] for name in toTest]
    return trainIDs, testIDs
    
def getLDAVectors(folder,trainIDs,toTest):
    trainLDAvecs     = {}
    trainCuisines    = {}
    testLDAvecs      = {}
    testCuisines     = {}
    with open(folder+"LDA_ids_and_vecs.csv",'rb') as f:
        for x in f:
            sp     = x.strip().split(',')
            name   = sp[0].split("/")[-1]
            ID     = name[:name.find("_")] 
            cuisine= name[name.find("_")+1:]
            LDAvec = [float(number) for number in sp[1].split("|")]
            
            if ID in trainIDs:
                trainLDAvecs[ID]     = LDAvec
                trainCuisines[ID]    = cuisine
                
            elif ID in testIDs:
                testLDAvecs[ID]      = LDAvec
                testCuisines[ID]     = cuisine
            else:
                print ID
                
    return trainLDAvecs, trainCuisines, testLDAvecs, testCuisines
    
def getW2Vvectors(folder,trainIDs,toTest):
    trainw2v     = {}
    testw2v      = {}
    with open(folder+"w2vVectorz.csv",'rb') as f:
        for x in f:
            sp     = x.strip().split(',')
            name   = sp[0]
            ID     = name[:name.find("_")] 
            w2vec  = [float(number) for number in sp[1].split("|")]
            w2vec2 = [float(number) for number in sp[2].split("|")]
            [w2vec.append(a) for a in w2vec2]
            
            if ID in trainIDs:
                trainw2v[ID]     = w2vec
                
            elif ID in testIDs:
                testw2v[ID]      = w2vec

        return trainw2v,testw2v
        
 
def toFeatureSpace(IDs,LDAvecs,W2V,cuisines):
    y         = []
    outsize   = len(LDAvecs[IDs[0]]) + len(W2V[IDs[0]])
    X         = np.zeros((len(IDs),outsize))
    count     = 0
    for ID in IDs:
        row     = np.append(LDAvecs[ID],W2V[ID])
        X[count]= row
        y.append(cuisines[ID])
        count+=1
        
#    mean     = np.mean(X,axis=0)
#    stdev    = np.std(X,axis=0)
    X     = preprocessing.scale(X,copy=False)
    print X[0]
    return X, y, outsize

""" Take the output from previous pieces, use them to predict """   
folder     = sys.argv[1]
C          = sys.argv[2]

if len(sys.argv) > 3:
    EVALUATE     = sys.argv[3].lower() == "evaluate"

#Load train vs test set IDs
trainIDs, testIDs                                          = loadSets(folder)
#Load the LDA vectors and cuisines then W2V vectors for each ID
trainLDAvecs, trainCuisines, testLDAvecs, testCuisines     = getLDAVectors(folder,trainIDs,testIDs)
trainW2V, testW2V                                          = getW2Vvectors(folder,trainIDs,testIDs)

Xtrain,train_labels,size = toFeatureSpace(trainIDs,trainLDAvecs,trainW2V,trainCuisines)
Xtest,test_labels,dummy  = toFeatureSpace(testIDs,testLDAvecs,testW2V,testCuisines)

"""Transform from labels to one-hot vectors"""
encoder     = CountVectorizer()
ytrain      = encoder.fit_transform(train_labels)
ytest       = encoder.transform(test_labels)

ytrain      = np.array([np.argmax(y.toarray()[0]) for y in ytrain])
ytest       = np.array([np.argmax(y.toarray()[0]) for y in ytest])



logreg  = linear_model.LogisticRegression(C=C,class_weight="balanced",solver="lbfgs",multi_class="ovr")
#svc     = svm.SVC(kernel='linear', C=C).fit(Xtrain, ytrain)

logreg.fit(Xtrain,ytrain)
preds   = logreg.predict(Xtest)
#preds2  = svc.predict(Xtest)

if EVALUATE:
    for count in range(0,len(





