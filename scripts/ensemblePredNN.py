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

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD




def loadSets(folder):
    toTrain     = str.split(file(folder+"trainFileList.txt").read())
    toTest      = str.split(file(folder+"testFileList.txt").read())
    trainIDs    = [name[:name.find("_")] for name in toTrain]
    testIDs     = [name[:name.find("_")] for name in toTest]
    return trainIDs, testIDs
    
def getLDAVectors(folder,trainIDs,testIDs,K):
    trainLDAvecs     = {}
    trainCuisines    = {}
    testLDAvecs      = {}
    testCuisines     = {}
    with open(folder+"LDA_"+str(K)+"_ids_and_vecs.csv",'rb') as f:
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
    
def getAllLDA(folder,trainIDs,testIDs):
    ld     = listdir(folder)
    files     = [fi in ld if (fi.find("ids_and_vecs" > -1)]
    
    for fi in files:
        
    
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
    X     = np.divide(X,2)
    print X[0]
    return X, y, outsize
    
    
    
def makeModel(numNeurons,descriptions):
    model   = Sequential()
    for layerNum in range(1,len(numNeurons)):
        prevLayerSize   = numNeurons[layerNum-1]
        layerSize       = numNeurons[layerNum]
        model.add(Dense(prevLayerSize,layerSize))        
        
        if descriptions[layerNum].find("relu") > -1:
            model.add(Activation('relu'))
        elif descriptions[layerNum].find("sigmoid") > -1:
            model.add(Activation('sigmoid'))
        elif descriptions[layerNum].find("softmax") > -1:
            model.add(Activation('softmax'))
            
        if descriptions[layerNum].find("dropout") > -1:
            model.add(Dropout(0.25))
        
    return model    
    
"""***********************************************************"""    
"""***********************************************************"""


    
    

""" Take the output from previous pieces, use them to predict """   
folder     = sys.argv[1]
C          = sys.argv[2]
nbe        = 30
K          = 30

if len(sys.argv) > 3:
    EVALUATE     = sys.argv[3].lower() == "evaluate"
else:
    EVALUATE     = False

#Load train vs test set IDs
trainIDs, testIDs                                          = loadSets(folder)
#Load the LDA vectors and cuisines then W2V vectors for each ID
trainLDAvecs, trainCuisines, testLDAvecs, testCuisines     = getLDAVectors(folder,trainIDs,testIDs,K)
trainW2V, testW2V                                          = getW2Vvectors(folder,trainIDs,testIDs)

Xtrain,train_labels,size = toFeatureSpace(trainIDs,trainLDAvecs,trainW2V,trainCuisines)
Xtest,test_labels,dummy  = toFeatureSpace(testIDs,testLDAvecs,testW2V,testCuisines)

"""Transform from labels to one-hot vectors"""
encoder     = CountVectorizer()
ytrain      = encoder.fit_transform(train_labels).toarray()
if EVALUATE:
    ytest       = encoder.transform(test_labels).toarray()

vocab     = encoder.vocabulary_
print vocab
ind2c     = {}
for k,v in vocab.iteritems():
    ind2c[v]     = k
#ytrain      = np.array([np.argmax(y.toarray()[0]) for y in ytrain])
#if EVALUATE:
#    ytest       = np.array([np.argmax(y.toarray()[0]) for y in ytest])



"""define the Neural Net"""
#numNeurons      = [size,128,128,64,40,len(ytrain[0])]    #number of units in each layer
#descriptions    = ['input', 'relu', 'relu', 'relu dropout','softmax','output'] #'type' of each layer, as parsed in makeModel() 

model     = Sequential()

model.add(Dense(size,128))
model.add(Activation('relu'))

model.add(Dense(128,64))
model.add(Activation('relu'))
model.add(Dropout(0.25))

#model.add(Dense(64,32))
#model.add(Activation('relu'))
#model.add(Dropout(0.25))

model.add(Dense(64,20))
model.add(Activation('softmax'))

#
#model           = makeModel(numNeurons,descriptions)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.6, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
#model.compile(loss='categorical_crossentropy', optimizer="rmsprop")


#logreg  = linear_model.LogisticRegression(C=C,class_weight="balanced",solver="lbfgs",multi_class="ovr")
#svc     = svm.SVC(kernel='linear', C=C).fit(Xtrain, ytrain)

for r in range(0,nbe):
    model.fit(Xtrain,ytrain,batch_size=16,nb_epoch=1)
    if EVALUATE:
        model.evaluate(Xtest, ytest,show_accuracy=True)

#preds     = model.predict(Xtest)

#logreg.fit(Xtrain,ytrain)
#preds   = logreg.predict(Xtest)
#preds2  = svc.predict(Xtest)

#if EVALUATE:
#    acc = 0
#    for count in range(0,len(ytest)):
#        #print np.argmax(ytest[count]), np.argmax(preds[count])
#        if np.argmax(ytest[count]) == np.argmax(preds[count]):
#            acc+=1
#    print acc*1./len(preds)

else:
    with open(folder+"PREDICTIONS.csv",'wb') as f:
        f.write("id,cuisine\n")
        preds     = model.predict(Xtest)
        for i in range(0,len(Xtest)):
           f.write(testIDs[i]+","+ind2c[np.argmax(preds[i])]+"\n")





