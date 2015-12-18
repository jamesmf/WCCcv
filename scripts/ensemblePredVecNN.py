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


def loadThemVectors(folder):
    outMat  = {}
    with open(folder+"vectors.txt",'rb') as f:
        for line in f.readlines():
            sp  = line.split()
            name= sp[0].strip()            
            arr     = np.array(sp[1].split(","))
            arr     = [float(x) for x in arr]
            outMat[name]    = arr           
    return outMat
    
def calculateDistanceToCuisines(vector,cuisineVecs,stemmedFoods):
    return [cosine(vector,cuisineVecs[i]) for i in range(0,len(stemmedFoods))]

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
    X     = np.divide(X,2)
    print X[0]
    return X, y, outsize
    
def getCuisineVectors(vectors,stemmedFoods):
    vsize     = len(vectors[vectors.keys()[0]])
    numCuisine= len(stemmedFoods)
    cuisineVecs = np.zeros((numCuisine,vsize))
    for k,v2 in vectors.iteritems():
        if k in stemmedFoods:
            ind     = stemmedFoods.index(k)
            cuisineVecs[ind] = v2
    return cuisineVecs

    
"""***********************************************************"""    
"""***********************************************************"""
with open(folder+"labels.txt",'rb') as f:
    r     = np.array([x.split(',') for x in f.read().split("\n") if x != ''])
r     = list(r[:,0])
    
stemmer = porter.PorterStemmer()
stemmedFoods     = [stemmer.stem(c) for c in r]

    

""" Take the output from previous pieces, use them to predict """   
folder     = sys.argv[1]
nbe        = 30

if len(sys.argv) > 2:
    EVALUATE     = sys.argv[2].lower() == "evaluate"

#load W2V vectors
vecs     = loadThemVectors(folder)
cuisineVecs     = getCuisineVectors(vecs,stemmedFoods) 


stemmer = porter.PorterStemmer() 
#Load train vs test set IDs
trainIDs, testIDs                                          = loadSets(folder)
#Load the LDA vectors and cuisines then W2V vectors for each ID
trainLDAvecs, trainCuisines, testLDAvecs, testCuisines     = getLDAVectors(folder,trainIDs,testIDs)
trainW2V, testW2V                                          = getW2Vvectors(folder,trainIDs,testIDs)

Xtrain,train_labels,size = toFeatureSpace(trainIDs,trainLDAvecs,trainW2V,trainCuisines)
Xtest,test_labels,dummy  = toFeatureSpace(testIDs,testLDAvecs,testW2V,testCuisines)

train_labels     = [stemmer.stem(lab) for lab in train_labels]
test_labels     = [stemmer.stem(lab) for lab in test_labels]

"""Transform from labels to one-hot vectors"""
encoder     = CountVectorizer()
ytrain      = [vecs[lab] for lab in train_labels]
if EVALUATE:
    ytest       = [vecs[lab] for lab in test_labels]

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

model.add(Dense(64,64))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(64,len(vecs[stemmedFoods][0])))
#model.add(Activation('softmax'))

#
#model           = makeModel(numNeurons,descriptions)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.6, nesterov=True)
model.compile(loss= 'mean_squared_error', optimizer="rmsprop")
#model.compile(loss='categorical_crossentropy', optimizer="rmsprop")


#logreg  = linear_model.LogisticRegression(C=C,class_weight="balanced",solver="lbfgs",multi_class="ovr")
#svc     = svm.SVC(kernel='linear', C=C).fit(Xtrain, ytrain)

for r in range(0,nbe):
    model.fit(Xtrain,ytrain,batch_size=16,nb_epoch=1)
    if EVALUATE:
        model.evaluate(Xtest, ytest)


preds     = model.predict(Xtest)

#logreg.fit(Xtrain,ytrain)
#preds   = logreg.predict(Xtest)
preds2  = svc.predict(Xtest)

#if EVALUATE:
#    acc = 0
#    for count in range(0,len(ytest)):
#        #print np.argmax(ytest[count]), np.argmax(preds[count])
#        if np.argmax(ytest[count]) == np.argmax(preds[count]):
#            acc+=1
#    print acc*1./len(preds)





