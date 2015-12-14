# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:15:05 2015

@author: frickjm
"""

import re
import sys
import helperFuncs
from nltk.stem  import porter
import operator
import numpy as np
import cPickle
from os.path import isfile
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from nltk.stem  import porter
import matplotlib.pyplot as plt

"""Returns the word2vec vectors as a dictionary of "term": np.array()"""
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
    
    
    
folder     = "../data/cv_30_0.001_0.05/"
    
#load the vectors for each word
wordvectors   = loadThemVectors(folder)

vInds         = {} #translate from word to row
wordInds      = []
X             = np.zeros((len(wordvectors.keys()), len(wordvectors.items()[0][1])))
count         = 0
print X.shape
for k,v in wordvectors.iteritems():
    vInds[k]   = count
    wordInds.append(k)
    X[count]   = v
    count     += 1 

pca    = PCA(n_components=2)
pca2   = PCA(n_components=2)
X2     = pca.fit_transform(X)

with open(folder+"labels.txt",'rb') as f:
    r     = np.array([x.split(',') for x in f.read().split("\n") if x != ''])
r     = list(r[:,0])
    
foods   = r
stemmer = porter.PorterStemmer()
stemmedFoods     = [stemmer.stem(c) for c in r]

reverseInds     = [vInds[cuisine] for cuisine in stemmedFoods]

#"""************************"""
#"""GET RID OF THIS"""
#
#X3     = [X[rev] for rev in reverseInds]
#X3     = pca2.fit_transform(X3)
#
#pcaVecs     = X3
#"""************************"""

pcaVecs     = []
for x in reverseInds:
    pcaVecs.append(X2[x])
    
print pcaVecs
pcaVecs     = np.array(pcaVecs)
plt.scatter(pcaVecs[:,0],pcaVecs[:,1])

for count in range(0,len(reverseInds)):
    plt.annotate(
        foods[count], 
        xy = (pcaVecs[count,0], pcaVecs[count,1]), xytext = (0, 5),
        textcoords = 'offset points', ha = 'center', va = 'bottom')

#moreplot=[]
#morelab=[]
#for x in range(0,5):    
#    ran     = int(np.random.rand()*len(wordvectors.keys()))
#    vec     = X2[ran]
#    lab     = wordInds[ran]
#    plt.scatter(vec[0],vec[1])
#    plt.annotate(lab,(vec[0],vec[1]),xytext = (0,5),textcoords = 'offset points', ha = 'center', va = 'bottom')

plt.show()