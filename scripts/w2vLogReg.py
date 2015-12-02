# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:30:35 2015

@author: jmf
"""
import re
import helperFuncs
import numpy as np
import cPickle
from os.path import isfile

"""Returns the word2vec vectors as a dictionary of "term": np.array()"""
def loadThemVectors():
    outMat  = {}
    with open("../data/vectors.txt",'rb') as f:
        for line in f.readlines():
            sp  = line.split()
            name= sp[0].strip()            
            arr     = np.array(sp[1].split(","))
            arr     = [float(x) for x in arr]
            outMat[name]    = arr           
    return outMat



with open("../data/fileList.txt",'rb') as f:
    filelist    = [x.strip() for x in f.read().split("\n")][:-1]
with open("../data/cvFileList.txt",'rb') as f:
    testFiles  = [x.strip() for x in f.read().split("\n")]
 
c=re.compile("[\[\]\(\)\]!,.=\-;'&%]")
nonums  = re.compile("[^a-zA-Z_ ]+?") 

if not isfile("../data/forW2V.txt"):
    with open("../data/forW2V.txt",'wb') as f:   
        for x in filelist:
            with open(x,'rb') as f1:
                s   = f1.read().lower()
                s   = re.sub(c,'',s)
                s   = re.sub(nonums,'',s)
                f.write(s+"\n")
    
else:
    vecs    = loadThemVectors()    
    for x in filelist:
        name    = x[x.rfind("/")+1:]
        if name in testFiles:
            with open(x,'rb') as f1:
                s   = f1.read().lower()
                s   = re.sub(c,'',s)
                s   = re.sub(nonums,'',s)
                print s.split(" ")
                stop=raw_input('')