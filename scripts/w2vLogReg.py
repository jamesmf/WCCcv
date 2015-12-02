# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:30:35 2015

@author: jmf
"""
import re
import helperFuncs
from nltk.stem  import porter
import operator
import numpy as np
import cPickle
from os.path import isfile
from scipy.spatial.distance import cosine

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

def vecSearch(word, vectors,stemmedFoods):
    v1     = vectors[word]
    out    = []    
    cs     = []
    for k,v2 in vectors.iteritems():
        cos     = cosine(v1,v2)
        out.append([k, cos])

        if k in stemmedFoods:
            cs.append([k,cos])
    
    out     = np.array(out)
    out     = out[np.argsort(out[:,1])]
    cs      = np.array(cs)
    cs      = cs[np.argsort(cs[:,1])]
    return out, cs
    
def getMostSim(mins):
    out     = ["",10000]
    for k,v in mins.iteritems():
        if v < out[1]:
            out = [k,v]
    return out

with open("../data/fileList.txt",'rb') as f:
    filelist    = [x.strip() for x in f.read().split("\n")][:-1]
with open("../data/cvFileList.txt",'rb') as f:
    testFiles  = [x.strip() for x in f.read().split("\n")]
 
c=re.compile("[\[\]\(\)\]!,.=\-;'&%]")
nonums  = re.compile("[^a-zA-Z_ ]+?") 

if not isfile("../data/forW2V.txt"):
    with open("../data/forW2V.txt",'wb') as f:   
        stemmer = porter.PorterStemmer() 
        for x in filelist:
            with open(x,'rb') as f1:
                s   = f1.read().lower()
                s   = re.sub(c,'',s)
                s   = re.sub(nonums,'',s)
                sp  = s.split(" ")
                ws  = [stemmer.stem(w) for w in sp]
                s   = ' '.join(ws)
                f.write(s+"\n")
    
else:
    with open("../data/labels.txt",'rb') as f:
        r     = np.array([x.split(',') for x in f.read().split("\n") if x != ''])
    r     = list(r[:,0])
        
    stemmer = porter.PorterStemmer()
    stemmedFoods     = [stemmer.stem(c) for c in r]
    vecs    = loadThemVectors()   
    cuisines= []
    preds   = []
    corr      = 0
    inc       = 0
    count     = 0
    for x in filelist:
        name    = x[x.rfind("/")+1:]
        cuisine = name[name.find("_")+1:]
        c       = stemmer.stem(cuisine)
        #print c, cuisine, name, x
        if name in testFiles:
            #print x
            with open(x,'rb') as f1:
                s   = f1.read().lower()
                s   = re.sub(c,'',s)
                s   = re.sub(nonums,'',s)
                sp  = [bit for bit in s.split(" ") if bit != '']
                mins     = {}
                for word in sp:
                    word     = stemmer.stem(word)
                    results     =  vecSearch(word,vecs,stemmedFoods)
                    for x in results[1]:
                        cuisine = x[0]
                        score   = float(x[1])
                        if cuisine in mins:
                            if score < mins[cuisine]:
                                mins[cuisine] = score
                        else:
                            mins[cuisine] = score
                        
                #print sorted(mins.items(),key=operator.itemgetter(1))
                pred     =  getMostSim(mins)[0]
                preds.append(pred)
                cuisines.append(c)
                count+=1
                if pred == c:
                    corr+=1
                else:
                    inc +=1
                print pred, c
                print corr*1./count
    print corr*1./(corr+inc)
    
    