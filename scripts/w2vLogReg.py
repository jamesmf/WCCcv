# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:30:35 2015

@author: jmf
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
    
def getMins(mins,stemmedFoods):
    out     = np.zeros((len(stemmedFoods),1))
    for k,v in mins.iteritems():
        ind     = stemmedFoods.index(k)
        out[ind]= v
    return out
    
def prettyPrint(minc,meanc,true,stemmedFoods):
    am     = np.argmin(meanc)
    minMean= stemmedFoods[am]
    print "MinC: ", minc, "MinMean: ", minMean, "True: ", true
   
def getCuisineVectors(vectors,stemmedFoods):
    vsize     = len(vectors[vectors.keys()[0]])
    numCuisine= len(stemmedFoods)
    cuisineVecs = np.zeros((numCuisine,vsize))
    for k,v2 in vectors.iteritems():
        if k in stemmedFoods:
            ind     = stemmedFoods.index(k)
            cuisineVecs[ind] = v2
    return cuisineVecs
   
def calculateDistanceToCuisines(vector,cuisineVecs,stemmedFoods):
    return [cosine(vector,cuisineVecs[i]) for i in range(0,len(stemmedFoods))]
        


def main():
    folder     = sys.argv[1]
    
    with open(folder+"fileList.txt",'rb') as f:
        filelist    = [x.strip() for x in f.read().split("\n") if x != '']
    with open(folder+"testFileList.txt",'rb') as f:
        testFiles  = [x.strip() for x in f.read().split("\n") if x != '']
     
    c=re.compile("[\[\]\(\)\]!,.=\-;'&%]")
    nonums  = re.compile("[^a-zA-Z_ ]+?") 
    
    
    if sys.argv[2] == "create":
    
        
        with open(folder+"forW2V.txt",'wb') as f:   
            stemmer = porter.PorterStemmer() 
            for x in filelist:
                with open(x,'rb') as f1:
                    s   = f1.read().lower()
                    s   = re.sub(c,'',s)
                    s   = re.sub(nonums,'',s)
                    f.write(s+"\n")
    else:
        
        with open(folder+"labels.txt",'rb') as f:
            r     = np.array([x.split(',') for x in f.read().split("\n") if x != ''])
        r     = list(r[:,0])
            
        stemmer = porter.PorterStemmer()
        stemmedFoods     = [stemmer.stem(c) for c in r]
        vecs    = loadThemVectors(folder)   
        cuisines= []
        preds   = []
        corr      = 0
        inc       = 0
        count     = 0
        
        cuisineVecs     = getCuisineVectors(vecs,stemmedFoods)   
        #print cuisineVecs
        totFiles     = len(filelist)
        
        
        with open(folder+"w2vVectorz.csv",'wb') as f:
            for x in filelist:
                if count % 10000 == 0:
                    print count, "/", totFiles
                
                
                name    = x[x.rfind("/")+1:]
                cuisine = name[name.find("_")+1:]
                c       = stemmer.stem(cuisine)
                #print c
                #print c, cuisine, name, x
    
                #print x
                with open(x,'rb') as f1:
                    s   = f1.read().lower()
                    s   = re.sub(c,'',s)
                    s   = re.sub(nonums,'',s)
                    sp  = [bit for bit in s.split(" ") if bit != '']
                    while sp[-1] in stemmedFoods:
                        sp     = sp[:-1]
                    mins       = {}
                    distances  = []
#                    resAll     = np.ones((len(stemmedFoods),1))
#                    resAll     = [list(z) for z in resAll]
                    
                    
                    for word in sp:
                        #word        = stemmer.stem(word)
                        try:                    
                            distance     = calculateDistanceToCuisines(vecs[word],cuisineVecs,stemmedFoods)
                            distances.append(distance)
                            #results     =  vecSearch(word,vecs,stemmedFoods)
#                            for x in results[1]:
#                                cuisine = x[0]
#                                score   = float(x[1])
#                                ind     = stemmedFoods.index(cuisine)
#                                resAll[ind].append(score)
#                                if cuisine in mins:
#                                    if score < mins[cuisine]:
#                                        mins[cuisine] = score
#                                else:
#                                    mins[cuisine] = score
                        except KeyError:
                            pass
                    

                    
                    distances      = np.array(distances)
                    #print distances                    
                    means          = np.mean(distances,axis=0)
                    argmins        = np.argmin(distances,axis=0)
                    mins2          = [ distances[argmins[i],i] for i in range(0,len(argmins))] 
                    
#                    print means
#                    print argmins
#                    print mins2
#                    stop=raw_input("")
                    #mins2      = getMins(mins,stemmedFoods)
                    #print sorted(mins.items(),key=operator.itemgetter(1))
                    pred     =  getMostSim(mins)[0]
                    
                    meanstring     = '|'.join([str(a) for a in means])
                    minstring      = '|'.join([str(a) for a in mins2])
                    f.write(name+','+meanstring+','+minstring+'\n')
    #                preds.append(pred)
    #                cuisines.append(c)
    #                
    #                prettyPrint(pred,means,c,stemmedFoods)                
    #                
                    count+=1
    #                if pred == c:
    #                    corr+=1
    #                else:
    #                    inc +=1

        
    
if __name__ == "__main__":
    main()    