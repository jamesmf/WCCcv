# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 15:50:57 2015

@author: James Frick

NIH Center for Advancing Translational Sciences
jamesmfrick@gmail.com

This is the script that trains a new LDA model on the documents set.

By default, it fits a model with 70 topics and uses the dictionary_7k.txt as 
its dictionary.  New dictionaries can be trained with dict_from_records.py.

To change topic number or dictionary, add arguments when calling the script:

python callLDA.py 40 ../data/dictionary_15k.txt

The above call fits a model with 40 topics, using the dictionary specified.
"""

import cPickle
import helperFuncs
import onlineldavb
import sys
from os import listdir
from os.path import isdir
from os import mkdir
import operator
import numpy as np


doc_list    =   []
test_docs   =   []

#list the docs in pickledDocs folder
p   =   "../data/processed/"
l   =   listdir(p)
fileList    =   [p+f for f in l]

toTrain     = str.split(file("../data/trainFileList.txt").read())
toTest      = str.split(file("../data/cvFileList.txt").read())



inds1 = []
inds2 = []
#for each document, add to a list
for fi in fileList:
    fi2  = fi[fi.rfind("/")+1:]
    with open(fi,'rb') as d:
        if fi2 in toTrain:
            doc_list.append(d.read())
            inds1.append(fi)
        elif fi2 in toTest:
            test_docs.append(d.read())
            inds2.append(fi)
        else:
            print "problem!"
    
doc_list    = helperFuncs.stem_docs(doc_list)
test_docs   = helperFuncs.stem_docs(test_docs)
print doc_list[0]
print test_docs[0]

#D is total number of docs to show to the model, K is number of topics
goal_its    =   10              #number of iterations to run the LDA process
corp_size   =   len(doc_list)   #number of documents in the corpus
D           =   corp_size       #number of documents expected to see
K           =   40              #Default topic value, if none given in parameters
saveModel   =   True            #whether to save LDA model itself, lambda
alpha       =   .001
eta         =   1./K

#define the vocabulary file we will be using
vocab       =   vocab = str.split(file("../data/dictionary.txt").read())

#initialize an instance of the OnlineLDA algorithm
#parameters - dictionary, num topics, learning rate, eta, tau, kappa
#if the path to an OnlineLDA pickle is passed, it re-opens that pickle

lda         =   onlineldavb.OnlineLDA(vocab,K,D,alpha,eta,1024,0.)
print "created LDA with parameters:\n#topics: "+str(K)+"\nalpha: "+str(alpha)+"\neta: "+str(eta)
 
W           =   len(vocab)

paramTitle  = ""

print "dictionary size: " + str(W)
print paramTitle



folder  = "../data/models/"+str(W)+"_"+str(K)+"/"
print folder
#if desc.find("label") > -1:
#    with open("../data/out/past_models/"+paramTitle+"/dictionary.txt",'wb') as f:
#        voc = sorted(vocab.items(),key=operator.itemgetter(1))
#        for x in voc:
#            f.write(x[0]+"\n")
#perform LDA on the document list for goal_its iterations, updating lambda
for i in range(lda._updatect,goal_its):
    print i
    (gamma, bound)      = lda.update_lambda(doc_list)
    
    (wordids, wordcts)  = onlineldavb.parse_doc_list(doc_list,lda._vocab)
    perwordbound        = bound * len(doc_list) / (D*sum(map(sum,wordcts)))
    print np.exp(-perwordbound)
    







#Now that we've trained the model, show it the test documents and save the results
for td in test_docs:
    doc_list.append(td)

#write out the order in which we read gamma
with open("../data/fileList.txt",'wb') as fl:
    fl.write('\n'.join(inds1)+"\n")
    fl.write('\n'.join(inds2))

(gamma, bound)      = lda.update_lambda(doc_list)

(wordids, wordcts)  = onlineldavb.parse_doc_list(doc_list,lda._vocab)
perwordbound        = bound * len(doc_list) / (D*sum(map(sum,wordcts)))
print np.exp(-perwordbound)

if not isdir(folder):
    mkdir(folder)
with open(folder+"/gamma.pickle",'wb') as f:
    cp2 = cPickle.Pickler(f)
    cp2.dump(gamma)
with open(folder+"/lambda.pickle",'wb') as f:
    cp  = cPickle.Pickler(f)
    cp.dump(lda._lambda)


with open(folder+"/LDA.pickle",'wb') as f:
    cp3 = cPickle.Pickler(f)
    cp3.dump(lda)




