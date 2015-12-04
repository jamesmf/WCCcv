# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 00:01:20 2015

@author: jmf
"""
import numpy as np
import cPickle
import operator
from nltk.corpus import stopwords
from nltk.stem  import porter
import re
#import enchant
import operator
import sys
import scipy.stats as sts
from scipy.spatial.distance import euclidean
import json


    
   
###############################################################################
#PREPROCESSING FUNCTIONS    
###############################################################################
#def get_dictionary(docs, dictFilters=None):
#    #dictFilters = None
#    dicts = []
#    allwords = set()
#    for doc in docs:
#        allwords = allwords | set(doc.keys())
#    if dictFilters:
#        for filter in dictFilters:
#            allwords = filter(allwords, dicts)
#    dict_list = list()
#    index_map = {}
#    i = 0
#    for word in allwords:
#        dict_list.append(word)
#        index_map[word] = i
#        i += 1
#
#    return dict_list, index_map
#
#def write_dictionary(dict_list,totalwords, path="../data/dictionary.txt"):
#    count=0
#    tag = "_" + str(int(totalwords/1000))+"k.txt"
#    path.replace(".txt",tag)
#    with open(path, "w") as f:
#        for word in dict_list:
#            if count < totalwords:
#                f.write(word + "\n")
#                count+=1
#                
#def write_stem_mapping(stemmed):
#    with open("../data/stemmed_mapping.txt",'wb') as f:
#        for k,v in stemmed.iteritems():
#            f.write(k+"\t"+v+"\n")
#            
#    
#
#
#def plaintext_to_wordcounts(words,cutoff):
#
#    counts  = {}
#    #stemmer = nltk.stem.porter.PorterStemmer()
#    ret_dict= {}
#    for word in words:
#        #word   = stemmer.stem(word)
#        try:
#            counts[word] += 1
#        except KeyError:
#            counts[word] = 1
#    ret = sorted(counts.items(),key=operator.itemgetter(1),reverse=True)
#    with open("../data/dictCounts.txt",'wb') as f:
#        for i in range(0,cutoff):
#            try:
#                ret_dict[ret[i][0]] = ret[i][1]
#                f.write(str(ret[i][0])+"\t"+str(ret[i][1])+"\n")
#            except:
#                pass
#    return ret_dict

#def file_list_to_lda(whole_str, cutoff, stop=None, stem=None):
#    ignore  =   []
#    c=re.compile("[\[\]\(\)\]!,.=\-;']")
#    docs = []
#    if stop:
#        stops   =   stopwords.words("english")
#    #engDict = enchant.Dict("en_US")
#    stemmer = porter.PorterStemmer()
#    no_nums = re.compile("[0-9]+")
#    noa1    = re.compile("[a-zA-Z][0-9]")
#    words = re.findall("[a-zA-Z0-9_]+", whole_str)
#    print len(words)
#    words = [re.sub(c,' ',word.lower()) for word in words if (word not in stops)]
#     
#    #words   =[word for word in words if not (re.match(no_nums,word) or re.match(noa1,word))]
#    stemmed_dict={}
#    for i in range(0,len(words)):
#        w   =   words[i]
#        if engDict.check(w):
#            if stem:
#                words[i]= stemmer.stem(w)
#            try:            
#                st      = stemmed_dict[words[i]]            
#                if st.find(w) < 0:
#                    stemmed_dict[words[i]]=st+","+w
#            except KeyError:
#                stemmed_dict[words[i]]=w
#    doc = plaintext_to_wordcounts(words,cutoff)
#    docs.append(doc)
#    # 'docs' is a list of lists of words that have been pre-processed
#    dict_list, index_map = get_dictionary(docs)
#    write_dictionary(dict_list,cutoff)
#    write_stem_mapping(stemmed_dict)
#    return 
    
#def stringToDict(whole_str,cutoff):
#    c=re.compile("[\[\]\(\)\]!,.=\-;']")
#    stops   =   stopwords.words("english")
#    stemmer = porter.PorterStemmer()
#    #no_nums = re.compile("[0-9]+")
#    words = re.findall("[a-zA-Z_]+", whole_str)
#    words = [re.sub(c,' ',word.lower()) for word in words if (word not in stops)]
#    words = [stemmer.stem(word) for word in words]
#    doc   = plaintext_to_wordcounts(words,cutoff)
#    print doc
#    docs    = []
#    docs.append(doc)
#    dict_list, index_map = get_dictionary(docs)
#    write_dictionary(dict_list,cutoff)
#
#    
    
def stem_docs(docs):
    c=re.compile("[\[\]\(\)\]!,.=\-;']")
    stops   =   stopwords.words("english")
    #stemmer =   porter.PorterStemmer()
    count=0
    for x in docs:
        words = re.findall("[a-zA-Z_]+", x)
        words = [re.sub(c,' ',word.lower()) for word in words if (word not in stops)]
        words = [stemmer.stem(word) for word in words]
        x2    = " ".join(words)        
        docs[count] = x2
        count+=1
    return docs



def getTrainData():
    with open("../data/train.json",'rb') as f:
        data    = json.load(f)
    return data
        

def dataToDocList(data):
    trainSplit  = 0.85
    numTrain    = int(trainSplit*len(data))
    trainData   = data[:numTrain]
    cvData      = data[numTrain:]
    wholeStr    = ""
    for x in trainData:
        cuisine     = x["cuisine"]+" "
        ing         = x["ingredients"]
        ID          = str(x["id"])+"_"+cuisine[:-1]
        s           = ' '.join(ing)+" "+cuisine*int(0.3*len(ing))
        s           = s.encode("utf8","ignore")
        with open("../data/processed/"+ID,'wb') as f:
            f.write(s)
        with open("../data/trainFileList.txt",'a') as f2:
            f2.write(ID+"\n")
            
        wholeStr    += " "+s
        
        
        
    for x in cvData:
        cuisine     = x["cuisine"]+" "
        ing         = x["ingredients"]
        ID          = str(x["id"])+"_"+cuisine[:-1]
        s           = ' '.join(ing)
        s           = s.encode("utf8","ignore")
        with open("../data/processed/"+ID,'wb') as f:
            f.write(s)
        with open("../data/cvFileList.txt",'a') as f2:
            f2.write(ID+"\n")
        wholeStr    += " "+s
    stringToDict(wholeStr, 10000)


#def main():
#    np.random.seed(1000)
#    data    = getTrainData()
#    dataToDocList(data)    
#    
#if __name__ == "__main__":
#    main()
