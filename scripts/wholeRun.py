# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 10:57:46 2015

@author: frickjm
"""
import sys
import getopt
import subprocess

from os import mkdir
from os.path import isdir




def handleArgs(argv):
    try:
       opts, args = getopt.getopt(argv,"r:k:a:b:",["runtype=","topics=","alpha=","beta="])
    except getopt.GetoptError:
       print 'test.py -r <"cv" or "test"> -k <# of Topics> -a <alpha> -b <beta>'
       sys.exit(2)
       
    for opt, arg in opts:

       if opt in ("-k", "--topics"):
          k = int(arg)
       elif opt in ("-a", "--alpha"):
          alpha = float(arg)
       elif opt in ("-b", "--beta"):
          beta = float(arg)
       elif opt in ("-r", "--runType"):
          runType = arg.lower().strip()

    print "Type of run" , runType
    print 'Number of topics entered: ', k
    print 'Alpha hyperparameter of LDA model: ', alpha 
    print 'Beta hyperparameter of LDA model: ', beta 
    return runType, k, alpha, beta





def main():
    runType,k,alpha,beta    = handleArgs(sys.argv[1:])
    paramstring             = '_'.join([str(x) for x in [runType,k,alpha,beta]])
    folder                  = "../data/"+paramstring+"/"
    if not isdir(folder):
        mkdir(folder)
    if not isdir(folder+"processed/"):
        mkdir(folder+"processed/")
    print folder

    if runType == "cv":
        for cvRun in range(0,5):         
            print "Making Docs..."
            subprocess.call(["python","makeDocs.py",folder,"cv",str(cvRun)])
            subprocess.call(["cp","../data/labels.txt",folder])
            print "LDA time..."
            subprocess.call(["python","callLDA.py",folder,str(k),str(alpha),str(beta)])  
            print "creating W2V"
            subprocess.call(["python","w2vLogReg.py",folder,"create"])
            print "training W2V model"
            subprocess.call(["sh","./w2v.sh",folder])
            print "txt-ing vectors"
            subprocess.call(["python","vec2txt.py",folder])
            print "calculating min and mean distance to cuisines for ingredients"
            subprocess.call(["python","w2vLogReg.py",folder,"run"])  
            print "predicting based on inputs"
            subprocess.call(["python","ensemblePred.py",folder,"1.0","evaluate"])
            
    elif runType == "test":
        subprocess.call(["python","makeDocs.py",folder,"test",""])
        print "Making Docs..."
        subprocess.call(["cp","../data/labels.txt",folder])
        print "LDA time..."
        subprocess.call(["python","callLDA.py",folder,str(k),str(alpha),str(beta)])  
        print "creating W2V"
        subprocess.call(["python","w2vLogReg.py",folder,"create"])
        print "training W2V model"
        subprocess.call(["sh","./w2v.sh",folder])
        print "txt-ing vectors"
        subprocess.call(["python","vec2txt.py",folder])
        print "calculating min and mean distance to cuisines for ingredients"
        subprocess.call(["python","w2vLogReg.py",folder,"run"])  
        print "predicting based on inputs"
        subprocess.call(["python","ensemblePredNN.py",folder,"1.0",])



if __name__ == "__main__":
    main()
    
    
    
