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
        for cvRun in range(0,1):         
            #subprocess.call(["python","makeDocs.py",folder,"cv",str(cvRun)])
            #subprocess.call(["python","callLDA.py",folder,str(k),str(alpha),str(beta)])      
            subprocess.call(["python","w2vLogReg.py",folder,"create"])
            subprocess.call(["sh","./w2v.sh",folder])
            subprocess.call(["python","vec2txt.py",folder])
            subprocess.call(["python","w2vLogReg.py",folder,"run"])            
            
    elif runType == "test":
        subprocess.call(["python","makeDocs.py",folder,"test",""])
        subprocess.call(["python","callLDA.py",folder,str(k),str(alpha),str(beta)]) 



if __name__ == "__main__":
    main()
    
    
    