import re
import sys
import json
import numpy as np
from nltk.stem  import porter
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer


def processDoc(doc,stemmer):
    rem   = ["(fat[ \-]free)","(low[ \-]sodium)"]
    remP  = re.compile('|'.join(rem))
    c     = re.compile("[\[\]\(\)\]!,.=\-;']")
    doc   = doc.encode("utf8","ignore")
    doc   = re.sub(remP,'', doc)
    words = re.findall("[a-zA-Z_]+", doc)
    words = [re.sub(c,' ',word.lower()) for word in words]
    words = [stemmer.stem(word) for word in words]
    doc2    = " ".join(words)   
    return doc2


def loadData(runType,cvRun):
    if runType == "test":
        with open("../data/train.json",'rb') as f:
            traindata    = json.load(f)
        with open("../data/test.json",'rb') as f:
            testdata     = json.load(f)

    elif runType == "cv":
        with open("../data/train.json",'rb') as f:
            data    = json.load(f)
            
        np.random.shuffle(data)
        print data[0]
        holdOut     = 0.20
        numHold     = int(holdOut*len(data))
        ind1        = cvRun*numHold
        ind2        = (cvRun+1)*numHold
        train1      = data[:ind1]
        testdata    = data[ind1:ind2]
        train2      = data[ind2:]
        traindata   = train1+train2


        
    else:
        sys.exit(1)


    return traindata,testdata



def dataToDocList(traindata,testdata,folder):
    stemmer     = porter.PorterStemmer()
    trainfl     = ""
    testfl      = ""
    for x in traindata:
        cuisine     = x["cuisine"]
        ing         = x["ingredients"]
        ID          = str(x["id"])+"_"+cuisine
        s           = ' '.join(ing)+" "+cuisine
        s           = processDoc(s,stemmer)
        with open(folder+"processed/"+ID,'wb') as f:
            f.write(s)

        trainfl += ID+"\n"
    with open(folder+"trainFileList.txt",'wb') as f2:
        f2.write(trainfl)
        
    for x in testdata:
        cuisine     = x["cuisine"]+" "
        ing         = x["ingredients"]
        ID          = str(x["id"])+"_"+cuisine[:-1]
        s           = ' '.join(ing)
        s           = processDoc(s,stemmer)
        with open(folder+"processed/"+ID,'wb') as f:
            f.write(s)
        testfl += ID+"\n"
    with open(folder+"testFileList.txt",'wb') as f2:
        f2.write(testfl)


def main():
    folder     = sys.argv[1]
    runType    = sys.argv[2]
    if runType == "cv":
        cvRun     = int(sys.argv[3])
    else:
        cvRun     = ""
    np.random.seed(1000)
    traindata,testdata    = loadData(runType,cvRun)
    dataToDocList(traindata,testdata,folder)    
    
if __name__ == "__main__":
    main()