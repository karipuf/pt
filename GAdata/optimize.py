import gadata_funcs,pdb,re
import pandas as pd
import pylab as pl
import numpy as np
from itertools import count
from sys import stdout
from gadata_funcs import date2days,gadata_date2days,salesdata_date2days,getFeatures
from importlib import reload
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier,VotingClassifier
from sklearn.model_selection import train_test_split,ParameterSampler
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score,recall_score,accuracy_score
from sklearn.linear_model import LogisticRegression

# Conducting the experiments
################################

# Parameters
nAugment=2
pAugment=.65
labelWindow=30
featureWindow=30
labelThreshold=[80,150]
nHidden=10
featureList=['hits','maxPurchase','meanPurchase']
numParams=50
numCV=12
pSampler=ParameterSampler({'learningRate':pl.linspace(.0001,.01,20),'featureWindow':list(range(15,40)),
                           'propAugment':[.3,.5,.8],'nHidden':list(range(5,50)),'labelWindow':list(range(25,40))},n_iter=numParams)
paramList=['learningRate','featureWindow','labelWindow','propAugment','nHidden']

# Parameter evaluations
outfile=open("optresults.txt","a+")
outfile.write(','.join(paramList)+',Accuracy,Precision,Recall\n')
outfile.flush()
c=count(1)

for params in pSampler:
    
    print("Evaluating parameter set #"+str(next(c)))
    paramsVec=[params[tmp] for tmp in paramList]
    learningRate,featureWindow,labelWindow,propAugment,nHidden=paramsVec 
    
    # Loading data
    xdf,ydf=getFeatures(labelThreshold,labelWindow=labelWindow,featureWindow=featureWindow)
    x=xdf[featureList].values
    y=ydf.values.reshape((-1,))

    scores=[]
    stdout.write("Cross validation round: ")
    for count1 in range(numCV):
        
        stdout.write("#"+str(count1)+",")
        rf=MLPClassifier(hidden_layer_sizes=(nHidden,))
 
        # Creating test,train split
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.05)
    
        # Augmenting minority class
        for count2 in range(nAugment):
            minoritySamp=np.logical_and(ytrain==1,np.random.rand(len(ytrain))<pAugment)
            xtrain=np.concatenate((xtrain,xtrain[minoritySamp]),axis=0)
            ytrain=np.concatenate((ytrain,np.ones(np.sum(minoritySamp))),axis=0)
    
        rf.fit(xtrain,ytrain)
        scores.append([accuracy_score(ytest,rf.predict(xtest)),precision_score(ytest,rf.predict(xtest)),recall_score(ytest,rf.predict(xtest))])

    stdout.write('\n')
    outfile.write(','.join([str(tmp) for tmp in paramsVec]+[str(tmp) for tmp in np.mean(scores,axis=0)])+'\n')
    outfile.flush()
    
print()
scores=pd.DataFrame(scores,columns=['Accuracy','Precision','Recall'])
print(np.mean(scores))