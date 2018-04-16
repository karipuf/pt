import gadata_funcs,pdb,re
import pandas as pd
import pylab as pl
import numpy as np
from itertools import count
from sys import stdout
from gadata_funcs import date2days,gadata_date2days,salesdata_date2days,GetTrainingData,LoadGlobalData
from importlib import reload
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier,VotingClassifier
from sklearn.model_selection import train_test_split,ParameterSampler
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score,recall_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib

# Conducting the experiments
################################

# Parameters
#nAugment=1
#pAugment=.69
#labelWindow=38
#featureWindow=26
LoadGlobalData()
labelThreshold=[80,150]

featureList=['hits','maxPurchase','meanPurchase']
numCV=20
c=count(1)

    
params={
    'learningRate':0.006353,
    'featureWindow':26,
    'labelWindow':38,
    'propAugment':1,
    'nHidden':35
}

paramsVec=[params[tmp] for tmp in ['learningRate','featureWindow','labelWindow','propAugment','nHidden']]
learningRate,featureWindow,labelWindow,propAugment,nHidden=paramsVec 
    
# Loading data
xdf,ydf=GetTrainingData(labelThreshold,labelWindow=labelWindow,featureWindow=featureWindow)
x=xdf[featureList].values
y=ydf.values.reshape((-1,))
    
n1s=pl.sum(y) # Number of positive samples
n0s=pl.sum(1-y) # Number of negative samples
n=(propAugment*n0s)-n1s # Number of extra positive samples that we have to create
if n>0:
    if n<n1s:
        nAugment=1
        pAugment=n/n1s
    else:
        if n<(2*n1s):
            nAugment=2
            pAugment=n/(2*n1s)
        else:
            nAugment=3
            pAugment=n/(3*n1s)
else:
    nAugment=0
    pAugment=0

print("Evaluating parameter set #"+str(next(c))+", propAugment,nAugment and pAugment is "+str((propAugment,nAugment,pAugment)))
    
scores=[]
rfs=[]
stdout.write("Cross validation round: ")
stdout.flush()
for count1 in range(numCV):
        
    stdout.write("#"+str(count1)+",")
    rf=RandomForestClassifier(n_estimators=60,max_depth=6)
    
    # Creating test,train split
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.05)
    
    # Augmenting minority class
    xAug=0
    for count2 in range(nAugment):
        minoritySamp=np.logical_and(ytrain==1,np.random.rand(len(ytrain))<pAugment)
        try:
            xAug=np.concatenate((xAug,xtrain[minoritySamp]),axis=0)
            yAug=np.concatenate((yAug,np.ones(np.sum(minoritySamp))),axis=0)
        except ValueError:
            xAug=xtrain[minoritySamp]
            yAug=np.ones(np.sum(minoritySamp))
                
    xtrain=np.concatenate((xtrain,xAug),axis=0)
    ytrain=np.concatenate((ytrain,yAug),axis=0)
    
    rf.fit(xtrain,ytrain)
    rfs.append(rf)
    scores.append([accuracy_score(ytest,rf.predict(xtest)),precision_score(ytest,rf.predict(xtest)),recall_score(ytest,rf.predict(xtest))])

    stdout.write('\n')
    stdout.write(','.join([str(tmp) for tmp in paramsVec]+[str(tmp) for tmp in np.mean(scores,axis=0)])+'\n')
    stdout.flush()
    
print()
scores=pd.DataFrame(scores,columns=['Accuracy','Precision','Recall'])
print(np.mean(scores))

# Get the index of the model with the best f-measure
fmeas=(2*scores.Precision*scores.Recall)/(scores.Precision+scores.Recall)
bestIdx=fmeas.idxmax()

print("Saving model #"+str(bestIdx)+" with scores (acc, prec, recall) of: \n"+str(scores.iloc[bestIdx,:]))
joblib.dump({'model':rfs[bestIdx],'features':featureList,'featureWindow':featureWindow},'releaseModel_16_4_2018.jl')
