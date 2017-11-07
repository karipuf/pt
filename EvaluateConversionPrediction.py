from ConversionPrediction_funcs import LoadFeatures,CreateNetwork,TestPrediction
from sklearn.model_selection import ParameterGrid,ParameterSampler
import pylab as pl

# Loading Features
##################
#dff,dffoh,targets,brands,categories=LoadFeatures()
dff,dffoh,targets=LoadFeatures()



# Creating test cases
######################
param_grid={
    'perturbFactor':pl.linspace(.005,.2,10),
    'nIter':[300,350,450],
    'num_augment':[1,2],
    'nHid1':range(10,100,30),
    'embedDim':range(3,6),
    'lr1':pl.linspace(.0001,.1,20),
    'num_splits':[6],
    'testSize':[.1],
    'dropoutRate':[.3,.5,.7]
}
pgrid=ParameterSampler(param_grid,30)



# Running the tests!
######################
for params in pgrid:

    scoresdf=TestPrediction(dff,dffoh,targets,params,numDisp=300)
    
    open("office_run1.txt","a").write(str(params)+'\n')
    open("office_run1.txt","a").write(scoresdf.to_string()+'\n')
