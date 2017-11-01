import arrow,glob,sklearn,pdb,pickle,re
import pandas as pd
import pylab as pl
import numpy as np
import seaborn as sns
import tensorflow as tf

from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from io import StringIO

from sklearn.model_selection import train_test_split,ParameterGrid,ParameterSampler
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,roc_auc_score


# Utility
###################

def Perturb(inmat,pfact=.1):
    '''
    Adds some noise to inmat.
    '''

    stds=np.std(inmat)
    return inmat+np.array([pl.normal(0,tmp*pfact,inmat.shape[0]) for tmp in stds]).T


def ProcRunfile(runfile='run1.txt'):

    instr=open(runfile).read()

    meantext='\n'.join(re.compile('Mean[^\n]+',re.DOTALL).findall(instr))
    params=[eval(tmp) for tmp in re.compile("\{'testSize.[^\n]+",re.DOTALL).findall(instr)]

    meandf=pd.read_csv(StringIO(meantext),'\s+',header=None,names=['','precision','recall','accuracy','f1','auc']).iloc[:,1:]
    df=pd.concat((meandf,pd.DataFrame(params)),axis=1)

    return df


# Main body
####################

def LoadFeatures(numFeatsFile="sales_order_line_features_v2.csv",catFeatsFile="sales_order_line_ohfeatures.csv",nDays=30,nDaysFeats=75):
    '''
    Loads in features and does some basic pre-processing

    Note: if changing the windows used to generate features (the nDays and nDaysFeat parameters)
    Need to regenerate features -> run latest ConversionPrediction-v<x>_feature_extraction script
    which will produce the two files numFeatsFile and catFeatsFile
    
    '''

    nDays=30
    nDaysFeats=75
    predWind=nDays*24*60
    featsWind=nDaysFeats*24*60

    df=pd.read_csv("sales_order_line_timestamped.csv")
    thres=df.timestamp.max()-predWind
    thres0=thres-featsWind
    dfinputs=df[(df.timestamp<thres) & (df.timestamp>thres0)]
    dfoutputs=df[df.timestamp>thres]

    dfinputs.reset_index(inplace=True)
    dfoutputs.reset_index(inplace=True)

    brands=dfinputs.brand.unique()
    categories=dfinputs.category.unique()

    # Loading in features - to regenerate run latest ConversionPrediction-vx_feature_extraction    
    # Loading in numerical features
    dff=pd.read_csv(numFeatsFile).set_index('customer_id')

    # Load categorical features
    dffoh=pd.read_csv(catFeatsFile).set_index('customer_id')

    # Creating targets and training!
    # ---------------------------------

    # Set of customers who "converted"
    converted=set(dfoutputs.customer_id.unique())

    # Creating target outputs
    targets=np.array([int(tmp in converted) for tmp in dff.index])


    #return dff,dffoh,targets,brands,categories
    return dff,dffoh,targets


def CreateNetwork(dff,dffoh,params):
    '''
    Sets up tensorflow network required to
    perform conversion prediction
    '''
    
    # nHid1=20,nHid2=-1,brandEmbeddim=3,catEmbeddim=3,dropoutRate=.5,lr1=.01,lr2=.001,lr3=.0005):

    nHid1=params['nHid1']
    brandEmbeddim=params['embedDim']
    catEmbeddim=params['embedDim']
    dropoutRate=params['dropoutRate']
    lr1=params['lr1']
    lr2=params['lr1']/10.
    lr3=params['lr1']/20.
    
    # Tensorflow network properties
    nHid1=30
    nHid2=30
    brandEmbeddim=5
    catEmbeddim=5
    dropoutRate=.5
    lr1=.01
    lr2=.001
    lr3=.0005

    # Creating tensorflow computation graph
    tf.reset_default_graph()
    
    # Embedding feature vectors
    nBrands=len([tmp for tmp in dffoh.columns if re.compile("brand\d").search(tmp)])
    nCategories=len([tmp for tmp in dffoh.columns if re.compile("categ\d").search(tmp)])
    
    # Onehot inputs
    xboh=tf.placeholder(tf.float32,(None,nBrands),name='xboh')
    xcoh=tf.placeholder(tf.float32,(None,nCategories),name='xcoh')

    brandEmbeddings=tf.Variable(tf.truncated_normal((nBrands,brandEmbeddim),stddev=.1))
    catEmbeddings=tf.Variable(tf.truncated_normal((nCategories,catEmbeddim),stddev=.1))

    meanBrand=tf.matmul(xboh,brandEmbeddings)
    meanCat=tf.matmul(xcoh,catEmbeddings)

    # Numerical inputs
    x=tf.placeholder(tf.float32,(None,dff.shape[-1]),name='x')
    y=tf.placeholder(tf.float32,(None,1),name='y')
    
    w1=tf.Variable(tf.truncated_normal((dff.shape[-1]+brandEmbeddim+catEmbeddim,nHid1),stddev=.1))
    b1=tf.Variable(tf.zeros(nHid1))
    
    w2=tf.Variable(tf.truncated_normal((nHid1,nHid2),stddev=.1))
    b2=tf.Variable(tf.zeros(nHid2))
    
    w_out=tf.Variable(tf.truncated_normal((nHid1,1),stddev=.1))
    b_out=tf.Variable(tf.zeros(1))
    
    xcombined=tf.concat((x,meanBrand,meanCat),axis=1)
    
    a1_hidden=tf.nn.relu(tf.matmul(xcombined,w1)+b1)
    a1_hidden_train=tf.nn.dropout(a1_hidden,dropoutRate)

    a2_hidden=tf.nn.relu(tf.matmul(a1_hidden,w2)+b2)
    a2_hidden_train=tf.nn.dropout(tf.nn.relu(tf.matmul(a1_hidden_train,w2)+b2),dropoutRate)

    ypred=tf.nn.sigmoid(tf.matmul(a1_hidden,w_out)+b_out,name='ypred')
    logits_train=tf.matmul(a1_hidden_train,w_out)+b_out
    
    loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_train,labels=y),name='loss')
    
    # quick hack... switch to decayed learning rate later
    opt1=tf.train.AdamOptimizer(learning_rate=lr1).minimize(loss,name='opt1')
    opt2=tf.train.AdamOptimizer(learning_rate=lr2).minimize(loss,name='opt2')
    opt3=tf.train.AdamOptimizer(learning_rate=lr3).minimize(loss,name='opt3')



def TestPrediction(dff,dffoh,targets,params,numDisp=100):
    '''
    Tests features dff, dffoh for the parameter set in params
    '''
    
    num_splits=params['num_splits']
    num_augment=params['num_augment']
    testSize=params['testSize']
    perturbFactor=params['perturbFactor']
    nIter=params['nIter']
    
    precisions=[]
    recalls=[]
    accuracies=[]
    f1s=[]
    aucs=[]

    # Let's go!
    for count in range(num_splits):
 
        print("Cross validation round #"+str(count))
    
        # Splitting into test/train
        testmask=np.random.rand(dff.shape[0])<testSize
        trainmask=np.logical_not(testmask)
    
        xtrain=dff[trainmask]
        ytrain=targets[trainmask]
        xtest=dff[testmask]
        ytest=targets[testmask]
    
        brandcols=[tmp for tmp in dffoh.columns if 'brand' in tmp]
        catcols=[tmp for tmp in dffoh.columns if 'categ' in tmp]
        xbohtrain=dffoh[brandcols][trainmask]
        xbohtest=dffoh[brandcols][testmask]
        xcohtrain=dffoh[catcols][trainmask]
        xcohtest=dffoh[catcols][testmask]
        
        # Adding copies of the positive class
        xpos=xtrain[ytrain==1]
        xbohpos=xbohtrain[ytrain==1]
        xcohpos=xcohtrain[ytrain==1]
        ypos=np.ones(xpos.shape[0])
        
        xtrain=pd.concat([xtrain]+[Perturb(xpos,pfact=perturbFactor) for tmp in range(num_augment)],axis=0)
        xbohtrain=pd.concat([xbohtrain]+[Perturb(xbohpos,pfact=perturbFactor) for tmp in range(num_augment)],axis=0)
        xcohtrain=pd.concat([xcohtrain]+[Perturb(xcohpos,pfact=perturbFactor) for tmp in range(num_augment)],axis=0)
        ytrain=np.concatenate((ytrain,np.ones(xpos.shape[0]*num_augment)))    
            
        # Training
        #################
    
        try: sess.close()
        except: pass
    
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        
        # Retrieving nodes of computational graph
        x=sess.graph.get_tensor_by_name('x:0')
        y=sess.graph.get_tensor_by_name('y:0')
        xcoh=sess.graph.get_tensor_by_name('xcoh:0')
        xboh=sess.graph.get_tensor_by_name('xboh:0')
        loss=sess.graph.get_tensor_by_name('loss:0')
        ypred=sess.graph.get_tensor_by_name('ypred:0')
        opt1=sess.graph.get_operation_by_name('opt1')
        opt2=sess.graph.get_operation_by_name('opt2')
        opt3=sess.graph.get_operation_by_name('opt3')
        
    
        for count in range(nIter):
        
            fd={x:xtrain,y:ytrain.reshape((-1,1)),xboh:xbohtrain,xcoh:xcohtrain}
        
            # again, this is terrible - fix soon!
            if count<100: sess.run(opt1,feed_dict=fd)
            else:
                if count<200: sess.run(opt2,feed_dict=fd)
                else: sess.run(opt3,feed_dict=fd)
                
            if count%numDisp==0: 
                print("Iteration #"+str(count)+": error "+str(sess.run(loss,feed_dict=fd)))
                #print("embedvec row 1: "+str(sess.run(brandEmbeddings)[0,:]))
        
                            
    
        # Prediction
        ################
            
        fd={x:xtest,y:ytest.reshape((-1,1)),xboh:xbohtest,xcoh:xcohtest}
        yproba=sess.run(ypred,feed_dict=fd)  #,feed_dict={x:xtest,y:ytest.reshape((-1,1))})
        ypredicted=np.array(yproba>.5,dtype=int)
    
        precisions.append(precision_score(ytest,ypredicted))
        recalls.append(recall_score(ytest,ypredicted))
        accuracies.append(accuracy_score(ytest,ypredicted))
        f1s.append(f1_score(ytest,ypredicted))
        aucs.append(roc_auc_score(ytest,yproba))

        #beds=sess.run(brandEmbeddings)
        #ceds=sess.run(catEmbeddings)
    
        sess.close()
        
    scoresdf=pd.DataFrame([precisions,recalls,accuracies,f1s,aucs])
    scoresdf.columns=['Split #'+str(tmp) for tmp in range(num_splits)]
    scoresdf.index=['Precision','Recall','Accuracy','F1','AUC']
    scoresdf['Mean']=scoresdf.mean(axis=1)
    scoresdf=scoresdf.T
    
    return scoresdf






###########################################################
# Feature Extraction Stuff
###########################################################
