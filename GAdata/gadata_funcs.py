import pandas as pd
import numpy as np
from collections import Iterable
from os.path import join
from sklearn.externals import joblib

# Utility stuff
##################

# Vars
monthDict={'01':0,'02':31,'03':59,'04':90,'05':120,'06':151,'07':181,'08':212,'09':243,'10':273,'11':304,'12':334} # Only for non-leap years OK
intMonthDict={int(tmp):monthDict[tmp] for tmp in monthDict.keys()}




# Functions
#############

date2days=lambda year,month,day: (int(year)-2017)*365+monthDict[month]+int(day)
gadata_date2days=lambda datestr: date2days(datestr[:4],datestr[4:6],datestr[6:8]) 
salesdata_date2days=lambda datestr: date2days(datestr[:4],datestr[5:7],datestr[8:10])


def LoadGlobalData(gaPath='.',salesDataPath='../withSalesData'):
    '''
    Loading in data and adding day info

    *Only required when training models*
    ------------------------------------

    '''

    global df,df2
    df=pd.read_csv(join(gaPath,"gadata.csv"),dtype={'date':str})
    df2=pd.read_csv(join(salesDataPath,"sales_order_line_timestamped.csv"))
    df['day']=df.date.apply(gadata_date2days)
    df2['day']=df2.transaction_date.apply(salesdata_date2days)


def ExtractFeaturesGA(featdf,featThreshold=-1,nanReplace=-10,featureWindow=26):
    '''
    Feature Extraction
    ------------------
    featdf is the window of Google Analytics events
    for which features are to be extracted

    def ExtractFeaturesGA(featdf,featThreshold=-1,nanReplace=-10):
    featdf - dataframe containing relevant chunk of GA data (in the format extracted from gadata.csv" 
    featThreshold - the day (day 1 -> Jan 1 2017) which is considered "now"
    nanReplace - replace nans.. best not to tamper ;-)
    '''

    # Prepping the data
    if featThreshold==-1:
        featThreshold=featdf.day.max()

    featdf=featdf.replace(np.nan,nanReplace)
    try:
        featdf=featdf[featdf['day']>featThreshold-featureWindow]
    except:
        featdf['day']=featdf.date.apply(gadata_date2days)
        featdf=featdf[featdf['day']>featThreshold-featureWindow]
        
    dg=featdf.groupby('user_id')
    
    # Feature extraction
    # x (inputs) - xdf_ is the dataframe containing all possible features
    # in next cell only interesting ones are extracted
    xdf=dg.hits.count()
    xdf=pd.concat((xdf,dg.country.first()),axis=1)
    xdf['malaysia']=(xdf.country=='Malaysia').astype(int)
    xdf['malsing']=(xdf.country.isin(['Malaysia','Singapore'])).astype(int)
    xdf['meanDay']=dg.day.mean() #-(featThreshold-featureWindow)
    xdf['stdDay']=dg.day.std().replace(np.nan,0)
    xdf['meanDuration']=dg.sessionDuration.mean()
    xdf['maxDuration']=dg.sessionDuration.max()
    xdf['meanPurchase']=dg.noPurchases.mean()
    xdf['maxPurchase']=dg.noPurchases.max()
    xdf['recency']=dg.day.apply(lambda tmp: np.min(featThreshold-tmp))
    xdf['userType']=(dg.userType.first()=='Returning Visitor').astype(int)
    xdf['bigCity']=(dg.city.first().isin(['Kuala Lumpur','Johor Bahru','Petaling Jaya'])).astype(int)
    xdf['city']=dg.city.first()
    xdf['bigRegion']=(dg.region.first().isin(['Federal Territory of Kuala Lumpur','Selangor','Johor','Penang']))
    xdf['region']=dg.region.first()
    xdf['maxSessions']=dg.sessions.max()
    xdf['meanSessions']=dg.sessions.mean()
    xdf['meanSessionsToTransaction']=dg['ga:sessionsToTransaction'].mean()

    return xdf

# Feature extraction
def GetTrainingData(featThresholds=[80,150],featureWindow=26,labelWindow=38,nanReplace=-10):

    if not isinstance(featThresholds,Iterable):
        featThresholds=[featThresholds]

        
    for featThreshold in featThresholds:

        # Preparing to extract features
        inFeatRange=set(list(range(featThreshold-featureWindow,featThreshold)))
        labelRange=set(list(range(featThreshold,featThreshold+labelWindow)))
    
        featdf=df[df.day.isin(inFeatRange)]
        targetdf=df2[df2.day.isin(labelRange)]
        targetdf=targetdf[targetdf.customer_id.isin(set(featdf.user_id))]

        xdf_=ExtractFeaturesGA(featdf,featThreshold,nanReplace)
        ydf_=pd.DataFrame(xdf_.index.isin(set(targetdf.customer_id)).astype(int),index=xdf_.index)

        try:
            xdf_=xdf_[np.logical_not(xdf_.index.isin(xdf.index))]
            ydf_=ydf_[np.logical_not(ydf_.index.isin(xdf.index))]
            xdf=pd.concat((xdf,xdf_),axis=0)
            ydf=pd.concat((ydf,ydf_),axis=0)
        except NameError:
            xdf=xdf_
            ydf=ydf_
        
    return xdf,ydf


def ConversionPrediction(inFile,dateStr='',modelFile='currentModel.jl'):
    '''Perform conversion prediction using GA data

    Parameters:

    - inFile is the name of the file containing the GA data for which
     predictions are to be generated Note: it should be structured the
     same as the gadata.csv file.  Alternatively, it can be a
     dataframe, but the 'date' field needs to be a string

    - dateStr is a string formatted as 'yyyymmdd'. if '' is provided
      (the default), then the latest date in the file is provided

    - modelfile is a joblib dump file containing a dict with two keys:
     'model' for the model, and 'features' with the list of features
     used

    '''
    
    # Prepping data and loading model
    if type(inFile)==str:
        indf=pd.read_csv(inFile,dtype={'date':str})
    else:
        indf=inFile

    if dateStr=='':
        featThreshold=-1
    else:
        featThreshold=gadata_date2days(dateStr)
        
    mod=joblib.load(modelFile)
    xdf=ExtractFeaturesGA(indf,featThreshold,featureWindow=mod['featureWindow'])
    x=xdf[mod['features']].values
    
    # Generating predictions
    hits=mod['model'].predict(x).astype(bool)
    return xdf.index[hits].values.tolist()
    
