import pandas as pd
import numpy as np
from collections import Iterable

# Utility stuff
##################

# Vars
monthDict={'01':0,'02':31,'03':59,'04':90,'05':120,'06':151,'07':181,'08':212,'09':243,'10':273,'11':304,'12':334} # Only for non-leap years OK
intMonthDict={int(tmp):monthDict[tmp] for tmp in monthDict.keys()}

# Functions
date2days=lambda month,day: monthDict[month]+int(day)
gadata_date2days=lambda datestr: date2days(datestr[4:6],datestr[6:8]) 
salesdata_date2days=lambda datestr: date2days(datestr[5:7],datestr[8:10])


# Loading in global data
###############################

# Loading in data and adding day info
df=pd.read_csv("gadata.csv",dtype={'date':str})
df2=pd.read_csv("../withSalesData/sales_order_line_timestamped.csv")
df['day']=df.date.apply(gadata_date2days)
df2['day']=df2.transaction_date.apply(salesdata_date2days)



# Functions
#############

# Feature extraction
def getFeatures(featThresholds=150,featureWindow=30,labelWindow=20,nanReplace=-10):

    if not isinstance(featThresholds,Iterable):
        featThresholds=[featThresholds]

        
    for featThreshold in featThresholds:

        # Preparing to extract features
        inFeatRange=set(list(range(featThreshold-featureWindow,featThreshold)))
        labelRange=set(list(range(featThreshold,featThreshold+labelWindow)))
    
        featdf=df[df.day.isin(inFeatRange)]
        targetdf=df2[df2.day.isin(labelRange)]
        targetdf=targetdf[targetdf.customer_id.isin(set(featdf.user_id))]
    
        featdf.replace(np.nan,nanReplace,inplace=True)
        
        dg=featdf.groupby('user_id')
    
        # Feature extraction

        # x (inputs) - xdf_ is the dataframe containing all possible features
        # in next cell only interesting ones are extracted
        xdf_=dg.hits.count()
        xdf_=pd.concat((xdf_,dg.country.first()),axis=1)
        xdf_['malaysia']=(xdf_.country=='Malaysia').astype(int)
        xdf_['malsing']=(xdf_.country.isin(['Malaysia','Singapore'])).astype(int)
        xdf_['meanDay']=dg.day.mean()-(featThreshold-featureWindow)
        xdf_['stdDay']=dg.day.std().replace(np.nan,0)
        xdf_['meanDuration']=dg.sessionDuration.mean()
        xdf_['maxDuration']=dg.sessionDuration.max()
        xdf_['meanPurchase']=dg.noPurchases.mean()
        xdf_['maxPurchase']=dg.noPurchases.max()
        xdf_['recency']=dg.day.apply(lambda tmp: np.min(featThreshold-tmp))
        xdf_['userType']=(dg.userType.first()=='Returning Visitor').astype(int)
        xdf_['bigCity']=(dg.city.first().isin(['Kuala Lumpur','Johor Bahru','Petaling Jaya'])).astype(int)
        xdf_['city']=dg.city.first()
        xdf_['bigRegion']=(dg.region.first().isin(['Federal Territory of Kuala Lumpur','Selangor','Johor','Penang']))
        xdf_['region']=dg.region.first()
        xdf_['maxSessions']=dg.sessions.max()
        xdf_['meanSessions']=dg.sessions.mean()
       
        # y (targets)
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
