import pandas as pd
import numpy as np

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
df2=pd.read_csv("/home/wlwoon/Dropbox/chicken_work/impersuasion_stuff/withSalesData/sales_order_line_timestamped.csv")
df['day']=df.date.apply(gadata_date2days)
df2['day']=df2.transaction_date.apply(salesdata_date2days)



# Functions
#############

# Feature extraction
def getFeatures(featThreshold=150,featLen=30,labelLen=20,nanReplace=-10):


    # Preparing to extract features
    inFeatRange=set(list(range(featThreshold-featLen,featThreshold)))
    labelRange=set(list(range(featThreshold,featThreshold+labelLen)))
    
    featdf=df[df.day.isin(inFeatRange)]
    targetdf=df2[df2.day.isin(labelRange)]
    targetdf=targetdf[targetdf.customer_id.isin(set(featdf.user_id))]
    
    featdf.replace(np.nan,nanReplace,inplace=True)

    dg=featdf.groupby('user_id')
    
    # Feature extraction

    # x (inputs) - xdf is the dataframe containing all possible features
    # in next cell only interesting ones are extracted
    xdf=dg.hits.count()
    xdf=pd.concat((xdf,dg.country.first()),axis=1)
    xdf['malaysia']=(xdf.country=='Malaysia').astype(int)
    xdf['malsing']=(xdf.country.isin(['Malaysia','Singapore'])).astype(int)
    xdf['meanDay']=dg.day.mean()
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

    # y (targets)
    ydf=pd.DataFrame(xdf.index.isin(set(targetdf.customer_id)).astype(int),index=xdf.index)

    return xdf,ydf
