import pandas as pd

# Utility Vars
monthDict={'01':0,'02':31,'03':59,'04':90,'05':120,'06':151,'07':181,'08':212,'09':243,'10':273,'11':304,'12':334} # Only for non-leap years OK
intMonthDict={int(tmp):monthDict[tmp] for tmp in monthDict.keys()}

#{1:0,2:31,3:59,4:90,5:120,6:151,7:181,8:212,9:243,10:273,11:304,12:334} # Only for non-leap years OK

# Utility Functions
date2days=lambda month,day: monthDict[month]+int(day)
gadata_date2days=lambda datestr: date2days(datestr[4:6],datestr[6:8]) 
salesdata_date2days=lambda datestr: date2days(datestr[5:7],datestr[8:10])
