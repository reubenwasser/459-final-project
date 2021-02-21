import pandas as pd
import numpy as np
import math


import sys
import os

#transform

def transform(dataLocation):
    temp = dataLocation[dataLocation.Country_Region == "US"]
    #create population column
    dataLocation["Population"] = np.nan
    tempDict = dataLocation.to_dict(orient='records')

    column_names = list(dataLocation.columns) 

    #create new dataframe
    dataLocationTrans = pd.DataFrame(columns = column_names)

    for i in tempDict:
        location = i["Combined_Key"]
        split = location.split(", ")
        if(len(split) >= 3 and split[2] == "US"):
            i["Combined_Key"] = split[1] + ", " + "United States"
            if(i["Incidence_Rate"] == 0 or math.isnan(i["Incidence_Rate"])):
                i["Population"] = 0
            else:
                i["Population"] = round((i["Confirmed"] * 100000) / i["Incidence_Rate"])       
        else:
            continue
            
    temp = 0
    for i in tempDict:
        dataLocationTrans.loc[temp] = i
        temp = temp +1

    return dataLocationTrans

    
