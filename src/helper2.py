import pandas as pd
import numpy as np
import math


import sys
import os


def checkDict(dict2,value):
    for (key,val) in dict2.items():
        if str(key) == str(value):
            return True
    return False

def checkComma(string_name):
    for element in range(0, len(string_name)):
        if (string_name[element] == ','):
            return True
    
    return False


def merge(dataLocationTrans,dataCase):
    dataLocationTransDict = dataLocationTrans['Combined_Key'].value_counts().to_dict()
    dataLocationTransDict = {key:val for key, val in dataLocationTransDict.items() if val != 1}

    combDF = dataCase

    combDF["Confirmed"] = np.nan
    combDF["Deaths"] = np.nan
    combDF["Recovered"] = np.nan
    combDF["Active"] = np.nan
    combDF["Incidence_Rate"] = np.nan
    combDF["Case-Fatality_Ratio"] = np.nan



    for index, row in dataLocationTrans.iterrows():
        
        location = str(row['Combined_Key'])
        if(checkDict(dataLocationTransDict,location)):
            
            sumConfirmed = dataLocationTrans.loc[dataLocationTrans['Combined_Key'] == location, 'Confirmed'].sum()
            sumDeath = dataLocationTrans.loc[dataLocationTrans['Combined_Key'] == location, 'Deaths'].sum()
            sumRecov = dataLocationTrans.loc[dataLocationTrans['Combined_Key'] == location, 'Recovered'].sum()
            sumActive = dataLocationTrans.loc[dataLocationTrans['Combined_Key'] == location, 'Active'].sum()
            sumPop = dataLocationTrans.loc[dataLocationTrans['Combined_Key'] == location, 'Population'].sum()
            sumCFR = 0
            if(sumConfirmed != 0):
                sumCFR = (sumDeath/sumConfirmed) * 100
            
            sumIr = 0
            if(sumPop != 0):
                sumIR = (sumConfirmed/sumPop) * 100000

            split = location.split(",")
            temp2 = split[1]
            if(temp2[0] == " "):
                split[1] = split[1][1:]

            indexTest = combDF.index[ (combDF['country'] == split[1]) & (combDF['province'] == split[0])].tolist()
            
            for i in indexTest:
                combDF.at[ i,'Case-Fatality_Ratio'] = sumCFR
                combDF.at[ i,'Confirmed'] = sumConfirmed
                combDF.at[ i,'Deaths'] = sumDeath
                combDF.at[ i,'Recovered'] = sumRecov
                combDF.at[ i,'Active'] = sumActive
                combDF.at[ i,'Incidence_Rate'] = sumIR
            
        else:
            
            if(checkComma(location) == False):
                #only country is in location
                indexTest = combDF.index[ combDF['country'] == location].tolist()
                for i in indexTest:
                    combDF.at[ i,'Case-Fatality_Ratio'] = dataLocationTrans.loc[dataLocationTrans.Combined_Key == location,'Case-Fatality_Ratio'].tolist()[0]
                    combDF.at[ i,'Confirmed'] = dataLocationTrans.loc[dataLocationTrans.Combined_Key == location,'Confirmed'].tolist()[0]
                    combDF.at[ i,'Deaths'] = dataLocationTrans.loc[dataLocationTrans.Combined_Key == location,'Deaths'].tolist()[0]
                    combDF.at[ i,'Recovered'] = dataLocationTrans.loc[dataLocationTrans.Combined_Key == location,'Recovered'].tolist()[0]
                    combDF.at[ i,'Active'] = dataLocationTrans.loc[dataLocationTrans.Combined_Key == location,'Active'].tolist()[0]
                    combDF.at[ i,'Incidence_Rate'] = dataLocationTrans.loc[dataLocationTrans.Combined_Key == location,'Incidence_Rate'].tolist()[0]

            else:
                #location = location.replace(",", ", ")
                split = location.split(",")
                temp2 = split[1]
                if(temp2[0] == " "):
                    split[1] = split[1][1:]
                
                
                indexTest = combDF.index[ (combDF['country'] == split[1]) & (combDF['province'] == split[0])].tolist()
                for i in indexTest:
                    combDF.at[ i,'Case-Fatality_Ratio'] = dataLocationTrans.loc[dataLocationTrans.Combined_Key == location,'Case-Fatality_Ratio'].tolist()[0]
                    combDF.at[ i,'Confirmed'] = dataLocationTrans.loc[dataLocationTrans.Combined_Key == location,'Confirmed'].tolist()[0]
                    combDF.at[ i,'Deaths'] = dataLocationTrans.loc[dataLocationTrans.Combined_Key == location,'Deaths'].tolist()[0]
                    combDF.at[ i,'Recovered'] = dataLocationTrans.loc[dataLocationTrans.Combined_Key == location,'Recovered'].tolist()[0]
                    combDF.at[ i,'Active'] = dataLocationTrans.loc[dataLocationTrans.Combined_Key == location,'Active'].tolist()[0]
                    combDF.at[ i,'Incidence_Rate'] = dataLocationTrans.loc[dataLocationTrans.Combined_Key == location,'Incidence_Rate'].tolist()[0]
                
    #drop values with missing location data -> ~8% of actual data            
    combDF  = combDF.dropna(how='any', subset=['Confirmed'])
    return combDF