import numpy as np
import pandas as pd
from data_cleaning import fix_age
from data_cleaning import impute_nulls_cases
from helper1 import transform
from helper2 import merge

import sys
import os


def main():
    cases_train = pd.read_csv("../data/cases_train.csv")
    cases_test = pd.read_csv("../data/cases_test.csv")
    cases_location = pd.read_csv("../data/location.csv")

    #1.2
    
    fix_age(cases_test)
    fix_age(cases_train)
    impute_nulls_cases(cases_train)
    impute_nulls_cases(cases_test)
    
    
    #1.3
    
    cases_train = cases_train[ cases_train.age < 100]
    
    
    #1.4
    dataLocationTrans = transform(cases_location)
    dataLocationTrans.to_csv("../results/location_transformed.csv", index = False)
    
    #1.5
    print(len(cases_train))
    print(len(cases_test))
    mergeTestDF = merge( dataLocationTrans,cases_test)
    mergeTrainDF = merge( dataLocationTrans,cases_train )
    mergeTestDF.to_csv("../results/cases_test_processed.csv", index = False)
    mergeTrainDF.to_csv("../results/cases_train_processed.csv", index = False)
    print(len(mergeTrainDF))
    print(len(mergeTestDF))
    
if __name__ == "__main__":
    main()
