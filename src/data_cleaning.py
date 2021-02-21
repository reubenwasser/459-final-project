import numpy as np
import pandas as pd


def fix_age(df):
    '''
    Reduces different formats in age column for cases_train/test.csv to a standard integer format  
    '''
    #drop nulls
    age_filter = df[~df['age'].astype(str).str.isdigit()]['age'].dropna()

    #For age ranges (ex 12-24)
    age_range = age_filter[age_filter.str.contains(r'\d+\s*-\s*\d+', regex=True)]
    low_age = age_range.replace(r'(\d+)(\s*-\s*)(\d+)', r"\1", regex=True).astype(int)
    high_age = age_range.replace(r'(\d+)(\s*-\s*)(\d+)', r"\3", regex=True).astype(int)
    median_age = (high_age + low_age) // 2

    #For age with dash(ex. 12-)
    age_dash = age_filter[age_filter.str.contains(r'^\d+[-+]\w*$', regex=True)]
    age_formatted = age_dash.replace(r'(\d+)([-+]\w*)', r"\1", regex=True)

    # For age in months (ex 12 month)
    age_months = age_filter[age_filter.str.contains(r'\d+ months*', regex=True)]
    age_months = age_months.replace(r'(\d+)( months*)', r"\1", regex=True)
    age_years = age_months.astype(int) // 12

    # For age in second
    # age_seconds = age_filter[age_filter.str.contains(r'\d+s', regex=True)]
    # print(age_seconds)
    #cast series as strings
    df.update(median_age.astype(str))
    df.update(age_formatted.astype(str))
    df.update(age_years.astype(str))

    # Convert NaNs to -1 in order to convert column
    df["age"] = df["age"].fillna(-1).astype(float).astype(int)

def impute_nulls_cases(df):
    '''
    Imputes null values for all applicable attributes in cases_test/train.csv
    '''

    df["sex"] = df["sex"].fillna("unknown")
    df["latitude"] = df["latitude"].fillna(0)
    df["longitude"] = df["longitude"].fillna("0")
    df["additional_information"] = df["additional_information"].fillna("none")
    df["source"] = df["source"].fillna("unknown")
    df["outcome"] = df["outcome"].fillna("unknown")



