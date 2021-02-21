import numpy as np
import pandas as pd
from data_cleaning import fix_age
from data_cleaning import impute_nulls_cases


def main():
    cases_train = pd.read_csv("../data/cases_train.csv")
    cases_test = pd.read_csv("../data/cases_test.csv")
    fix_age(cases_test)
    fix_age(cases_train)
    impute_nulls_cases(cases_train)
    impute_nulls_cases(cases_test)



if __name__ == "__main__":
    main()
