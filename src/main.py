import numpy as np
import pandas as pd
from data_cleaning import fix_age
from data_cleaning import impute_nulls_cases


def main():
    cases = pd.read_csv("../data/cases_test.csv")
    #ages_fixed = fix_age(cases)
    imputed_nulls = impute_nulls_cases(cases)
    print(imputed_nulls)


if __name__ == "__main__":
    main()
