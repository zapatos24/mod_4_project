import csv
import pandas as pd
import numpy as np
import datetime
from ProcessData import ProcessData


start_date = '2010-01-01'
pct_change_filter = .01
std_dev_filter = .33
cols = ['Zipcode', 'City', 'State', 'Pct_change', 'Std_dev', 'Beta']

def filter_data(dataframes):
    filtered_dfs = []
    for dataframe in dataframes:
        filtered_df = dataframe.loc[start_date:]
        filtered_dfs.append(filtered_df)
    return filtered_dfs

def diff_month(date1, date2):
    return abs((date1.year - date2.year) * 12 + date1.month - date2.month)

def get_stats(dataframes):
    zipcode_stats = []
    for df in dataframes:
        price_diff = df.iloc[-1].Price - df.iloc[0].Price
        pct_change = (price_diff / df.iloc[0].Price) * 100
        std_dev = np.std(df.Price)
        variance = std_dev ** 2
        covariance = np.cov(df.Price, monthly_medians.values)[0][1]
        beta = covariance/variance
        curr_zip_stats = [df.iloc[0].Zipcode, df.iloc[0].City,
                          df.iloc[0].State, pct_change, std_dev, beta]
        zipcode_stats.append(curr_zip_stats)
    return zipcode_stats

def get_top_zipcodes(zipcode_stats):
    sorted_by_pct_change = sorted(zipcode_stats, key=lambda x: x[3], reverse=True)
    top_pct_changes = sorted_by_pct_change[: int(len(sorted_by_pct_change) * pct_change_filter)]
    sorted_by_std = sorted(top_pct_changes, key=lambda x: x[4])
    lowest_std_devs = sorted_by_std[: int(len(sorted_by_std) * std_dev_filter)]
    top_zipcodes = sorted(lowest_std_devs, key=lambda x: x[3], reverse=True)
    return pd.DataFrame(top_zipcodes, columns=cols)

def compare_top_zipcodes(top_zipcodes):
    top_states = top_zipcodes.State.unique()
    already_used = dict(zip(top_states, [False for _ in range(len(top_states))]))
    zipcodes_to_model = []
    #  ensures only 1 zipcode per state
    for row in top_zipcodes.iterrows():
        if not already_used[row[1].State]:
            zipcodes_to_model.append(row[1])
            already_used[row[1].State] = True
    return pd.DataFrame(zipcodes_to_model, columns=cols)


# zipcode_stats = get_stats(dataframes)
# top_zipcodes = get_top_zipcodes(zipcode_stats)
# zipcodes_to_model = compare_top_zipcodes(top_zipcodes)
