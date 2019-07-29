import csv
import pandas as pd
import numpy as np
import datetime

start_date = '2010-01-01'
pct_change_filter = .01
std_dev_filter = .33
filename = 'data.csv'
cols = ['Zipcode', 'City', 'State', 'Pct_change', 'Std_dev', 'Beta']

def diff_month(date1, date2):
    return abs((date1.year - date2.year) * 12 + date1.month - date2.month)

def prep_data(filename, start_year):
    raw_data = pd.io.parsers.read_csv(filename, dtype={'Zipcode': 'str'}, index_col=[0])
    raw_data.index = pd.to_datetime(raw_data.index)
    data = raw_data.loc[start_date:]
    return data

def get_stats(all_zipcodes):
    zipcode_stats = []
    rows_per_zip = int(len(data) / data.Zipcode.nunique())
    for index, curr_zipcode in enumerate(all_zipcodes):
        start = index * rows_per_zip
        zipcode_data = data.iloc[start: start + rows_per_zip]
        price_diff = zipcode_data.iloc[-1].Price - zipcode_data.iloc[0].Price
        pct_change = (price_diff / zipcode_data.iloc[0].Price) * 100
        std_dev = np.std(zipcode_data.Price)
        variance = std_dev ** 2
        covariance = np.cov(zipcode_data.Price, monthly_medians.values)[0, 1]
        beta = covariance/variance
        curr_zip_stats = [zipcode_data.iloc[0].Zipcode, zipcode_data.iloc[0].City,
                          zipcode_data.iloc[0].State, pct_change, std_dev, beta]
        zipcode_stats.append(curr_zip_stats)
    return zipcode_stats

# data.csv is output from process_data.py
data = prep_data(filename, start_year)
all_zipcodes = data.Zipcode.unique()
monthly_medians = data.groupby('Time').agg({'Price': 'median'}).loc[start_date:]
zipcode_stats = get_stats(all_zipcodes)

def get_top_zipcodes(zipcode_stats):
    sorted_by_pct_change = sorted(zipcode_stats, key=lambda x: x[3], reverse=True)
    top_pct_changes = sorted_by_pct_change[: int(len(sorted_by_pct_change) * pct_change_filter)]
    sorted_by_std = sorted(top_pct_changes, key=lambda x: x[4])
    lowest_std_devs = sorted_by_std[: int(len(sorted_by_std) * std_dev_filter)]
    top_zipcodes = sorted(lowest_std_devs, key=lambda x: x[3], reverse=True)
    return pd.DataFrame(top_zipcodes, columns=cols)

top_zipcodes = get_top_zipcodes(zipcode_stats)
top_zipcodes
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


# 2004-2010
zipcodes_to_model = compare_top_zipcodes(top_zipcodes)
zipcodes_to_model
