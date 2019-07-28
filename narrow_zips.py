import csv
import pandas as pd
import numpy as np
import datetime

start_year = 2012
filename = 'data.csv'
pct_change_filter = .01
std_dev_filter = .33

def prep_data(filename, start_year):
    raw_data = pd.io.parsers.read_csv(filename, dtype={'Zipcode': 'str'}, index_col=[0])
    raw_data.index = pd.to_datetime(raw_data.index)
    start_date = datetime.datetime(start_year, 1, 1)
    data = raw_data.loc['2012-01-01':]
    return data

def get_stats(all_zipcodes):
    zipcode_stats = []
    for curr_zipcode in all_zipcodes:
        zipcode_data = data[data.Zipcode == curr_zipcode]
        price_diff = zipcode_data.iloc[-1].price - zipcode_data.iloc[0].price
        pct_change = price_diff / (zipcode_data.iloc[0].price * 100)
        std_dev = np.std(zipcode_data.price)
        curr_zip_stats = [zipcode_data.iloc[0].Zipcode, zipcode_data.iloc[0].City,
                          zipcode_data.iloc[0].State, pct_change, std_dev]
        zipcode_stats.append(curr_zip_stats)
    return zipcode_stats

# data.csv is output from process_data.py

data = prep_data(filename, start_year)
all_zipcodes = data.Zipcode.unique()
zipcode_stats = get_stats(all_zipcodes)

def get_top_zipcodes(zipcode_stats):
    sorted_by_pct_change = sorted(zipcode_stats, key=lambda x: x[3], reverse=True)
    top_pct_changes = sorted_by_pct_change[: int(len(sorted_by_pct_change) * pct_change_filter)]
    sorted_by_std = sorted(top_pct_changes, key=lambda x: x[4])
    lowest_std_devs = sorted_by_std[: int(len(sorted_by_std) * std_dev_filter)]
    top_zipcodes = sorted(lowest_std_devs, key=lambda x: x[3], reverse=True)
    cols = ['Zipcode', 'City', 'State', 'Pct_change', 'Std_dev']
    return pd.DataFrame(top_zipcodes, columns=cols)

top_zipcodes = get_top_zipcodes(zipcode_stats)

def compare_top_zipcodes(top_zipcodes):
    top_states = top_zipcodes.State.unique()
    already_used = dict(zip(top_states, [False for _ in range(len(top_states))]))
    zipcodes_to_model = []
    for row in top_zipcodes.iterrows():
        if not already_used[row[1].State]:
            zipcodes_to_model.append(row[1])
            already_used[row[1].State] = True
    return pd.DataFrame(zipcodes_to_model, columns=cols)
    
zipcodes_to_model = compare_top_zipcodes(top_zipcodes)
