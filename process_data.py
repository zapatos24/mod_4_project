import pandas as pd
import numpy as np
import datetime

# changes data from wide format to long format
def melt_data(df):
    melted = pd.melt(df, id_vars=['RegionName', 'City', 'State', 'Metro', 'CountyName', 'RegionID', 'SizeRank'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted.set_index('time', inplace=True)
    melted.drop('RegionID', inplace=True, axis=1)
    melted.rename(columns = {'RegionName':'Zipcode', 'value':'price'}, inplace=True)
    return melted

# backfills (estimates) a zipcode's historical price according to avg % change in housing market
def backfill_price(group):
    for i in range(len(group) - 1):
        if np.isnan(group.iloc[-1-i]['price']):
            base_price = group.iloc[-i]['price']
            growth_rate = monthly_medians.iloc[-1-i] / monthly_medians.iloc[-i]
            group.iloc[-1-i]['price'] = (base_price * growth_rate)

# returns dataframe with every zipcode fully backfilled
def process_data(data):
    groups = data.groupby('Zipcode')
    groups.apply(backfill_price)
    dataframes = []
    for name, group in groups:
        dataframes.append(group)
    return pd.concat(dataframes)

# 'zillow_data.csv' contains average monthly housing prices in USA by zipcode
raw_data = pd.io.parsers.read_csv('zillow_data.csv', dtype={'RegionName': 'str'})
data = melt_data(raw_data)
monthly_medians = data.groupby('time').agg({'price':'median'})
data.sort_values(by=['Zipcode', 'time'], ascending=True, inplace=True)

# uncomment next two lines to process and save data for analysis (takes ~30 minutes)
# all_data = process_data(data)
# all_data.to_csv('data.csv')
