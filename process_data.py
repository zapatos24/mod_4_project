import pandas as pd
import numpy as np
import datetime

# changes data from wide format to long format
class Process_data:
    def __init__(self, filename):
        raw_data = pd.io.parsers.read_csv('zillow_data.csv', dtype={'RegionName': 'str', 'Price': 'int'})
        melted = melt_data(raw_data)

    def melt_data(df):
        melted = pd.melt(df, id_vars=['RegionName', 'City', 'State', 'Metro', 'CountyName', 'RegionID', 'SizeRank'], var_name='time')
        melted.rename(columns = {'RegionName':'Zipcode', 'value':'Price', 'time':'Time'}, inplace=True)
        melted['Time'] = pd.to_datetime(melted['Time'], infer_datetime_format=True)
        melted.set_index('Time', inplace=True)
        melted.drop('RegionID', inplace=True, axis=1)
        return melted

    # backfills (estimates) a zipcode's historical price according to avg % change in housing market
    def backfill_price(group):
        prices = []
        for i in range(len(group)):
            curr_price = group.iloc[-1-i]['Price']
            if np.isnan(curr_price):
                base_price = prices[i-1]
                growth_rate = monthly_medians.iloc[-1-i] / monthly_medians.iloc[-i]
                prices.append(int(base_price * growth_rate))
            else:
                prices.append(int(curr_price))
        return reversed(prices)

    # 'zillow_data.csv' contains average monthly housing prices in USA by zipcode

    monthly_medians = melted.groupby('Time').agg({'Price':'median'})
    groups = melted.groupby('Zipcode')
    data = []
    for group in groups:
        prices = backfill_price(group)
        group['Prices'] = prices
        data.append(group)

    f = open(filename, 'a')
    for df in l1:
        df.to_csv(f)
    f.close()

    data.to_csv('data.csv')
