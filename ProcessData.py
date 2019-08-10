import pandas as pd
import numpy as np
import datetime

# changes data from wide format to long format


class ProcessData:
    def __init__(self, start_date):
        raw_data = pd.io.parsers.read_csv('zillow_data.csv', dtype={'RegionName': 'str', 'Price': 'int'})
        melted = self.melt_data(raw_data)
        self.monthly_medians = melted.groupby('Time').agg({'Price': 'median'})
        melted = melted.loc[start_date:]
        dataframes = []
        zipcodes = []
        monthly_medians = []
        for name, group in melted.groupby('Zipcode'):
            if group.Price.isna().sum() >= 1:
                continue
            else:
                dataframes.append(group)
                zipcodes.append(group.iloc[0].Zipcode)
        self.dataframes = dataframes
        self.zipcodes = zipcodes
        self.get_historical_data()

    def melt_data(self, df):
        melted = pd.melt(df, id_vars=['RegionName', 'City', 'State',
                                      'Metro', 'CountyName', 'RegionID', 'SizeRank'], var_name='time')
        melted.rename(columns={'RegionName': 'Zipcode',
                               'value': 'Price', 'time': 'Time'}, inplace=True)
        melted['Time'] = pd.to_datetime(
            melted['Time'], infer_datetime_format=True)
        melted.set_index('Time', inplace=True)
        melted.drop('RegionID', inplace=True, axis=1)
        return melted


    # backfills (estimates) a zipcode's historical price according to avg % change in housing market - not used
    def backfill_price(self, df):
        prices = []
        for i in range(len(df)):
            curr_price = df.iloc[-1-i]['Price']
            if np.isnan(curr_price):
                base_price = prices[i-1]
                growth_rate = self.monthly_medians.iloc[-1 - i] / self.monthly_medians.iloc[-i]
                prices.append(int(base_price * growth_rate))
            else:
                prices.append(int(curr_price))
        return prices[::-1]


    def get_historical_data(self):
        raw_data = pd.io.parsers.read_csv('zillow_data.csv', dtype={'RegionName': 'str', 'Price': 'int'})
        melted = self.melt_data(raw_data)
        self.all_monthly_medians = melted.groupby('Time').agg({'Price':'median'})
        data = dict()
        for name, group in melted.groupby('Zipcode'):
            if group.Price.isna().sum() >= 1:
                continue
            else:
                price_diff = group.iloc[-1].Price - group.iloc[0].Price
                pct_change = (price_diff / group.iloc[0].Price) * 100
                curr_data = dict(df=group, pct_change=pct_change)
                data.update(dict(name=curr_data))
        self.all_zipcodes_dict = data
>>>>>>> origin/master
