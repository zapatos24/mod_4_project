import csv
import numpy as np
import datetime
import pandas as pd
# import statsmodels as sm
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
# from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import datetime as datetime
import warnings
import sys
import os
from ProcessData import ProcessData


# What does this thing do?
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def get_stats(dataframes):
    zipcode_stats = []
    for index, df in enumerate(dataframes):
        price_diff = df.iloc[-1].Price - df.iloc[0].Price
        pct_change = (price_diff / df.iloc[0].Price) * 100
        std_dev = np.std(df.Price)
        variance = std_dev ** 2
        # covariance = np.cov(df.Price, monthly_medians.values)[0][1]
        # add beta to curr_zip_stats
        # beta = covariance/variance
        curr_zip_stats = [df.iloc[0].Zipcode, df.iloc[0].City,
                          df.iloc[0].State, pct_change, std_dev, index]
        zipcode_stats.append(curr_zip_stats)
    return zipcode_stats


def get_top_zipcodes(zipcode_stats, pct_change_filter, std_dev_filter):
    sorted_by_pct_change = sorted(zipcode_stats, key=lambda x: x[3], reverse=True)
    top_pct_changes = sorted_by_pct_change[: int(len(sorted_by_pct_change) * pct_change_filter)]
    sorted_by_std = sorted(top_pct_changes, key=lambda x: x[4])
    lowest_std_devs = sorted_by_std[: int(len(sorted_by_std) * std_dev_filter)]
    top_zipcodes = sorted(lowest_std_devs, key=lambda x: x[3], reverse=True)
    return top_zipcodes


def top_from_each_state(top_zipcodes):
    top_states = set()
    for curr_zipcode in top_zipcodes:
        top_states.add(curr_zipcode[2])

    already_used = dict(zip(top_states, [False for _ in range(len(top_states))]))
    zipcodes_to_model = []
    #  ensures only 1 zipcode per state
    for curr_zipcode in top_zipcodes:
        state = curr_zipcode[2]
        if not already_used[state]:
            zipcodes_to_model.append(curr_zipcode)
            already_used[state] = True
    return zipcodes_to_model


def dataframes_to_model(zipcodes_to_model, dataframes):
    dfs = []
    for zipcode_to_model in zipcodes_to_model:
        index = zipcode_to_model[-1]
        df = dataframes[index]
        dfs.append(df)
    return dfs


def get_best_zipcodes(dataframes, pct_change_filter, std_dev_filter, different_states, years_to_forecast=5):
    zipcode_stats = get_stats(dataframes)
    top_zipcodes = get_top_zipcodes(zipcode_stats, pct_change_filter, std_dev_filter)

    if different_states:
        top_zipcodes = top_from_each_state(top_zipcodes)
    to_model = dataframes_to_model(top_zipcodes, dataframes)
    periods = 12 * years_to_forecast
    # blockPrint()
    forecasts = []
    for df in to_model:
        model = auto_arima(df.Price, trace=True, error_action='ignore', suppress_warnings=True)
        model.fit(df.Price, return_all=False)
        forecast = model.predict(int(periods))
        forecasts.append(forecast)
    # enablePrint()
    return forecasts, top_zipcodes


def get_final_zipcodes(start_date='2011-01-01', years_to_forecast=5, pct_change_filter=.005, std_dev_filter=1, different_states=False, zipcode_count=5):
    processed_data = ProcessData(start_date)
    dataframes = processed_data.dataframes
    forecasts, zipcodes_to_model = get_best_zipcodes(dataframes, pct_change_filter, std_dev_filter, different_states, years_to_forecast)
    growths = []
    for forecast in forecasts:
        pct_growth = (forecast[-1] - forecast[0]) * 100 / forecast[0]
        growths.append(pct_growth)
    zipped = zip(growths, zipcodes_to_model)  # this is the error
    top_zipcodes = sorted(zipped, key=lambda x: x[0], reverse=True)
    print("\n\nHere are the top five zipcodes you should invest in over a {} year period!\n\nProjected Growth".format(years_to_forecast))
    for zipcode in top_zipcodes[:zipcode_count]:
        print(zipcode[0], zipcode[1][:3])


# start date
# datelist =
#

# get_final_zipcodes()

# use the datelist, return an entire dataframe with forecasts
# create all 3 dataframes
# listg of dataframes thast correspond to each zipcode 10 years_to_forecast
# zipcode key, list of % changes, largest_time_frame dataframe
