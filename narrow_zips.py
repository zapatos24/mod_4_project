import csv
import pandas as pd
import numpy as np
import datetime
from ProcessData import ProcessData
import pandas as pd
import numpy as np
import statsmodels as sm
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
# import seaborn as sns
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
#
from sklearn.metrics import mean_squared_error
# from dateutil.relativedelta import relativedelta
# from fbprophet import Prophet
import datetime as datetime
import warnings

def evaluate_arima_model(time_series, arima_order):
    # prepare training dataset
    time_series_filtered = time_series.squeeze()
    train_size = int(len(time_series_filtered) * 0.66)
    train = time_series_filtered[0:train_size]
    test =  time_series_filtered[train_size:]
    history = list(train)

    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error

def evaluate_models(time_series, p_values, d_values, q_values):
    time_series = time_series.astype('float32')
    best_score, best_cfg = 100000000.0, None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(time_series, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return best_cfg


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

def get_top_zipcodes(zipcode_stats):
    sorted_by_pct_change = sorted(zipcode_stats, key=lambda x: x[3], reverse=True)
    top_pct_changes = sorted_by_pct_change[: int(len(sorted_by_pct_change) * pct_change_filter)]
    sorted_by_std = sorted(top_pct_changes, key=lambda x: x[4])
    lowest_std_devs = sorted_by_std[: int(len(sorted_by_std) * std_dev_filter)]
    top_zipcodes = sorted(lowest_std_devs, key=lambda x: x[3], reverse=True)
    return top_zipcodes

def compare_top_zipcodes(top_zipcodes):
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

def dataframes_to_model(zipcodes_to_model):
    dfs = []
    for zipcode_to_model in zipcodes_to_model:
        index = zipcode_to_model[-1]
        df = dataframes[index]
        dfs.append(df)
    return dfs

def get_best_zipcodes(dataframes, years_to_forecast = 5):
    zipcode_stats = get_stats(dataframes)
    top_zipcodes = get_top_zipcodes(zipcode_stats)
    zipcodes_to_model = compare_top_zipcodes(top_zipcodes)
    to_model = dataframes_to_model(zipcodes_to_model)
    periods = 12 * years_to_forecast
    forecasts = []
    for df in to_model:
        model = auto_arima(df.Price, trace=True, error_action='ignore', suppress_warnings=True)
        model.fit(df.Price)
        forecast = model.predict(periods)
        forecasts.append(forecast)
    return forecasts

# start date
pct_change_filter = .3
std_dev_filter = .33
cols = ['Zipcode', 'City', 'State', 'Pct_change', 'Std_dev', 'df_index']

start_date = '2011-01-01'
end_of_collection = '2018-04-01'
years_to_forecast = 5
processed_data = ProcessData(start_date)
monthly_medians = processed_data.monthly_medians
dataframes = processed_data.dataframes
zipcodes = processed_data.zipcodes
forecasts = get_best_zipcodes(dataframes)

growths = []
for forecast in forecasts:
    pct_growth = (forecast[-1] - forecast[0]) * 100 / forecast[0]
    growths.append(pct_growth)
growths
zipped = zip(growths, zipcodes)
sorted(zipped, key = lambda x: x[0], reverse=True)
