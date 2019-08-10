import statsmodels as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings


def evaluate_arima_model(time_series, arima_order):
    # prepare training dataset
    time_series_post_crash = time_series['2010':].squeeze()
    train_size = int(len(time_series_post_crash) * 0.66)
    train, test = time_series_post_crash[0:train_size], time_series_post_crash[train_size:]
    history = [x for x in train]

    # make predictions
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


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = 100000000.0, None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return best_cfg


# warnings.filterwarnings("ignore")
# p_values = range(0, 4)
# d_values = range(0, 3)
# q_values = range(0, 3)
# best_post_crash_arima = []
# for index in top_indexes[:3]:
#     print('Index:', index, 'Zipcode:', zipcodes[index])
#     pdq = evaluate_models(time_series[index], p_values, d_values, q_values)
#     print()
#     best_post_crash_arima.append({'index': index, 'zipcode': zipcodes[index], 'pdq': pdq})
# # %%
# best_post_crash_arima
