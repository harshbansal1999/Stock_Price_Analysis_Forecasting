import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import pmdarima as pm

def time_series_model(tick,name):

    print("------------------------------------- Analysis Based on Today's Data -------------------------------------")
    print("------------------------------------- {} -------------------------------------\n".format(name))

    df =  yf.download(tick,period='1d', interval='1m',progress=False,auto_adjust=True)
    data=df.Close
    autocorrelation_plot(data)

    model = pm.auto_arima(data, error_action='ignore', suppress_warnings=True, seasonal=False, stepwise=False, approximation=False, n_jobs=-1)
    print(model.summary())
    plt.show()
