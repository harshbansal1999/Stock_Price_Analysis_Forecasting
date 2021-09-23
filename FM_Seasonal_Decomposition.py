import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
import math
plt.style.use(['ggplot'])
sns.set()

def seasonal_decompose(tick,name):

    print("------------------------------------- Analysis Based on Last 1 Year Data -------------------------------------")
    print("------------------------------------- {} -------------------------------------\n".format(name))

    end = datetime.datetime.now()
    start = datetime.datetime.now() - datetime.timedelta(days=365)
    df = yf.download(tick,start=start,end=end,progress=False,auto_adjust=True)

    def adf_test(x):
        indices = ['Test Statistic', 'p-value','# of Lags Used', '# of Observations Used']
        adf_test = adfuller(x, autolag='AIC')
        results = pd.Series(adf_test[0:4], index=indices)

        for key, value in adf_test[4].items():
            results[f'Critical Value ({key})'] = value

        return results

    x=adf_test(df.Close)

    if(x['p-value']<0.05):
        print('Data is Stationairy')
    else:
        print('Data is not Stationairy')


    df['TREND'] = np.nan

    for i in range(6,df['Close'].size-6):
        df['TREND'][i] = np.round(
            df['Close'][i - 6] * (1.0 / 24) +
            (   df['Close'][i - 5] +
                df['Close'][i - 4] +
                df['Close'][i - 3] +
                df['Close'][i - 2] +
                df['Close'][i - 1] +
                df['Close'][i] +
                df['Close'][i + 1] +
                df['Close'][i + 2] +
                df['Close'][i + 3] +
                df['Close'][i + 4] +
                df['Close'][i + 5] ) * (1.0 / 12) + df['Close'][i + 6] * (1.0 / 24))

    #plot the trend component
    fig = plt.figure(figsize=(15,8))
    fig.canvas.set_window_title("{} Trend".format(name))
    fig.suptitle('TREND')
    df['TREND'].plot()
    #plt.show()

    df['SEASONALITY AND NOISE'] = df['Close']/df['TREND']
    fig = plt.figure(figsize=(15,5))
    fig.canvas.set_window_title("{} SEASONALITY AND NOISE".format(name))
    fig.suptitle('SEASONALITY and NOISE components')
    df['SEASONALITY AND NOISE'].plot()
    #plt.show()


    #first add a month column
    df['MONTH'] = df.index.strftime('%m').astype(np.int)
    #initialize the month based dictionaries to store the running total of the month wise  seasonal sums and counts
    average_seasonal_values = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0}
    average_seasonal_value_counts = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0}
    #calculate the sums and counts
    for i in range(0, df['SEASONALITY AND NOISE'].size):
        if math.isnan(df['SEASONALITY AND NOISE'][i]) is False:
            average_seasonal_values[df['MONTH'][i]] = average_seasonal_values[df['MONTH'][i]] + df['SEASONALITY AND NOISE'][i]
            average_seasonal_value_counts[df['MONTH'][i]] = average_seasonal_value_counts[df['MONTH'][i]] + 1
    #calculate the average seasonal component for each month
    for i in range(1, 13):
        average_seasonal_values[i] = average_seasonal_values[i] / average_seasonal_value_counts[i]
    #create a new column in the data frame and fill it with the value of the average seasonal component for the corresponding month
    df['SEASONALITY'] = np.nan
    for i in range(0, df['SEASONALITY AND NOISE'].size):
        if math.isnan(df['SEASONALITY AND NOISE'][i]) is False:
            df['SEASONALITY'][i] = average_seasonal_values[df['MONTH'][i]]

    #plot the seasonal component
    fig = plt.figure(figsize=(15,5))
    fig.canvas.set_window_title("{} Seasonal Component".format(name))
    fig.suptitle('The \'pure\' SEASONAL component')
    #plt.ylim(0, 1.3)
    df['SEASONALITY'].plot()
    #plt.show()

    df['NOISE'] = df['SEASONALITY AND NOISE']/df['SEASONALITY']
    #plot the seasonal component
    fig = plt.figure(figsize=(15,5))
    fig.canvas.set_window_title("{} Noise".format(name))
    fig.suptitle('The NOISE component')
    #plt.ylim(0, 1.3)
    df['NOISE'].plot()
    plt.show()
