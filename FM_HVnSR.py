import yfinance as yf
import pandas as pd
import datetime
import numpy as np
import plotly.graph_objs as go

def rv_sr(tick,name):

    end = datetime.datetime.now()
    start = datetime.datetime.now() - datetime.timedelta(days=3650)
    data = yf.download(tick,start=start,end=end,progress=False,auto_adjust=True)

    data1=data.iloc[-365:]

    TRADING_DAYS = 252

    returns = np.log(data1['Close']/data1['Close'].shift(1))
    returns.fillna(0, inplace=True)
    volatility = returns.rolling(window=TRADING_DAYS).std()*np.sqrt(TRADING_DAYS)

    sharpe_ratio = returns.mean()/volatility

    print("------------------------------------- Analysis Based on Last 1 Year Data -------------------------------------")
    print("------------------------------------- {} -------------------------------------\n".format(name))

    print('Last 1 year Volatility Statistics')
    print('-------------------------------------')
    print('Average:',round(volatility.mean(),4))
    print('Highest:',round(volatility.max(),4))
    print('Lowest:',round(volatility.min(),4))

    print('\n\n\n')


    print('Last 1 year Sharp Ratio Statistics')
    print('-------------------------------------')
    print('Average:',round(sharpe_ratio.mean(),4))
    print('Highest:',round(sharpe_ratio.max(),4))
    print('Lowest:',round(sharpe_ratio.min(),4))

    data=go.Scatter(
            y=volatility.values[251:], x=volatility.index[251:],mode='lines',
            name='Annualized Volatility',
            marker=dict(size=14,color='green',line=dict(width=2)))

    layout=go.Layout(
    title='Annualized Volatility',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Value'),
    hovermode='closest')

    figure=go.Figure(data=data,layout=layout)
    figure.show()

    data=go.Scatter(
            y=sharpe_ratio.values[251:], x=sharpe_ratio.index[251:],mode='lines',
            name='Sharpe ratio with the annualized volatility',
            marker=dict(size=14,color='green',line=dict(width=2)))

    layout=go.Layout(
    title='Sharpe ratio with the annualized volatility',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Value'),
    hovermode='closest')

    figure=go.Figure(data=data,layout=layout)
    figure.show()
