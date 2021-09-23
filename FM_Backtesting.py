import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import yfinance as yf

def backtesting(tick,name):

    print("------------------------------------- Analysis Based on Last 1 Month months Data -------------------------------------")
    print("------------------------------------- {} -------------------------------------\n".format(name))

    end = datetime.datetime.now()
    start = datetime.datetime.now() - datetime.timedelta(days=30)
    data = yf.download(tick,start=start,end=end,progress=False,auto_adjust=True)

    df =  yf.download(tick,period='1d', interval='2m',progress=False,auto_adjust=True)

    data['SMA20'] = data['Close'].rolling(20).mean()
    data['SMA30'] = data['Close'].rolling(30).mean()

    df1=df[(df['Close']>data['SMA30'].mean())&(df['Close']>data['SMA20'].mean())]
    df2=df[(df['Close']>data['SMA20'].mean())&(df['Close']<data['SMA30'].mean())]

    plt1=go.Scatter(
                y=df['Close'], x=df.index,mode='lines',
                name='Price',
                marker=dict(size=14,color='green',line=dict(width=2)))

    plt2=go.Scatter(
                y=df1['Close'], x=df1.index,
                name='Price goes above 20 & 30 Day Average',mode='markers')

    plt3=go.Scatter(
                y=df2['Close'], x=df2.index,
                name='Best time to buy at',mode='markers')

    plt=[plt1,plt2,plt3]
    layout=go.Layout(
    title='{} Backtesting Strategy'.format(name),
    xaxis=dict(title='Time'),
    yaxis=dict(title='Price'),
    hovermode='closest')
    figure=go.Figure(data=plt,layout=layout)
    figure.add_hline(y=data['SMA20'].mean(),line_color="red",line_dash="dash")
    figure.add_hline(y=data['SMA30'].mean(),line_color="blue",line_dash="dash")
    figure.show()
