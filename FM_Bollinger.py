import yfinance as yf
import pandas as pd
import datetime
import numpy as np
import plotly.graph_objs as go

def bollinger(tick,name):

    print("------------------------------------- Analysis Based on Today's Data -------------------------------------")
    print("------------------------------------- {} -------------------------------------\n".format(name))

    df =  yf.download(tick,period='1d', interval='2m',progress=False,auto_adjust=True)

    period = 20

    #Simple Moving Average
    df['SMA'] = df['Close'].rolling(window=period).mean()

    #Standard Deviation
    df['STD'] = df['Close'].rolling(window=period).std()

    #Upper Band
    df['Upper'] = df['SMA'] + (df['STD']*2)

    #Lower Band
    df['Lower'] = df['SMA'] - (df['STD']*2)

    def get_signal(data):
        buy_signal = []
        sell_signal = []

        for i in range(len(data['Close'])):
            if(data['Close'][i] > data['Upper'][i]):
                buy_signal.append(np.nan)
                sell_signal.append(data['Close'][i])
            elif(data['Close'][i] < data['Lower'][i]):
                sell_signal.append(np.nan)
                buy_signal.append(data['Close'][i])
            else:
                buy_signal.append(np.nan)
                sell_signal.append(np.nan)

        return (buy_signal,sell_signal)

    new_df = df[period-1:]
    new_df['Buy'] = get_signal(new_df)[0]
    new_df['Sell'] = get_signal(new_df)[1]

    plt1=go.Scatter(
            y=new_df['Close'], x=new_df.index,mode='lines',
            name='Close Price',
            marker=dict(size=14,color='black',line=dict(width=2)))
    plt2=go.Scatter(
                y=new_df['SMA'], x=new_df.index,mode='lines',
                name='SMA',
                marker=dict(size=14,color='blue',line=dict(width=2)))
    plt3=go.Scatter(
                y=new_df['Buy'], x=new_df.index,mode='markers',
                name='Buy',
                marker=dict(size=10,color='green',line=dict(width=2)))
    plt4=go.Scatter(
                y=new_df['Sell'], x=new_df.index,mode='markers',
                    name='Sell',
                     marker=dict(size=10,color='red',line=dict(width=2)))

    plt=[plt1,plt2,plt3,plt4]

    layout=go.Layout(
    title='{} Bollinger Band'.format(name),
    xaxis=dict(title='Time'),
    yaxis=dict(title='Price'),
    hovermode='closest')
    figure=go.Figure(data=plt,layout=layout)
    figure.add_trace(go.Scatter(x=new_df.index, y=new_df['Upper'], line = dict(color='rgba(0,0,0,0)')))
    figure.add_trace(go.Scatter(x=new_df.index, y=new_df['Lower'], line = dict(color='rgba(0,0,0,0)'),fill='tonexty'))
    #figure.update_layout(showlegend=False)
    figure.show()
