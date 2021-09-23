import yfinance as yf
import pandas as pd
import datetime
import wikipedia as wp
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from bs4 import BeautifulSoup


def score(df):
    val1=df['High'].mean()/df['Low'].mean()
    val2=df[df['Negative Return']==0].shape[0]/df.shape[0]
    val3=(df[df['Negative Return']==0]['Change Returns'].mean()-df[df['Negative Return']==1]['Change Returns'].mean())
    val4=df['Returns Total'][-1]

    sum_=0
    for i in range(len(df)):
        open_=df['Open'][i]
        close_=df['Close'][i]
        val=(close_-open_)
        sum_+=val

    val5=abs(sum_)/len(df)
    val=val1+val2+val3+val4+val5

    return val

def sensex():

    html = wp.page("List of BSE SENSEX companies").html().encode("UTF-8")
    df = pd.read_html(html)[0]
    tickers=list(df['Symbol'])
    tickers=[x[:-3] for x in tickers]
    cmp_name=list(df['Companies'])

    end = datetime.datetime.now()
    start = datetime.datetime.now() - datetime.timedelta(days=100)

    final=pd.DataFrame(columns=['Company','Ticker','Score'])
    i=0

    for tick in tickers:
        tick = tick.upper()
        tick_ = tick+'.NS'
        data = yf.download(tick_,start=start,end=end,progress=False,auto_adjust=True)
        data=data.iloc[-7:]
        data['Returns(Price)']=data['Close'].diff()
        data['Returns Total']=data['Returns(Price)'].cumsum()
        data['Negative Return']=data['Returns(Price)'].apply(lambda x:1 if x<0 else 0)
        data['Change Returns']=data['Close'].pct_change()

        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(data)
        data.loc[:,:] = scaled_values

        val=score(data)

        final.loc[len(final)]=[cmp_name[i],tick,val]
        i=i+1

    final=final.sort_values(by='Score',ascending=False)
    final=final.reset_index().drop('index',axis=1)

    d=go.Bar(
            x=list(final['Company'].values),
            y=list(final['Score'].values),
            name='Sensex')

    layout=go.Layout(
    title='Sensex',
    xaxis=dict(title='Week'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest')
    figure=go.Figure(data=d,layout=layout)
    figure.show()

def nifty():

    URL = "https://en.wikipedia.org/wiki/NIFTY_50"

    nifty=pd.DataFrame(columns=['Company','Ticker'])

    res = requests.get(URL).text
    soup = BeautifulSoup(res,'lxml')
    for items in soup.find('table', class_='wikitable').find_all('tr')[1::1]:
        l=[]
        data = items.find_all(['th','td'])
        l.append(data[0].text)
        l.append(data[1].text)
        nifty.loc[len(nifty)]=l

    nifty['Score']=0.0
    end = datetime.datetime.now()
    start = datetime.datetime.now() - datetime.timedelta(days=100)

    for i in range(len(nifty)):
        tick=nifty['Ticker'][i]
        tick_ = tick.upper()
        data = yf.download(tick_,start=start,end=end,progress=False,auto_adjust=True)
        data=data.iloc[-7:]
        data['Returns(Price)']=data['Close'].diff()
        data['Returns Total']=data['Returns(Price)'].cumsum()
        data['Negative Return']=data['Returns(Price)'].apply(lambda x:1 if x<0 else 0)
        data['Change Returns']=data['Close'].pct_change()

        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(data)
        data.loc[:,:] = scaled_values

        val=score(data)
        nifty['Score'][i]=val

    nifty=nifty.sort_values(by='Score',ascending=False)
    nifty=nifty.reset_index().drop('index',axis=1)

    d=go.Bar(
            x=list(nifty['Company'].values),
            y=list(nifty['Score'].values),
            name='Nifty Score')

    layout=go.Layout(
    title='Nifty',
    xaxis=dict(title='Week'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest')
    figure=go.Figure(data=d,layout=layout)
    figure.show()
