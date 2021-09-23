import yfinance as yf
import pandas as pd
import datetime
import requests

def data_fetch():

    def get_symbol(symbol):
        url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)

        result = requests.get(url).json()

        for x in result['ResultSet']['Result']:
            if x['symbol'] == symbol:
                return x['name']

    print('\n\n')
    print('------------------------------------------------------------------------------------------------------------')
    tick=str(input("Enter the Company Ticker:"))
    tick_=tick.upper()+".NS"

    end = datetime.datetime.now()
    start = datetime.datetime.now() - datetime.timedelta(days=200)
    data = yf.download(tick_,start=start,end=end,progress=False,auto_adjust=True)

    if(data.shape[0]!=0):
        data.index=data.index.tz_localize(None)
        data=data.fillna(0)
        data=data.astype(int)

        df =  yf.download(tick_,period='1d', interval='2m',progress=False,auto_adjust=True)
        df=df[df.index.day==df.index[-1].day]
        df.index=df.index.tz_localize(None)
        df=df.fillna(0)
        df=df.astype(int)
        company = get_symbol(tick_)

    else:
        print('No record. Please try again')
        data_fetch()

    return data,df,company,tick_
