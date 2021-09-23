import yfinance as yf
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt

def rsi(tick,name):

    print("------------------------------------- Analysis Based on Today's Data -------------------------------------")
    print("------------------------------------- {} -------------------------------------\n".format(name))

    def RSI(series, n=14):
        # Get the difference in price from previous price
        series = series.copy().diff()

        # Get upwards and downwards gains
        up, down = series.copy(), series.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        # Calculate the exponential weighted values
        roll_up1 = up.ewm(span=n,min_periods=0,adjust=False,ignore_na=False).mean()
        roll_down1 = down.abs().ewm(span=n,min_periods=0,adjust=False,ignore_na=False).mean()

        # Calculate the RSI based on EWMA
        RS1 = roll_up1 / roll_down1
        RSI1 = 100.0 - (100.0 / (1.0 + RS1))
        return(RSI1)


    df =  yf.download(tick,period='14d', interval='30m',progress=False,auto_adjust=True)
    df=df.loc[:,['Close']]

    n = 14
    df['RSI'] = RSI(df['Close'], n)

    print('Total Oversold Situations:',df[df['RSI']<=30].shape[0])
    print('Total Oversbought Situations:',df[df['RSI']>70].shape[0])

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.canvas.set_window_title("{} RSI".format(name))

    plt.plot(df.index, df['RSI'], color='red')
    plt.title('{} - '.format(name) + str(n) + ' Bar Period')
    plt.grid()
    ax.legend(loc='upper left', frameon=False)

    # overbought
    plt.axhline(70, color='gray', linewidth=2, linestyle='-.' )
    # oversold
    plt.axhline(30, color='gray', linewidth=2, linestyle='-.' )

    # Get second axis
    ax2 = ax.twinx()
    plt.plot(df.index,  df['Close'], label='Price',color='blue')
    ax2.legend(loc='upper right', frameon=False)
    plt.show()
