import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import numpy as np
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from scipy.stats import norm
from statsmodels.stats.diagnostic import lilliefors
import yfinance as yf
plt.style.use(['ggplot'])
sns.set()

def stylized_facts(tick,name):

    print("------------------------------------- Analysis Based on Last 2 month Data -------------------------------------")
    print("------------------------------------- {} -------------------------------------\n".format(name))

    end = datetime.datetime.now()
    start = datetime.datetime.now() - datetime.timedelta(days=60)
    df = yf.download(tick,start=start,end=end,progress=False,auto_adjust=True)

    def gaussian(df):

        print('------------------ Non-Gaussian distribution of returns ------------------\n')

        df['simple_rtn'] = df.Close.pct_change()
        df['log_rtn'] = np.log(df.Close/df.Close.shift(1))
        df.fillna(0,inplace=True)

        print('---------- Descriptive Statistics ----------')
        print('Range of dates:', min(df.index.date), '-', max(df.index.date))
        print('Number of observations:', df.shape[0])
        print(f'Mean: {df.log_rtn.mean():.4f}')
        print(f'Median: {df.log_rtn.median():.4f}')
        print(f'Min: {df.log_rtn.min():.4f}')
        print(f'Max: {df.log_rtn.max():.4f}')
        print(f'Standard Deviation: {df.log_rtn.std():.4f}')
        print(f'Skewness: {df.log_rtn.skew():.4f}')
        print(f'Kurtosis: {df.log_rtn.kurtosis():.4f}')

        my_data = norm.rvs()
        p_value=lilliefors(df.log_rtn)[0]
        if(p_value>0.05):
            print('Normal Distribution (',p_value,')')
        else:
            print('Non-Normal Distribution (',p_value,')')

        r_range = np.linspace(min(df.log_rtn), max(df.log_rtn), num=1000)
        mu = df.log_rtn.mean()
        sigma = df.log_rtn.std()
        norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        fig.canvas.set_window_title("{} Distribution".format(name))

        # histogram
        sns.distplot(df.log_rtn, kde=False, norm_hist=True, ax=ax[0])
        ax[0].set_title('Distribution of MSFT returns', fontsize=16)
        ax[0].plot(r_range, norm_pdf, 'g', lw=2,label=f'N({mu:.2f}, {sigma**2:.4f})')
        ax[0].legend(loc='upper left');

        # Q-Q plot
        qq = sm.qqplot(df.log_rtn.values, line='s', ax=ax[1])
        ax[1].set_title('Q-Q plot', fontsize = 16)
        print('--------------------------------------------------------\n\n')
        plt.show()

    def volatility_clustering(df):
        print('------------------ Volatility Clustering ------------------\n')

        df['Returns']=df.Close-df.Close.shift(1)
        df['log_rtn'] = np.log(df.Close/df.Close.shift(1))
        x1=df[df['Returns']>=0].shape[0]
        x2=df[df['Returns']<0].shape[0]


        print('# of Positive Return:',df[df['Returns']>=0].shape[0])
        print('# of Negative Return:',df[df['Returns']<0].shape[0])
        print('\n')

        if(abs(x1-x2)>7):
            print('High Volatility')
        else:
            print('Normal Volatility')

        data=go.Scatter(
                    y=df['log_rtn'], x=df.index,mode='lines',
                    name='Log Returns',
                    marker=dict(size=14,color='green',line=dict(width=2)))

        layout=go.Layout(
        title='Log Returns',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Value'),
        hovermode='closest')

        figure=go.Figure(data=data,layout=layout)
        figure.show()
        print('--------------------------------------------------------\n\n')


    def auto_corr(df):
        print('------------------ Absence of autocorrelation in returns ------------------\n')

        N_LAGS = df.shape[0]-1
        SIGNIFICANCE_LEVEL = 0.05
        df['Returns']=df.Close-df.Close.shift(1)
        acf = smt.graphics.plot_acf(df.Close, lags=N_LAGS,alpha=SIGNIFICANCE_LEVEL)
        print('--------------------------------------------------------\n\n')
        plt.show()

    def auto_corr_sq(df):

        print('--------- Small and decreasing autocorrelation in squared/absolute returns ---------\n')

        df['log_rtn'] = np.log(df.Close/df.Close.shift(1))
        N_LAGS = df.shape[0]-1
        SIGNIFICANCE_LEVEL = 0.05
        df.fillna(0,inplace=True)

        fig, ax = plt.subplots(2, 1, figsize=(12, 10))
        smt.graphics.plot_acf(df.log_rtn ** 2, lags=N_LAGS,alpha=SIGNIFICANCE_LEVEL, ax = ax[0])
        ax[0].set(title='Autocorrelation Plots', ylabel='Squared Returns')
        smt.graphics.plot_acf(np.abs(df.log_rtn), lags=N_LAGS,alpha=SIGNIFICANCE_LEVEL, ax = ax[1])
        ax[1].set(ylabel='Absolute Returns', xlabel='Lag')
        print('--------------------------------------------------------\n\n')
        plt.show()


    gaussian(df)
    volatility_clustering(df)
    auto_corr(df)
    auto_corr_sq(df)
